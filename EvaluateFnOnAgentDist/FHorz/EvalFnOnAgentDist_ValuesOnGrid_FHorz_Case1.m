function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, Parallel,simoptions)
% Parallel is an optional input.

if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
elseif isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
end

if ~exist('simoptions','var')
    simoptions=struct();
end

if isfield('simoptions','n_semiz') % If using semi-exogenous shocks
    n_z=[n_z,simoptions.n_semiz]; % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

%% This implementation is slightly inefficient when shocks are not age dependent, but speed loss is fairly trivial
if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
    simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
end

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

jointgrid_z=0;
if fieldexists_pi_z_J==1
    z_grid_J=simoptions.z_grid_J;
elseif fieldexists_ExogShockFn==1
    z_grid_J=zeros(sum(n_z),N_j);
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
        else
            [z_grid,~]=simoptions.ExogShockFn(jj);
        end
        z_grid_J(:,jj)=z_grid;
    end
else
    if all(size(z_grid)==[N_z,l_z]) && l_z>1  % joint grid (correlated z shocks) 
        jointgrid_z=1;
    else
        jointgrid_z=0;
        z_grid_J=repmat(z_grid,1,N_j);
    end
end
if Parallel==2 
    if jointgrid_z==0
        z_grid_J=gpuArray(z_grid_J);
    else %if jointgrid_z==1
        z_grid=gpuArray(z_grid);
    end
end

if isfield(simoptions,'n_e')
    % Because of how FnsToEvaluate works I can just get the e variables and then 'combine' them with z

    if isfield(simoptions,'EiidShockFn') % If using EiidShockFn then figure out the parameter names
        simoptions.EiidShockFnParamNames=getAnonymousFnInputNames(simoptions.EiidShockFn);
    end
    
    eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
    eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
    eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')
    
    N_e=prod(simoptions.n_e);
    l_e=length(simoptions.n_e);
    
    jointgrid_e=0;
    if fieldexists_pi_e_J==1
        e_grid_J=simoptions.e_grid_J;
    elseif fieldexists_EiidShockFn==1
        e_grid_J=zeros(sum(simoptions.n_e),N_j);
        for jj=1:N_j
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [e_grid,~]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [e_grid,~]=simoptions.EiidShockFn(jj);
            end
            e_grid_J(:,jj)=gather(e_grid);
        end
    else
        if all(size(simoptions.e_grid)==[N_e,l_e]) && l_e>1 % joint grid (correlated e shocks)
            jointgrid_e=1;
        else
            jointgrid_e=0;
            e_grid_J=repmat(simoptions.e_grid,1,N_j);
        end
    end
    
    % Now combine into z
    if jointgrid_e==0 && jointgrid_z==0
        if n_z(1)==0
            l_z=l_e;
            n_z=simoptions.n_e;
            z_grid_J=e_grid_J;
        else
            l_z=l_z+l_e;
            n_z=[n_z,simoptions.n_e];
            z_grid_J=[z_grid_J; e_grid_J];
        end
        jointgrids=0;
    elseif jointgrid_e==1 && jointgrid_z==1
        l_z=l_z+l_e;
        n_z=[n_z,simoptions.n_e];
        z_gridvals=[kron(ones(N_e,1),z_grid),kron(simoptions.e_grid,ones(N_z,1))];
        jointgrids=1;
        z_grid_J=[]; % This is just needed in case use parfor as Matlab otherwise throws an error that it cannot find it
    else
        error('Have not yet implemented a mix where only one of z and e uses joint-grids and the other does not. Email me and I will')
    end

    N_z=prod(n_z);
else
    jointgrids=0;
    if jointgrid_z==1
        jointgrids=1;
        z_gridvals=z_grid; % Note: Does not yet permit age-dependent joint grids
    end
end

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    elseif simoptions.keepoutputasmatrix==2
        FnsToEvaluateStruct=2;
    end
end

%%
if Parallel==2
    ValuesOnGrid=zeros(N_a*N_z,N_j,length(FnsToEvaluate),'gpuArray');
    
    PolicyValues=PolicyInd2Val_FHorz_Case1(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);
    for ff=1:length(FnsToEvaluate)        
        Values=nan(N_a*N_z,N_j,'gpuArray');
        for jj=1:N_j
            if jointgrids==0
                z_grid=z_grid_J(:,jj);
            else % jointgrids==1
                z_grid=z_gridvals;
            end
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
            end
            Values(:,jj)=reshape(EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{ff}, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,Parallel),[N_a*N_z,1]);
        end
        ValuesOnGrid(:,:,ff)=Values;
    end
    
elseif Parallel==0
    ValuesOnGrid=zeros(N_a*N_z,N_j,length(FnsToEvaluate));

    a_gridvals=CreateGridvals(n_a,a_grid,2);
    
    sizePolicyIndexes=size(PolicyIndexes);
    if sizePolicyIndexes(2:end)~=[N_a,N_z,N_j] % If not in vectorized form
        PolicyIndexes=reshape(PolicyIndexes,[sizePolicyIndexes(1),N_a,N_z,N_j]);
    end
    
    if jointgrids==1
        z_gridvals=num2cell(z_gridvals);
    end
    
    for ff=1:length(FnsToEvaluate)
        Values=zeros(N_a,N_z,N_j);
        if l_d==0
            for jj=1:N_j
                if jointgrids==0
                    z_grid=z_grid_J(:,jj);
                    z_gridvals=CreateGridvals(n_z,z_grid,2);
                % elseif  jointgrids==1
                %    z_gridvals is independent of age and already created above
                end
                
                [~, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
                if ~isempty(FnsToEvaluateParamNames(ff).Names)
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
                end
                for a_c=1:N_a
                    for z_c=1:N_z
                        % Includes check for cases in which no parameters are actually required
                        if isempty(FnsToEvaluateParamNames(ff).Names)
                             Values(a_c,z_c,jj)=FnsToEvaluate{ff}(aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:});
                        else
                             Values(a_c,z_c,jj)=FnsToEvaluate{ff}(aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end
            end
        else
            for jj=1:N_j
                if jointgrids==0
                    z_grid=z_grid_J(:,jj);
                    z_gridvals=CreateGridvals(n_z,z_grid,2);
                % elseif jointgrids==1
                %    z_gridvals is independent of age and already created above
                end
                
                [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
                if ~isempty(FnsToEvaluateParamNames(ff).Names)
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
                end
                for a_c=1:N_a
                    for z_c=1:N_z
                        % Includes check for cases in which no parameters are actually required
                        if isempty(FnsToEvaluateParamNames(ff).Names)
                             Values(a_c,z_c,jj)=FnsToEvaluate{ff}(d_gridvals{a_c+(z_c-1)*N_a,:},aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:});
                        else
                             Values(a_c,z_c,jj)=FnsToEvaluate{ff}(d_gridvals{a_c+(z_c-1)*N_a,:},aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end
            end
        end
        ValuesOnGrid(:,:,ff)=reshape(Values,[N_a*N_z,N_j,1]);
    end
    
else
    ValuesOnGrid=zeros(N_a*N_z,N_j,length(FnsToEvaluate));

    a_gridvals=CreateGridvals(n_a,a_grid,2);
    
    sizePolicyIndexes=size(PolicyIndexes);
    if ~all(sizePolicyIndexes(2:4)==[N_a,N_z,N_j]) % If not in vectorized form
        PolicyIndexes=reshape(PolicyIndexes,[sizePolicyIndexes(1),N_a,N_z,N_j]);
    end
    
    z_gridvals_J=cell(N_z,l_z,N_j);
    if jointgrids==1
        for jj=1:N_j
            z_gridvals_J(:,:,jj)=num2cell(z_gridvals);
        end
    else
        for jj=1:N_j
            z_grid=z_grid_J(:,jj);
            z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid,2);
        end
    end
    
    parfor ff=1:length(FnsToEvaluate) % Probably not the best level at which to implement the parfor, but will do for now
        FnToEvaluateParamsCell={}; % This line was just needed to stop Matlab complaining
        Values=zeros(N_a,N_z,N_j);
        if l_d==0
            for jj=1:N_j
                z_gridvals=z_gridvals_J(:,:,jj);

                [~, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
                if ~isempty(FnsToEvaluateParamNames(ff).Names)
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
                end
                for a_c=1:N_a
                    for z_c=1:N_z
                        % Includes check for cases in which no parameters are actually required
                        if isempty(FnsToEvaluateParamNames(ff).Names)
                             Values(a_c,z_c,jj)=FnsToEvaluate{ff}(aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:});
                        else
                             Values(a_c,z_c,jj)=FnsToEvaluate{ff}(aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end
            end
        else
            for jj=1:N_j
                z_gridvals=z_gridvals_J(:,:,jj);

                [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
                if ~isempty(FnsToEvaluateParamNames(ff).Names)
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
                end
                for a_c=1:N_a
                    for z_c=1:N_z
                        % Includes check for cases in which no parameters are actually required
                        if isempty(FnsToEvaluateParamNames(ff).Names)
                             Values(a_c,z_c,jj)=FnsToEvaluate{ff}(d_gridvals{a_c+(z_c-1)*N_a,:},aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:});
                        else
                             Values(a_c,z_c,jj)=FnsToEvaluate{ff}(d_gridvals{a_c+(z_c-1)*N_a,:},aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end
            end
        end
        ValuesOnGrid(:,:,ff)=reshape(Values,[N_a*N_z,N_j,1]);
    end
end

if FnsToEvaluateStruct==1
    ValuesOnGrid2=ValuesOnGrid;
    clear ValuesOnGrid
    ValuesOnGrid=struct();
    for ff=1:length(AggVarNames)
        ValuesOnGrid.(AggVarNames{ff})=reshape(ValuesOnGrid2(:,:,ff),[n_a,n_z,N_j]);
        % Change the ordering and size so that ProbDensityFns has same kind of shape as StationaryDist
    end
elseif FnsToEvaluateStruct==0
    % Change the ordering and size so that ProbDensityFns has same kind of
    % shape as StationaryDist, except first dimension indexes the
    % 'FnsToEvaluate'.
    ValuesOnGrid=permute(ValuesOnGrid,[3,1,2]);
    ValuesOnGrid=reshape(ValuesOnGrid,[length(FnsToEvaluate),n_a,n_z,N_j]);
elseif FnsToEvaluateStruct==2 % Just a rearranged version of FnsToEvaluateStruct=0 for use internally when length(FnsToEvaluate)==1
%     ValuesOnGrid=reshape(ValuesOnGrid,[N_a*N_z,N_j]);
    % The output is already in this shape anyway, so no need to actually reshape it at all
end

end
