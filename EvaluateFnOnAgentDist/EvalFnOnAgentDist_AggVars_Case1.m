function AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, simoptions, EntryExitParamNames, PolicyWhenExiting)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
%
% Parallel, simoptions, EntryExitParamNames and PolicyWhenExiting are
% optional inputs, later two only needed when using endogenous entry and
% endogenous exit.

%%
if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
elseif isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
end
simoptions.parallel=Parallel;

if ~exist('simoptions', 'var')
    simoptions=struct();
end

if n_d(1)==0
    l_d=0;
    N_d=0;
else
    l_d=length(n_d);
    N_d=prod(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);


l_daprime=size(PolicyIndexes,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);
if all(size(z_grid)==[sum(n_z),1]) % stacked-column
    z_gridvals=CreateGridvals(n_z,z_grid,1);
elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid 
    z_gridvals=z_grid;
end

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluate_copy=FnsToEvaluate; % keep a copy in case needed for conditional restrictions
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        FnsToEvalNames=simoptions.AggVarNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end


%% Check for functions that use value function and send these off to a subversion called EvalFnOnAgentDist_AggVars_Case1_withV()
if isfield(simoptions,'eval_valuefn')
    AggVarsExtra=struct();
    for ff=1:length(FnsToEvalNames)
        if ~isempty(FnsToEvaluateParamNames(ff).Names)
            if strcmp(FnsToEvaluateParamNames(ff).Names{1},simoptions.eval_valuefnname{1})
                % This function to evaluate has value function as an input
                % Send it to subfunction, and remove it from FnsToEvaluate
                tempFnsToEvaluateParamNames.Names={};
                if length(FnsToEvaluateParamNames(ff).Names)>1
                    tempFnsToEvaluateParamNames(ff).Names=FnsToEvaluateParamNames(ff).Names{2:end};
                end
                AggVarsExtra=EvalFnOnAgentDist_AggVars_Case1_withV(simoptions.eval_valuefn,StationaryDist, PolicyIndexes, {FnsToEvaluate{ff}}, {FnsToEvalNames{ff}}, Parameters, tempFnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, simoptions);
                FnsToEvaluate2=FnsToEvaluate; FnsToEvaluateParamNames2=FnsToEvaluateParamNames;
                clear FnsToEvaluateParamNames
                FnsToEvaluate={}; % Note: there may be no other FnsToEvaluate
                for gg=1:length(FnsToEvalNames)
                    if gg<ff
                        FnsToEvaluateParamNames(gg).Names=FnsToEvaluateParamNames2(gg).Names;
                        FnsToEvaluate{gg}=FnsToEvaluate2{gg};
                    elseif gg>ff
                        FnsToEvaluateParamNames(gg-1).Names=FnsToEvaluateParamNames2(gg).Names;
                        FnsToEvaluate{gg-1}=FnsToEvaluate2{gg};
                    end
                end
            end
        end
    end
    if isempty(FnsToEvaluate) % There are no FnsToEvaluate that do not have a dependence on the value function
        AggVars=AggVarsExtra;
        return
    end
end



%%
if isfield(simoptions,'statedependentparams')
    n_SDP=length(simoptions.statedependentparams.names);
    sdp=zeros(length(FnsToEvaluate),length(simoptions.statedependentparams.names));
    for ii=1:length(FnsToEvaluate)
        for jj=1:n_SDP
            for kk=1:length(FnsToEvaluateParamNames(ii).Names)
                if strcmp(simoptions.statedependentparams.names{jj},FnsToEvaluateParamNames(ii).Names{kk})
                    sdp(ii,jj)=1;
                    % Remove the statedependentparams from FnsToEvaluateParamNames
                    FnsToEvaluateParamNames(ii).Names=setdiff(FnsToEvaluateParamNames(ii).Names,simoptions.statedependentparams.names{jj});
                    % Set up the SDP variables
                end
            end
        end
    end
    if N_d>1
        n_full=[n_d,n_a,n_a,n_z];
    else
        n_full=[n_a,n_a,n_z];
    end
    
    % First state dependent parameter, get into form needed for the valuefn
    SDP1=Params.(simoptions.statedependentparams.names{1});
    SDP1_dims=simoptions.statedependentparams.dimensions.(simoptions.statedependentparams.names{1});
    %     simoptions.statedependentparams.dimensions.kmax=[3,4,5,6,7]; % The d,a & z variables (in VFI toolkit notation)
    temp=ones(1,l_d+l_a+l_a+l_z);
    for jj=1:max(SDP1_dims)
        [v,ind]=max(SDP1_dims==jj);
        if v==1
            temp(jj)=n_full(ind);
        end
    end
    if isscalar(SDP1)
        SDP1=SDP1*ones(temp);
    else
        SDP1=reshape(SDP1,temp);
    end
    if n_SDP>=2
        % Second state dependent parameter, get into form needed for the valuefn
        SDP2=Params.(simoptions.statedependentparams.names{2});
        SDP2_dims=simoptions.statedependentparams.dimensions.(simoptions.statedependentparams.names{2});
        temp=ones(1,l_d+l_a+l_a+l_z);
        for jj=1:max(SDP2_dims)
            [v,ind]=max(SDP2_dims==jj);
            if v==1
                temp(jj)=n_full(ind);
            end
        end
        if isscalar(SDP2)
            SDP2=SDP2*ones(temp);
        else
            SDP2=reshape(SDP2,temp);
        end
    end
    if n_SDP>=3
        % Third state dependent parameter, get into form needed for the valuefn
        SDP3=Params.(simoptions.statedependentparams.names{3});
        SDP3_dims=simoptions.statedependentparams.dimensions.(simoptions.statedependentparams.names{3});
        temp=ones(1,l_d+l_a+l_a+l_z);
        for jj=1:max(SDP3_dims)
            [v,ind]=max(SDP3_dims==jj);
            if v==1
                temp(jj)=n_full(ind);
            end
        end
        if isscalar(SDP3)
            SDP3=SDP3*ones(temp);
        else
            SDP3=reshape(SDP3,temp);
        end
    end
    
    % Currently SDP1 is on (n_d,n_aprime,n_a,n_z). It will be better
    % for EvalFnOnAgentDist_Grid_Case1_SDP if this is reduced to just
    % (n_a,n_z) using the Policy function.
    if l_d==0
        PolicyIndexes_sdp=reshape(PolicyIndexes(l_a,N_a,N_z));
        PolicyIndexes_sdp=permute(PolicyIndexes_sdp,[2,3,1]);
        if l_a==1
            aprime_ind=PolicyIndexes_sdp(:,:,1);
        elseif l_a==2
            aprime_ind=PolicyIndexes_sdp(:,:,1)+n_a(1)*(PolicyIndexes_sdp(:,:,2)-1);
        elseif l_a==3
            aprime_ind=PolicyIndexes_sdp(:,:,1)+n_a(1)*(PolicyIndexes_sdp(:,:,2)-1)+prod(n_a(1:2))*(PolicyIndexes_sdp(:,:,3)-1);
        elseif l_a==4
            aprime_ind=PolicyIndexes_sdp(:,:,1)+n_a(1)*(PolicyIndexes_sdp(:,:,2)-1)+prod(n_a(1:2))*(PolicyIndexes_sdp(:,:,3)-1)+prod(n_a(1:3))*(PolicyIndexes_sdp(:,:,4)-1);
        end
        aprime_ind=reshape(aprime_ind,[N_a*N_z,1]);
        a_ind=reshape((1:1:N_a)'*ones(1,N_z),[N_a*N_z,1]);
        z_ind=reshape(ones(N_a,1)*1:1:N_z,[N_a*N_z,1]);
        aprimeaz_ind=aprime_ind+N_a*(a_ind-1)+N_a*N_a*(z_ind-1);
        SDP1=SDP1(aprimeaz_ind);
        if n_SDP>=2
            SDP2=SDP2(aprimeaz_ind);
        end
        if n_SDP>=3
            SDP3=SDP3(aprimeaz_ind);
        end
    else
        PolicyIndexes_sdp=reshape(PolicyIndexes(l_d+l_a,N_a,N_z));
        PolicyIndexes_sdp=permute(PolicyIndexes_sdp,[2,3,1]);
        if l_d==1
            d_ind=PolicyIndexes_sdp(:,:,1);
        elseif l_d==2
            d_ind=PolicyIndexes_sdp(:,:,1)+n_d(1)*(PolicyIndexes_sdp(:,:,2)-1);
        elseif l_d==3
            d_ind=PolicyIndexes_sdp(:,:,1)+n_d(1)*(PolicyIndexes_sdp(:,:,2)-1)+prod(n_d(1:2))*(PolicyIndexes_sdp(:,:,3)-1);
        elseif l_d==4
            d_ind=PolicyIndexes_sdp(:,:,1)+n_d(1)*(PolicyIndexes_sdp(:,:,2)-1)+prod(n_d(1:2))*(PolicyIndexes_sdp(:,:,3)-1)+prod(n_d(1:3))*(PolicyIndexes_sdp(:,:,4)-1);
        end
        if l_a==1
            aprime_ind=PolicyIndexes_sdp(:,:,l_d+1);
        elseif l_a==2
            aprime_ind=PolicyIndexes_sdp(:,:,l_d+1)+n_a(1)*(PolicyIndexes_sdp(:,:,l_d+2)-1);
        elseif l_a==3
            aprime_ind=PolicyIndexes_sdp(:,:,l_d+1)+n_a(1)*(PolicyIndexes_sdp(:,:,l_d+2)-1)+prod(n_a(1:2))*(PolicyIndexes_sdp(:,:,l_d+3)-1);
        elseif l_a==4
            aprime_ind=PolicyIndexes_sdp(:,:,l_d+1)+n_a(1)*(PolicyIndexes_sdp(:,:,l_d+2)-1)+prod(n_a(1:2))*(PolicyIndexes_sdp(:,:,l_d+3)-1)+prod(n_a(1:3))*(PolicyIndexes_sdp(:,:,l_d+4)-1);
        end
        d_ind=reshape(d_ind,[N_a*N_z,1]);
        aprime_ind=reshape(aprime_ind,[N_a*N_z,1]);
        a_ind=reshape((1:1:N_a)'*ones(1,N_z),[N_a*N_z,1]);
        z_ind=reshape(ones(N_a,1)*1:1:N_z,[N_a*N_z,1]);
        daprimeaz_ind=d_ind+N_d*aprime_ind+N_d*N_a*(a_ind-1)+N_d*N_a*N_a*(z_ind-1);
        SDP1=SDP1(daprimeaz_ind);
        if n_SDP>=2
            SDP2=SDP2(daprimeaz_ind);
        end
        if n_SDP>=3
            SDP3=SDP3(daprimeaz_ind);
        end
    end
    
    
    if n_SDP>3
        fprintf('WARNING: currently only three state dependent parameters are allowed. If you have a need for more please email robertdkirkby@gmail.com and let me know (I can easily implement more if needed) \n')
        dbstack
        return
    end
end



%% Deal with Entry and/or Exit if approprate
if isstruct(StationaryDist)
    % Note: if you want the agent mass of the stationary distribution you have to call it 'agentmass'
    if ~isfield(simoptions,'endogenousexit')
        simoptions.endogenousexit=0;
    end
    if simoptions.endogenousexit~=2
        AggVars=EvalFnOnAgentDist_AggVars_Case1_Mass(StationaryDist.pdf,StationaryDist.mass, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, simoptions);
    elseif simoptions.endogenousexit==2
        exitprobabilities=CreateVectorFromParams(Parameters, simoptions.exitprobabilities);
        exitprobs=[1-sum(exitprobabilities),exitprobabilities];
        AggVars=EvalFnOnAgentDist_AggVars_Case1_Mass_MixExit(StationaryDist.pdf,StationaryDist.mass, PolicyIndexes, PolicyWhenExiting, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, exitprobs);
    end
    
    if FnsToEvaluateStruct==1
        % Change the output into a structure
        AggVars2=AggVars;
        clear AggVars
        AggVars=struct();
        %     AggVarNames=fieldnames(FnsToEvaluate);
        for ff=1:length(FnsToEvalNames)
            AggVars.(FnsToEvalNames{ff}).Aggregate=AggVars2(ff);
        end
    end
    
    return
end


%%
if Parallel==2
    StationaryDist=gpuArray(StationaryDist);
    PolicyIndexes=gpuArray(PolicyIndexes);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid);
    % permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    % if N_z==0
    %     PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a]),[2,1]); %[N_a,l_d+l_a]
    % else
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]
    
    for ff=1:length(FnsToEvaluate)
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);        
        Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
        Values=reshape(Values,[N_a*N_z,1]);
        % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
        temp=Values.*StationaryDistVec;
        AggVars(ff)=sum(temp(~isnan(temp)));
    end
    
else % CPU
    
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=num2cell(a_gridvals);
    z_gridvals=num2cell(z_gridvals);

    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    StationaryDistVec=gather(StationaryDistVec);
    
    AggVars=zeros(length(FnsToEvaluate),1);
    
    if l_d>0
        
        for ff=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(ff)=sum(temp(~isnan(temp)));
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(ff)=sum(temp(~isnan(temp)));
            end
        end
    
    else % l_d=0
        
        for ff=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(ff)=sum(temp(~isnan(temp)));
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(ff)=sum(temp(~isnan(temp)));
            end
        end
    end
    
end


%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    AggVars2=AggVars;
    clear AggVars
    AggVars=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        AggVars.(FnsToEvalNames{ff}).Mean=AggVars2(ff);
    end
end

if isfield(simoptions,'eval_valuefn')
    % There are some thing in AggVarsExtra that depended on the value function, put them into AggVars
    extraNames=fieldnames(AggVarsExtra);
    for ff=1:length(extraNames)
        AggVars.(extraNames{ff})=AggVarsExtra.(extraNames{ff});
    end
end



end
