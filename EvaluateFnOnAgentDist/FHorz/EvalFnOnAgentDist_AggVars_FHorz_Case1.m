function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist,PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,Parallel,simoptions)
% Parallel is an optional input. If not given, will guess based on where StationaryDist

if ~exist('Parallel','var')
    if isa(StationaryDist, 'gpuArray')
        Parallel=2;
    else
        Parallel=1;
    end
else
    if isempty(Parallel)
        if isa(StationaryDist, 'gpuArray')
            Parallel=2;
        else
            Parallel=1;
        end
    end
end

if ~exist('simoptions','var')
    simoptions.lowmemory=0;
else
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0;
    end
end

if Parallel==1
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_cpu(StationaryDist,PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
end

%%
l_a=length(n_a);
N_a=prod(n_a);

%% Exogenous shock grids
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate 
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);

%% Implement new way of handling FnsToEvaluate
% Figure out l_daprime from Policy
l_daprime=size(PolicyIndexes,1);

% Note: l_z includes e and semiz (when appropriate)
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        AggVarNames=simoptions.AggVarNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end

%%
if N_z==0
    if simoptions.lowmemory==0
        AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

        StationaryDist=reshape(StationaryDist,[N_a,N_j]);

        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1]); % (N_a,N_j,l_daprime)

        a_gridvals=CreateGridvals(n_a,a_grid,1);

        for ff=1:length(FnsToEvaluate)
            % Values=nan(N_a,N_z,N_j,'gpuArray');

            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names)
                ParamCell=cell(0,1);
            else
                % Create a matrix containing all the return function parameters (in order).
                % Each column will be a specific parameter with the values at every age.
                FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

                nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

                ParamCell=cell(nFnToEvaluateParams,1);
                for ii=1:nFnToEvaluateParams
                    ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-1)}; % (a,j,l_d+l_a), so we want j to be after N_a
                end
            end
            
            Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},ParamCell,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]);
            AggVars(ff)=sum(sum(sum(Values.*StationaryDist)));
        end

    elseif simoptions.lowmemory==1 % Loop over age j
        AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

        StationaryDist=reshape(StationaryDist,[N_a,N_j]);

        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);

        for ii=1:length(FnsToEvaluate)
            Values=nan(N_a,N_j,'gpuArray');
            for jj=1:N_j

                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                    FnToEvaluateParamsVec=[];
                else
                    FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                end
                Values(:,jj)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ii}, FnToEvaluateParamsVec,PolicyValues(:,:,:,jj),l_daprime,n_a,0,a_grid,[]);
            end
            AggVars(ii)=sum(sum(Values.*StationaryDist));
        end
    end

else % N_z

    if simoptions.lowmemory==0
        AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
        StationaryDist=reshape(StationaryDist,[N_a,N_z,N_j]);

        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)

        a_gridvals=CreateGridvals(n_a,a_grid,1);

        for ff=1:length(FnsToEvaluate)
            % Values=nan(N_a,N_z,N_j,'gpuArray');

            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names)
                ParamCell=cell(0,1);
            else
                % Create a matrix containing all the return function parameters (in order).
                % Each column will be a specific parameter with the values at every age.
                FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

                nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

                ParamCell=cell(nFnToEvaluateParams,1);
                for ii=1:nFnToEvaluateParams
                    ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-2)}; % (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
                end
            end

            Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},ParamCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J);
            AggVars(ff)=sum(sum(sum(Values.*StationaryDist)));
        end

    elseif simoptions.lowmemory==1 % Loop over age j
        AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

        StationaryDist=reshape(StationaryDist,[N_a*N_z,N_j]);

        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);

        for ii=1:length(FnsToEvaluate)
            Values=nan(N_a*N_z,N_j,'gpuArray');
            for jj=1:N_j

                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                    FnToEvaluateParamsVec=[];
                else
                    FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                end
                Values(:,jj)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ii}, FnToEvaluateParamsVec,PolicyValues(:,:,:,jj),l_daprime,n_a,n_z,a_grid,z_gridvals_J(:,:,jj));
            end
            AggVars(ii)=sum(sum(Values.*StationaryDist));
        end
    elseif simoptions.lowmemory==2 % loop over age j, and over the exogenous state
        % Loop over z (which is all exogenous states, if there is an e or semiz variable it has just been rolled into z)
        AggVars=zeros(length(FnsToEvaluate),N_z,'gpuArray');
        StationaryDist=permute(reshape(StationaryDist,[N_a,N_z,N_j]),[1,3,2]); % permute moves z to last dimension
        if n_d(1)>0
            PolicyIndexes=reshape(PolicyIndexes,[size(PolicyIndexes,1),N_a,N_z,N_j]);
        else
            PolicyIndexes=reshape(PolicyIndexes,[N_a,N_z,N_j]);
        end

        for z_c=1:N_z
            StationaryDistVec_z=StationaryDist(:,:,z_c); % This is why I did the permute (to avoid a reshape here). Not actually sure whether all the reshapes would be faster than the permute or not?
            if n_d(1)>0
                PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes(:,:,z_c,:),n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1); % Note PolicyIndexes input is the wrong shape, but because this is parellel=2 the first thing PolicyInd2Val does is anyway to reshape() it.
            else
                PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes(:,z_c,:),n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1); % Note PolicyIndexes input is the wrong shape, but because this is parellel=2 the first thing PolicyInd2Val does is anyway to reshape() it.
            end

            for ii=1:length(FnsToEvaluate)
                Values=nan(N_a*N_z,N_j,'gpuArray');
                for jj=1:N_j
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                        FnToEvaluateParamsVec=[];
                    else
                        FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                    end
                    Values(:,jj)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ii}, [e_gridvals_J(e_c,:,jj),FnToEvaluateParamsVec],PolicyValues(:,:,:,jj),l_daprime,n_a,n_z,a_grid,z_gridvals_J(:,:,jj));
                end
                AggVars(ii,z_c)=sum(sum(Values.*StationaryDistVec_z));
            end
        end
        AggVars=sum(AggVars,2); % sum over e (note that the weighting by pi_e is already implicit in the stationary dist)
    end
end



%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    AggVars2=AggVars;
    clear AggVars
    AggVars=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        AggVars.(AggVarNames{ff}).Mean=AggVars2(ff);
    end
end


end
