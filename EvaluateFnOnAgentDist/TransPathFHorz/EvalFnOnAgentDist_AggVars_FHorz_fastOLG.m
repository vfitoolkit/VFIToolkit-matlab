function AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist,Policy, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,l_d,n_a,n_z,N_j,daprime_gridvals,a_gridvals,z_gridvals_J,fastOLGpolicy,outputasstructure)
% fastOLG: so (a,j)-by-z
% Policy can be in fastOLG for or not, use fastOLGpolicy=1 or 0 to indicate this
% No z is treated elsewhere

% daprime_gridvals is [N_d*N_aprime,l_d+l_aprime]
% a_grivdals is [N_a,l_a]
% z_gridvals_J is [N_j,N_z,l_z] (will use shiftdim( ,-1) to make it [1,N_j,N_z,l_z])

% parameters that depend on age must be [1,N_j]

% Note: FnsToEvaluate is already cell (converted from struct)

% simoptions.outputasstructure=0; % hardcoded

l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

% if isfield(simoptions,'SemiExoStateFn') % If using semi-exogenous shocks
%     % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
%     n_z=[n_z,simoptions.n_semiz];
%     z_grid_J=[z_grid_J;simoptions.semiz_grid.*ones(1,N_j)];
%     l_z=length(n_z);
%     N_z=prod(n_z);
% end

z_gridvals_J=shiftdim(z_gridvals_J,-1);

if fastOLGpolicy==0 
    Policy=permute(Policy,[1,3,2]);
    PolicyValues=reshape(daprime_gridvals(Policy(:),:),[N_a,N_j,N_z,l_d+l_a]);
else % ==1 if Policy is in fastOLG form
    PolicyValues=reshape(daprime_gridvals(Policy(:),:),[N_a,N_j,N_z,l_d+l_a]);
end

%%
if outputasstructure==1
    AggVars=struct();
else % outputasstructure==0
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
end

% AgentDist is [N_a*N_j*N_z,1]
% PolicyValues is [N_a,N_j,N_z]

for ff=1:length(FnsToEvaluate)
    Values=zeros(N_a,N_j,N_z,'gpuArray');


    if isempty(FnsToEvaluateParamNames(ff).Names)
        ParamCell=cell(0,1);
    else
        % Create a matrix containing all the return function parameters (in order).
        % Each column will be a specific parameter with the values at every age.
        FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

        nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

        ParamCell=cell(nFnToEvaluateParams,1);
        for ii=1:nFnToEvaluateParams
            ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-1)}; % (a,j,z,l_d+l_a), so we want j to be after a (which is N_a)
        end
    end
    
    if l_z==1
        if l_d==0
            if l_a==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), a_gridvals, z_gridvals_J(1,:,:,1), ParamCell{:});
            elseif l_a==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J(1,:,:,1), ParamCell{:});
            end
        elseif l_d==1
            if l_a==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), a_gridvals, z_gridvals_J(1,:,:,1), ParamCell{:});
            elseif l_a==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), PolicyValues(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J(1,:,:,1), ParamCell{:});
            end
        end
    elseif l_z==2
        if l_d==0
            if l_a==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), a_gridvals, z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), ParamCell{:});
            elseif l_a==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), ParamCell{:});
            end
        elseif l_d==1
            if l_a==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), a_gridvals, z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), ParamCell{:});
            elseif l_a==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), PolicyValues(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), ParamCell{:});
            end
        end
    elseif l_z==3
        if l_d==0
            if l_a==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), a_gridvals, z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), z_gridvals_J(1,:,:,3), ParamCell{:});
            elseif l_a==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), z_gridvals_J(1,:,:,3), ParamCell{:});
            end
        elseif l_d==1
            if l_a==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), a_gridvals, z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), z_gridvals_J(1,:,:,3), ParamCell{:});
            elseif l_a==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), PolicyValues(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), z_gridvals_J(1,:,:,3), ParamCell{:});
            end
        end
    elseif l_z==4
        if l_d==0
            if l_a==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), a_gridvals, z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), z_gridvals_J(1,:,:,3), z_gridvals_J(1,:,:,4), ParamCell{:});
            elseif l_a==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), z_gridvals_J(1,:,:,3), z_gridvals_J(1,:,:,4), ParamCell{:});
            end
        elseif l_d==1
            if l_a==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), a_gridvals, z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), z_gridvals_J(1,:,:,3), z_gridvals_J(1,:,:,4), ParamCell{:});
            elseif l_a==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,1), PolicyValues(:,:,:,2), PolicyValues(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J(1,:,:,1), z_gridvals_J(1,:,:,2), z_gridvals_J(1,:,:,3), z_gridvals_J(1,:,:,4), ParamCell{:});
            end
        end
    end

    if outputasstructure==1
        AggVars.(AggVarNames{ff}).Mean=sum(Values(:).*AgentDist);
    else % outputasstructure==0
        AggVars(ff)=sum(Values(:).*AgentDist);
    end
end


end
