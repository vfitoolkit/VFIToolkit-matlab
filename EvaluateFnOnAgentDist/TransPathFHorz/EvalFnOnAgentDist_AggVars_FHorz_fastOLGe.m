function AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLGe(AgentDist,Policy, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,l_d,n_a,n_z,n_e,N_j,daprime_gridvals,a_gridvals,z_gridvals_J, e_gridvals_J)
% fastOLG: so (a,j)-by-z-by-e
% Policy is in Kron form
% No z is treated elsewhere

% daprime_gridvals is [N_d*N_aprime,l_d+l_aprime]
% a_grivdals is [N_a,l_a]
% z_gridvals_J is [N_j,N_z,l_z] (will use shiftdim( ,-1) to make it [1,N_j,N_z,l_z])
% e_gridvals_J is [N_j,1,N_e,l_e] (will use shiftdim( ,-1) to make it [1,N_j,1,N_e,l_e])

% parameters that depend on age must be [1,N_j]

% Note: FnsToEvaluate is already cell (converted from struct)

% simoptions.outputasstructure=0; % hardcoded

l_a=length(n_a);
% l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% if isfield(simoptions,'SemiExoStateFn') % If using semi-exogenous shocks
%     % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
%     n_z=[n_z,simoptions.n_semiz];
%     z_grid_J=[z_grid_J;simoptions.semiz_grid.*ones(1,N_j)];
%     l_z=length(n_z);
%     N_z=prod(n_z);
% end

z_gridvals_J=shiftdim(z_gridvals_J,-1);
e_gridvals_J=shiftdim(e_gridvals_J,-1);


%%
AggVars=struct();

% AgentDist is [N_a*N_j*N_z,1]
% Policy is [N_a*N_j,N_z], it contains the index for (d,aprime)

for ff=1:length(FnsToEvaluate)
    Values=zeros(N_a,N_j,N_z,N_e,'gpuArray');

    PolicyValues=reshape(daprime_gridvals(Policy(:),:),[N_a,N_j,N_z,N_e,l_d+l_a]);

    if isempty(FnsToEvaluateParamNames(ff).Names)
        ParamCell=cell(0,1);
    else
        % Create a matrix containing all the return function parameters (in order).
        % Each column will be a specific parameter with the values at every age.
        FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

        nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

        ParamCell=cell(nFnToEvaluateParams,1);
        for ii=1:nFnToEvaluateParams
            ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-1)}; % (a,j,z,e,l_d+l_a), so we want j to be after a (which is N_a)
        end
    end
    
    if l_d==0 && l_a==1
        Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,:,1), a_gridvals, z_gridvals_J, ParamCell{:});
    elseif l_d==1 && l_a==1
        Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,:,:,1), PolicyValues(:,:,:,:,2), a_gridvals, z_gridvals_J, e_gridvals_J, ParamCell{:});
    end
    
    AggVars.(AggVarNames{ff}).Mean=sum(Values(:).*AgentDist(:));
end



end
