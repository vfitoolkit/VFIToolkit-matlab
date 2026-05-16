function VPath=ValueFnFromPolicyOnTransPath_FHorz(PolicyPath,V_final,ParamPath,PricePath,T,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, transpathoptions, vfoptions)
% Backward-evaluate the value function along a transition path, taking
% PolicyPath as given. FHorz analog of ValueFnFromPolicyOnTransPath_InfHorz.
%
% PolicyPath:  [size(Policy,1), N_a, N_z, N_j, T]  (or compatible reshape)
% V_final:     [N_a, N_z, N_j]   value function at t=T (terminal-period boundary)
% Output VPath: [n_a, n_z, N_j, T]

if ~exist('vfoptions','var')
    vfoptions.gridinterplayer=0;
    % divide-and-conquer is not relevant for ValueFnFromPolicy
else
    if gpuDeviceCount==0
        error('ValueFnFromPolicyOnTransPath_FHorz is only available on GPU')
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    end
    if isfield(vfoptions,'exoticpreferences')
        error('ValueFnFromPolicyOnTransPath_FHorz() does not yet work with exotic preferences. Please ask on forum if you want/need this feature. \n');
    end
    if isfield(vfoptions,'experienceasset')
        if vfoptions.experienceasset==1
            error('ValueFnFromPolicyOnTransPath_FHorz() does not yet work with experienceasset. Please ask on forum if you want/need this feature. \n');
        end
    end
    if isfield(vfoptions,'n_e')
        if prod(vfoptions.n_e)>0
            error('ValueFnFromPolicyOnTransPath_FHorz() does not yet work with iid e variables. Please ask on forum if you want/need this feature. \n');
        end
    end
    if isfield(vfoptions,'n_semiz')
        if prod(vfoptions.n_semiz)>0
            error('ValueFnFromPolicyOnTransPath_FHorz() does not yet work with semi-exogenous variables. Please ask on forum if you want/need this feature. \n');
        end
    end
    % divide-and-conquer is not relevant for ValueFnFromPolicy
end
if ~exist('transpathoptions','var')
    transpathoptions=struct();
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_FHorz_StructToMatrix(PricePath,ParamPath,N_j,T);

%% Set up shock grids over the transition path
[z_gridvals_J, pi_z_J, ~, ~, ~, ~, ~, transpathoptions, vfoptions]=ExogShockSetup_FHorz_TPath(n_z,z_grid,pi_z,N_a,N_j,Parameters,PricePathNames,ParamPathNames,transpathoptions,vfoptions,3);
% transpathoptions.zpathtrivial=1 if z_gridvals_J / pi_z_J don't vary over the path;
% =0 otherwise (then transpathoptions.z_gridvals_J_T and .pi_z_J_T hold the full path)

PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,N_j,T]);

if N_d==0 && isscalar(n_a) && vfoptions.gridinterplayer==0
    l_daprime=1;
else
    l_daprime=size(PolicyPath,1);
    if vfoptions.gridinterplayer==1
        l_daprime=l_daprime-1;
    end
end
a_gridvals=CreateGridvals(n_a,a_grid,1);

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

%% Make sure inputs are on GPU
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
PricePath=gpuArray(PricePath);
V_final=gpuArray(V_final);

%% Backward iterate
VPath=zeros(N_a,N_z,N_j,T,'gpuArray');
VPath(:,:,:,T)=reshape(V_final,[N_a,N_z,N_j]);

for ttr=1:T-1
    tt=T-ttr; % T-1 down to 1

    % Update Parameters with the t=tt slice of the price/param paths
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(T-ttr,kk);
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,kk);
    end

    % Per-tt z grid and transition (handle time-varying case)
    if transpathoptions.zpathtrivial==0
        z_gridvals_J_tt=transpathoptions.z_gridvals_J_T(:,:,:,tt);
        pi_z_J_tt=transpathoptions.pi_z_J_T(:,:,:,tt);
    else
        z_gridvals_J_tt=z_gridvals_J;
        pi_z_J_tt=pi_z_J;
    end

    % PolicyPath at this tt (across all ages)
    PolicyPath_tt=PolicyPath(:,:,:,:,tt);
    PolicyValues_tt=PolicyInd2Val_FHorz(PolicyPath_tt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    PolicyIndexesKron_tt=KronPolicyIndexes_FHorz_Case1(PolicyPath_tt, n_d, n_a, n_z, N_j);

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current age, counts backwards from N_j

        % Evaluate Return Fn at policy (a, Policy(a,z,jj,tt), z, jj)
        PolicyValuesPermute=permute(PolicyValues_tt(:,:,:,jj),[2,3,1]); %[N_a,N_z,l_d+l_a]
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J_tt(:,:,jj));

        if jj==N_j
            VPath(:,:,jj,tt)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

            % Continuation: for each (a, z), need sum_{z'} pi(z, z') * V(aprime(a,z), z', jj+1, tt+1).
            % First compute EV_next(aprime, z) = sum_{z'} pi(z, z') * V_next(aprime, z') = V_next * pi_z_J(:,:,jj)'
            V_next=VPath(:,:,jj+1,tt+1); % [N_a, N_z]
            EV_next=V_next*pi_z_J_tt(:,:,jj)'; % [N_a, N_z] indexed by (aprime, z)
            EV_next(isnan(EV_next))=0; % -Inf * 0 → NaN, replace with 0

            % aprime(a, z) from policy
            if N_d==0
                optaprime=PolicyIndexesKron_tt(:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron_tt(2,:,:,jj),1);
            end
            % Linear index into EV_next at (aprime(a,z), z): index = aprime(a,z) + N_a*(z-1)
            aprime_index=reshape(optaprime,[N_a*N_z,1])+N_a*(kron((1:1:N_z)',ones(N_a,1,'gpuArray'))-1);
            EVnextAtPolicy=EV_next(aprime_index); % [N_a*N_z, 1]

            VPath(:,:,jj,tt)=FofPolicy_jj+beta*reshape(EVnextAtPolicy,[N_a,N_z]);
        end
    end
end

VPath=reshape(VPath,[n_a,n_z,N_j,T]);

end
