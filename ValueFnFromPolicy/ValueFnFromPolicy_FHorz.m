function varargout=ValueFnFromPolicy_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% vfoptions is an optional input (not required)
% we will fill in defaults if needed (e.g. `gridinterplayer`).

if ~exist('vfoptions','var')
    vfoptions.n_e=0;
    vfoptions.n_semiz=0;
    vfoptions.experienceasset=0;
    vfoptions.experienceassetu=0;
    vfoptions.experienceassetz=0;
    vfoptions.experienceassete=0;
    vfoptions.experienceassetze=0;
    vfoptions.riskyasset=0;
    vfoptions.gridinterplayer=0;
    % divide-and-conquer is not relevant for ValueFnFromPolicy
else
    if gpuDeviceCount==0
        error('ValueFnFromPolicy_FHorz is only available on GPU')
    end
    if ~isfield(vfoptions,'n_e')
        vfoptions.n_e=0;
    end
    if ~isfield(vfoptions,'n_semiz')
        vfoptions.n_semiz=0;
    end
    if ~isfield(vfoptions,'experienceasset')
        vfoptions.experienceasset=0;
    end
    if ~isfield(vfoptions,'experienceassetu')
        vfoptions.experienceassetu=0;
    end
    if ~isfield(vfoptions,'experienceassetz')
        vfoptions.experienceassetz=0;
    end
    if ~isfield(vfoptions,'experienceassete')
        vfoptions.experienceassete=0;
    end
    if ~isfield(vfoptions,'experienceassetze')
        vfoptions.experienceassetze=0;
    end
    if ~isfield(vfoptions,'riskyasset')
        vfoptions.riskyasset=0;
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    end
    % divide-and-conquer is not relevant for ValueFnFromPolicy
end

%% Move grids and Policy to GPU (downstream code assumes gpuArray).
% z, e, semiz are handled inside ExogShockSetup_FHorz / SemiExogShockSetup_FHorz, so don't need conversion here.
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
Policy=gpuArray(Policy);


%% Exogenous shock grids
% Switch to z_gridvals
[z_gridvals_J, pi_z_J,vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);
% Convert z and e to age-dependent joint-grids and transtion matrix
% output: z_gridvals_J, pi_z_J, options.e_gridvals_J, options.pi_e_J

%% Dispatch to QuasiHyperbolic subfn if exoticpreferences=='QuasiHyperbolic'
if isfield(vfoptions,'exoticpreferences') && strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V,Valt};
    return
end

%% Dispatch to ExpAsset subfn if experienceasset==1
if vfoptions.experienceasset==1
    V=ValueFnFromPolicy_FHorz_ExpAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Dispatch to ExpAssetu subfn if experienceassetu==1
if vfoptions.experienceassetu==1
    V=ValueFnFromPolicy_FHorz_ExpAssetu(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Dispatch to ExpAssetz subfn if experienceassetz==1
if vfoptions.experienceassetz==1
    V=ValueFnFromPolicy_FHorz_ExpAssetz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Dispatch to ExpAssete subfn if experienceassete==1
if vfoptions.experienceassete==1
    V=ValueFnFromPolicy_FHorz_ExpAssete(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Dispatch to ExpAssetze subfn if experienceassetze==1
if vfoptions.experienceassetze==1
    V=ValueFnFromPolicy_FHorz_ExpAssetze(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Dispatch to RiskyAsset subfn if riskyasset==1
if vfoptions.riskyasset==1
    V=ValueFnFromPolicy_FHorz_RiskyAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Dispatch to SemiExo subfn if n_semiz>0
if prod(vfoptions.n_semiz)>0
    V=ValueFnFromPolicy_FHorz_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    V=ValueFnFromPolicy_FHorz_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%%
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
% Basic setup: the first inputs of ReturnFn will be (d,aprime,a,z,..) and everything after this is a parameter, so we get the names of all these parameters.
% But this changes if you have e, semiz, or just multiple d, and if you use riskyasset, expasset, etc.
% So figure out which setup we have, and get the relevant ReturnFnParamNames

%%
a_gridvals=CreateGridvals(n_a,a_grid,1);

%%
if N_z==0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); %[N_a,l_d+l_a,N_j]
    % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, N_j, vfoptions);

    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        % Evaluate Return Fn
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);

        if jj==N_j
            V(:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=V(:,jj+1);

            if N_d==0
                optaprime=PolicyIndexesKron(1,:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
            end

            aprimez_index=reshape(optaprime,[N_a,1]); % N_a*(z_index-1), but just with lots of kron

            EVnextOfPolicy=EVnext(aprimez_index);

            V(:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,1]);
        end
    end

    %Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,N_j]);

elseif N_z==0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_e,l_d+l_a,N_j]
    % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, vfoptions.n_e, N_j, vfoptions);

    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_e,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        % Evaluate Return Fn
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));

        if jj==N_j
            V(:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=sum(V(:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-1),2); % expectation over iid

            if N_d==0
                optaprime=PolicyIndexesKron(1,:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
            end

            % e is iid -> EVnext shape [N_a,1] depends only on aprime
            EVnextOfPolicy=EVnext(reshape(optaprime,[N_a*N_e,1]));

            V(:,:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,N_e]);
        end
    end

    %Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,vfoptions.n_e,N_j]);

elseif N_z>0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_z,l_d+l_a,N_j]
    % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, N_j, vfoptions);

    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_z,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        % Evaluate Return Fn
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));

        if jj==N_j
            V(:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            % EVnext(a, z_from) = sum_{z_to} pi(z_from, z_to) * V(a, z_to, jj+1)
            EVnext=V(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z_from]
            EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)

            if N_d==0
                optaprime=PolicyIndexesKron(1,:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
            end

            aprimez_index=reshape(optaprime,[N_a*N_z,1])+N_a*(kron((1:1:N_z)',ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron

            EVnextOfPolicy=EVnext(aprimez_index);

            V(:,:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,N_z]);
        end
    end

    %Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,n_z,N_j]);

elseif N_z>0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a,N_z*N_e,l_d+l_a,N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
    % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions);


    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_z,N_e,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        % Evaluate Return Fn
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);

        if jj==N_j
            V(:,:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=sum(V(:,:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-2),3); % expectation over iid
            EVnext=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1); % size N_z-by-1
            EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EVnext=sum(EVnext,2); % sum over z', leaving a singular second dimension
%             EVnext=reshape(EVnext,[N_a,N_z]); % Not necessary as just index into it

            if N_d==0
                optaprime=PolicyIndexesKron(1,:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
            end

            aprimez_index=reshape(optaprime,[N_a*N_z*N_e,1])+N_a*(kron(kron(ones(N_e,1,'gpuArray'),(1:1:N_z)'),ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron

            EVnextOfPolicy=EVnext(aprimez_index);

            V(:,:,:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,N_z,N_e]);
        end

    end

    % Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,n_z,vfoptions.n_e,N_j]);
end

varargout={V};

end
