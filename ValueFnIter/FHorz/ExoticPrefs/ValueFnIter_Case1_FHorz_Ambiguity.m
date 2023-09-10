function [V, Policy]=ValueFnIter_Case1_FHorz_Ambiguity(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Ambiguity Aversion
% See appendix to the 'Intro to Life-Cycle models' for an explanation

V=nan;
Policy=nan;

N_d=prod(n_d);
% N_a=prod(n_a);
N_z=prod(n_z);
l_z=length(n_z);

%% Some Epstein-Zin specific options need to be set if they are not already declared
if ~isfield(vfoptions,'n_ambiguity')
    error('When using Ambiguity Aversion you must declare vfoptions.n_ambiguity (number of multiple priors)')
end
if ~isfield(vfoptions,'ambiguity_pi_z_J') &&  ~isfield(vfoptions,'ambiguity_pi_z')
    error('When using Ambiguity Aversion you must declare either vfoptions.ambiguity_pi_z or vfoptions.ambiguity_pi_z_J (defines the multiple priors)')
end
if all(size(vfoptions.n_ambiguity)==[1,1])
    n_ambiguity=vfoptions.n_ambiguity*ones(1,N_j);
elseif all(size(vfoptions.n_ambiguity)==[1,N_j])  || all(size(vfoptions.n_ambiguity)==[N_j,1]) 
    n_ambiguity=vfoptions.n_ambiguity;
else 
    error('When using Ambiguity Aversion you must declare vfoptions.n_ambiguity to be either a scalar, or a vector that depends on age/period')
end
if isfield(vfoptions,'ambiguity_pi_z') % If using ambiguity_pi_z, I just convert it to ambiguity_pi_z_J
    if ~all(size(vfoptions.ambiguity_pi_z)==[N_z,N_z,max(n_ambiguity)])
        error('When using Ambiguity Aversion you must declare vfoptions.ambiguity_pi_z to of size [N_z,N_z,max(n_ambiguity)]')
    end
    vfoptions.ambiguity_pi_z_J=reshape(vfoptions.ambiguity_pi_z,[N_z,N_z,1,max(n_ambiguity)]).*ones(1,1,N_j,1);
end
if ~all(size(vfoptions.ambiguity_pi_z_J)==[N_z,N_z,N_j,max(n_ambiguity)])
    error('When using Ambiguity Aversion you must declare vfoptions.ambiguity_pi_z_J to of size [N_z,N_z,N_j,max(n_ambiguity)]')
end

%% Create z_grid_J and pi_z_J (using joint-grid)
z_grid_J=z_grid.*ones(1,N_j);
pi_z_J=pi_z.*ones(1,1,N_j);

if isfield(vfoptions,'z_grid_J')
    z_grid_J=vfoptions.z_grid_J;
end
if isfield(vfoptions,'pi_z_J')
    pi_z_J=vfoptions.pi_z_J;
end
if isfield(vfoptions,'ExogShockFn')
    z_grid_J=zeros(sum(n_z),N_j);
    pi_z_J=zeros(N_z,N_z,N_j);
    for jj=1:N_j
        if isfield(vfoptions,'ExogShockFnParamNames')
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
        end
        z_grid_J(:,jj)=gpuArray(z_grid);
        pi_z_J(:,:,jj)=gpuArray(pi_z);
    end
end
% % And use the joint-z-grid setup for everything
% if all(size(z_grid_J)==[sum(n_z),N_j])
%     z_grid_J_backup=z_grid_J;
%     z_grid_J=zeros(N_z,l_z,N_j);
%     for jj=1:N_j
%         % I should make a CreateGridvals_J()
%         z_grid_J(:,:,jj)=CreateGridvals(n_z,z_grid_J_backup(:,jj),1); % The 1 at end indicates want output in form of matrix.
%     end
% elseif all(size(z_grid_J)==[prod(n_z),l_z,N_j])
%     % Do nothing, already using joint-z-grid setup
% else
%     error('Size of z_grid should be [sum(n_z),1] or if depends on age [sum(n_z),N_j]; or if using a joint-grid should be [prod(n_z),length(n_z)] or if depends on age [prod(n_z),length(n_z),N_j]')
% end
% 
% n_z_joint=[N_z,ones(1,l_z-1)];

%% If using e, create e_grid_J and pi_e_J (using joint-grid)
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
    l_e=length(vfoptions.n_e);

    if isfield(vfoptions,'e_grid_J')
        e_grid_J=vfoptions.e_grid_J;
    elseif isfield(vfoptions,'e_grid')
        e_grid_J=vfoptions.e_grid.*ones(1,N_j);
    end
    if isfield(vfoptions,'pi_e_J')
        pi_e_J=vfoptions.pi_e_J;
    elseif isfield(vfoptions,'pi_e')
        pi_e_J=vfoptions.pi_e.*ones(1,N_j);
    end
    % % And use the joint-e-grid setup for everything
    % if all(size(e_grid_J)==[sum(n_e),N_j])
    %     e_grid_J_backup=e_grid_J;
    %     e_grid_J=zeros(N_e,l_e,N_j);
    %     for jj=1:N_j
    %         % I should make a CreateGridvals_J()
    %         e_grid_J(:,:,jj)=CreateGridvals(n_e,e_grid_J_backup(:,jj),1); % The 1 at end indicates want output in form of matrix.
    %     end
    % elseif all(size(e_grid_J)==[prod(n_e),l_e,N_j])
    %     % Do nothing, already using joint-z-grid setup
    % else
    %     error('Size of e_grid should be [sum(n_e),1] or if depends on age [sum(n_e),N_j]; or if using a joint-grid should be [prod(n_e),length(n_e)] or if depends on age [prod(n_e),length(n_e),N_j]')
    % end
    % 
    % n_e_joint=[N_e,ones(1,l_e-1)];

end

%% Note that with ambiguity we have no need for pi_z (nor pi_e, just ambiguity_pi_z_J and ambiguity_pi_e_J)
if vfoptions.parallel==2
    if N_d==0
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_Ambiguity_nod_noz_e_raw(n_ambiguity, n_a, vfoptions.n_e, N_j, a_grid, e_grid_J, vfoptions.ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_Ambiguity_nod_e_raw(n_ambiguity, n_a, n_z, vfoptions.n_e, N_j, a_grid, z_grid_J, e_grid_J, vfoptions.ambiguity_pi_z_J, vfoptions.ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                error('Cannot use Ambiguity Aversion without any shocks (what is the point?); you have n_z=0 and no e variables')
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_Ambiguity_nod_raw(n_ambiguity, n_a, n_z, N_j, a_grid, z_grid_J, vfoptions.ambiguity_pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_Ambiguity_noz_e_raw(n_ambiguity, n_d, n_a, vfoptions.n_e, N_j, d_grid, a_grid, e_grid_J, vfoptions.ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_Ambiguity_e_raw(n_ambiguity, n_d, n_a, n_z, vfoptions.n_e, N_j, d_grid, a_grid, z_grid_J, e_grid_J, vfoptions.ambiguity_pi_z_J, vfoptions.ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                error('Cannot use Ambiguity Aversion without any shocks (what is the point?); you have n_z=0 and no e variables')
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_Ambiguity_raw(n_ambiguity, n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid_J, vfoptions.ambiguity_pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
elseif vfoptions.parallel==0 || vfoptions.parallel==1
    error('Ambiguity Aversion only implemented for Parallel=2 (gpu)')
end

if vfoptions.outputkron==0
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
    end
else
    V=VKron;
    Policy=PolicyKron;
end

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    fprintf('USING vfoptions to force integer... \n')
    % First, give some output on the size of any changes in Policy as a
    % result of turning the values into integers
    temp=max(max(max(abs(round(Policy)-Policy))));
    while ndims(temp)>1
        temp=max(temp);
    end
    fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
    % Do the actual rounding to integers
    Policy=round(Policy);
    % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
    temp=min(min(min(Policy)));
    while ndims(temp)>1
        temp=min(temp);
    end
    fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
%     Policy=uint64(Policy);
%     Policy=double(Policy);
elseif vfoptions.policy_forceintegertype==2
    % Do the actual rounding to integers
    Policy=round(Policy);
end

end