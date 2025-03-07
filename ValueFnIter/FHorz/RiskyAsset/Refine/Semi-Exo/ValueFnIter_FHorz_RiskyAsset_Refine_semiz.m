function [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_semiz(n_d,n_a1,n_a2,n_z,n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);

if vfoptions.refine_d(1)>0
    n_d1=n_d(1:vfoptions.refine_d(1));
else 
    n_d1=0;
end
if vfoptions.refine_d(2)>0
    n_d2=n_d(vfoptions.refine_d(1)+1:vfoptions.refine_d(1)+vfoptions.refine_d(2));
else 
    n_d2=0;
end
if vfoptions.refine_d(3)>0
    n_d3=n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1:vfoptions.refine_d(1)+vfoptions.refine_d(2)+vfoptions.refine_d(3));
else 
    n_d3=0;
end
if vfoptions.refine_d(4)>0
    n_d4=n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+vfoptions.refine_d(3)+1:end);
else 
    n_d4=0;
end
d1_grid=d_grid(1:sum(n_d1));
d2_grid=d_grid(sum(n_d1)+1:sum(n_d1)+sum(n_d2));
d3_grid=d_grid(sum(n_d1)+sum(n_d2)+1:sum(n_d1)+sum(n_d2)+sum(n_d3));
d4_grid=d_grid(sum(n_d1)+sum(n_d2)+sum(n_d3)+1:end);

%%
% Set up the semi-exogenous state
if ~isfield(vfoptions,'n_semiz')
    error('When using vfoptions.SemiExoShockFn you must declare vfoptions.n_semiz')
end
if ~isfield(vfoptions,'semiz_grid')
    error('When using vfoptions.SemiExoShockFn you must declare vfoptions.semiz_grid')
elseif all(size(vfoptions.semiz_grid)==[1,sum(vfoptions.n_semiz)]) && sum(vfoptions.n_semiz)>1
    error('vfoptions.semiz_grid must be a column vector (you have a row vector)')
end
% Following is different with riskyasset
if isfield(vfoptions,'numd_semiz')
    error('When using semiz together with riskyasset, you can no longer use vfoptions.numd_semiz, instead you determine decision variables for semiz via vfoptions.refine_d')
end
l_d4=length(n_d4);
% Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
N_semiz=prod(vfoptions.n_semiz);
l_semiz=length(vfoptions.n_semiz);
temp=getAnonymousFnInputNames(vfoptions.SemiExoStateFn);
if length(temp)>(l_semiz+l_semiz+l_d4) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
    SemiExoStateFnParamNames={temp{l_semiz+l_semiz+l_d4+1:end}}; % the first inputs will always be (semiz,semizprime,d)
else
    SemiExoStateFnParamNames={};
end
% Create pi_semiz_J
if N_semiz<=1000
    pi_semiz_J=zeros(N_semiz,N_semiz,prod(n_d4),N_j);
    for jj=1:N_j
        SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
        pi_semiz_J(:,:,:,jj)=CreatePiSemiZ(n_d4,vfoptions.n_semiz,d4_grid,vfoptions.semiz_grid,vfoptions.SemiExoStateFn,SemiExoStateFnParamValues);
    end
    % Check that pi_semiz_J has rows summing to one, if not, print a warning
    for jj=1:N_j
        if any(abs(sum(pi_semiz_J(:,:,:,jj),2)-1)>10^(-14))
            warning('Using semi-exo shocks, your transition matrix has some rows that dont sum to one for age %i',jj)
        end
    end
else % same, but set up pi_semi_J as a structure, with fields being sparse gpu array pi_semiz for each age
    fprintf('prod(n_semiz) is huge, so having to use sparse pi_semiz_J. Will take a while to create: Start ... ')
    pi_semiz_J=struct(); % Note: structure so that can easily index jj, otherwise if just one big sparse matrix, indexing it takes forever
    for jj=1:N_j
        SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
        temp=CreatePiSemiZ(n_d4,vfoptions.n_semiz,d4_grid,vfoptions.semiz_grid,vfoptions.SemiExoStateFn,SemiExoStateFnParamValues);
        pi_semiz_J.(['j',num2str(jj)])=gather(sparse(reshape(temp,[N_semiz*N_semiz,prod(n_d4)])));
    end
    fprintf('Finish. (Note: did not check the transtion matrix rows sum to one.) \n')
end

save huh.mat pi_semiz_J

% Create semiz_gridvals_J
if ndims(vfoptions.semiz_grid)==2
    if all(size(vfoptions.semiz_grid)==[sum(vfoptions.n_semiz),1])
        semiz_gridvals_J=CreateGridvals(vfoptions.n_semiz,vfoptions.semiz_grid,1).*ones(1,1,N_j,'gpuArray');
    elseif all(size(vfoptions.semiz_grid)==[prod(vfoptions.n_semiz),length(vfoptions.n_semiz)])
        semiz_gridvals_J=vfoptions.semiz_grid.*ones(1,1,N_j,'gpuArray');
    end
else % Already age-dependent
    if all(size(vfoptions.semiz_grid)==[sum(vfoptions.n_semiz),N_j])
        semiz_gridvals_J=zeros(prod(vfoptions.n_semiz),length(vfoptions.n_semiz),N_j,'gpuArray');
        for jj=1:N_j
            semiz_gridvals_J(:,:,jj)=CreateGridvals(vfoptions.n_semiz,vfoptions.semiz_grid(:,jj),1);
        end
    elseif all(size(vfoptions.semiz_grid)==[prod(vfoptions.n_semiz),length(vfoptions.n_semiz),N_j])
        semiz_gridvals_J=vfoptions.semiz_grid;
    end
end

%%
N_d1=prod(n_d1);
N_a1=prod(n_a1);
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
else
    N_e=0;
end
N_z=prod(n_z);


if N_e>0
    error('riskyasset+semiz with e not yet implemented')
    % if N_a1==0
    %     if N_z==0
    %         if N_d1==0
    %             % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_noa1_semiz_noz_e_raw(n_d2,n_d3,n_a2,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    %         else
    %             % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_noa1_semiz_noz_e_raw(n_d1,n_d2,n_d3,n_a2,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    %         end
    %     else
    %         if N_d1==0
    %             % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_semiz_noa1_e_raw(n_d2,n_d3,n_a2,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    %         else
    %             % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_noa1_semiz_e_raw(n_d1,n_d2,n_d3,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    %         end
    %     end
    % else % N_a1>0
    %     if N_z==0
    %         if N_d1==0
    %             % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_semiz_noz_e_raw(n_d2,n_d3,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    %         else
    %             % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_semiz_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    %         end
    %     else
    %         if N_d1==0
    %             % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_semiz_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    %         else
    %             % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_semiz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    %         end
    %     end
    % end
else % N_e==0
    if N_a1==0
        error('riskyasset+semiz without a1 not yet implemented')
        % if N_z==0
        %     if N_d1==0
        %         % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_noa1_semiz_noz_raw(n_d2,n_d3,n_a2,n_u, N_j, d2_grid, d3_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        %     else
        %         % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_noa1_semiz_noz_raw(n_d1,n_d2,n_d3,n_a2,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        %     end
        % else
        %     if N_d1==0
        %         % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_noa1_semiz_raw(n_d2,n_d3,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        %     else
        %         % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_noa1_semiz_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        %     end
        % end
    else % N_a1>0
        if N_z==0
            error('riskyasset+semiz with z not yet implemented')
            % if N_d1==0
            %     % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_semiz_noz_raw(n_d2,n_d3,n_a1,n_a2,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            % else
            %     % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_semiz_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            % end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_semiz_raw(n_d2,n_d3,n_d4,n_a1,n_a2,vfoptions.n_semiz,n_z,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                error('riskyasset+semiz with d1 not yet implemented')
                % [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_semiz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
end


%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if n_a1(1)==0
    n_a=n_a2;
else
    n_a=[n_a1,n_a2];
    n_d=[n_d,n_a1]; % just to UnKron
end
% Note, Policy has same shape as Case2, so just use that command
if vfoptions.outputkron==0
    if N_e>0
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_semiz,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, vfoptions.n_semiz, vfoptions.n_e, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,vfoptions.n_semiz,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, [vfoptions.n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_semiz,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, vfoptions.n_semiz, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,vfoptions.n_semiz,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, [vfoptions.n_semiz,n_z], N_j, vfoptions);
        end
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