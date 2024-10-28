function [V,Policy]=ValueFnIter_Case1_FHorz_DC(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_z=prod(n_z);

if ~isfield(vfoptions,'level1n')
    if length(n_a)==1
        vfoptions.level1n=ceil(n_a(1)/50);
    elseif length(n_a)==2
        vfoptions.level1n=[ceil(n_a(1)/50),n_a(2)]; % default is DC2B
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a); % Otherwise causes errors

%% 1 endogenous state
if length(n_a)==1
    if N_d==0
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % N_d
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_grid, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_noz_raw(n_d,n_a, N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
%% 2 endogenous states
elseif length(n_a)==2
    if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the second endogenous state
        vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2B
        if N_d==0
            if isfield(vfoptions,'n_e')
                if N_z==0
                    if vfoptions.lowmemory==0
                        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e
                        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_nod_noz_e_lowmem_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                else
                    if vfoptions.lowmemory==0
                        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e
                        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_nod_e_lowmem_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==2 % loop over e and z
                        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_nod_e_lowmem2_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                end
            else
                if N_z==0
                    [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    if vfoptions.lowmemory==0
                        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over z
                        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_nod_lowmem_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                end
            end
        else % N_d
            if isfield(vfoptions,'n_e')
                if N_z==0
                    if vfoptions.lowmemory==0
                        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_grid, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e
                        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_noz_e_lowmem_raw(n_d,n_a,  vfoptions.n_e, N_j, d_grid, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                else
                    if vfoptions.lowmemory==0
                        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e
                        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_e_lowmem_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==2 % loop over e and z
                        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_e_lowmem2_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                end
            else
                if N_z==0
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_noz_raw(n_d,n_a, N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    if vfoptions.lowmemory==0
                        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1
                        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2B_lowmem_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                end
            end
        end
    else
        error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
        % if N_d==0
        %     if isfield(vfoptions,'n_e')
        %         if N_z==0
        %             % [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        %         else
        %             % [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        %         end
        %     else
        %         if N_z==0
        %             % [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        %         else
        %             % [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC2_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        %         end
        %     end
        % else % N_d
        %     if isfield(vfoptions,'n_e')
        %         if N_z==0
        %             % [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_grid, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        %         else
        %             % [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        %         end
        %     else
        %         if N_z==0
        %             % [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2_noz_raw(n_d,n_a, N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        %         else
        %             if vfoptions.lowmemory==0
        %                 [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        %             elseif vfoptions.lowmemory==1 % loop over z
        %                 [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC2_lowmem_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        %             end
        %         end
        %     end
        % end
    end
else
    error('Cannot use vfoptions.divideandconquer with more than two endogenous states (you have length(n_a)>2)')
end


%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==0
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
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
