function [V,Policy]=ValueFnIter_InfHorz_SemiExo(V0,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions)
% Infinite horizon value function iteration with semi-exogenous states semiz.
% The last vfoptions.l_dsemiz decision variables (d2) determine the transition probabilities of semiz:
% vfoptions.pi_semiz(semiz,semizprime,d2). Any other decision variables are d1.
% The exogenous states are (semiz,z), with semiz varying first, then z. With iid e (vfoptions.n_e) the states are
% (semiz,z,e), and e is kept apart from (semiz,z) (its expectation is taken first, and the return matrix is per e).
%
% With a d1, the default (purediscretization_refinement) refines out d1 from the return matrix (it does not affect the
% transitions), and then solves the problem with only d2. With solnmethod='purediscretization' (d1,d2) are chosen jointly.
% lowmemory=1 is implemented for the refinement (it builds the return matrix one (semiz,z) at a time, or with e one e at
% a time); without a d1 there is nothing to refine and lowmemory is ignored, as for the models without semiz.

N_a=prod(n_a);
N_z=prod(n_z);
n_semiz=vfoptions.n_semiz;
N_semiz=prod(n_semiz);
n_e=vfoptions.n_e;
N_e=prod(n_e);

%% Split the decision variables into d1 and d2 (d2 are the last vfoptions.l_dsemiz, they determine the semiz transitions)
l_d2=vfoptions.l_dsemiz;
n_d2=n_d(end-l_d2+1:end);
N_d2=prod(n_d2);
if length(n_d)>l_d2
    n_d1=n_d(1:end-l_d2);
else
    n_d1=0;
end
N_d1=prod(n_d1);
l_d1=length(n_d1);
% d_gridvals is the joint grid on (d1,d2), with d1 varying first
if N_d1==0
    d2_gridvals=d_gridvals;
else
    d2_gridvals=d_gridvals(1:N_d1:end,l_d1+1:end);
end

%% The exogenous states (semiz,z), semiz varies first
if N_z==0
    n_bothz=n_semiz;
    bothz_gridvals=vfoptions.semiz_gridvals;
else
    n_bothz=[n_semiz,n_z];
    bothz_gridvals=[repmat(vfoptions.semiz_gridvals,N_z,1),repelem(z_gridvals,N_semiz,1)];
end
N_bothz=prod(n_bothz);

if strcmp(vfoptions.solnmethod,'purediscretization_refinement') && N_d1>0
    refine=1;
else
    refine=0;
end

%% Return matrix
if vfoptions.verbose==1
    disp('Creating return fn matrix')
end
if N_e>0
    if N_d1==0
        if vfoptions.lowmemory~=0
            warning('vfoptions.lowmemory is ignored by infinite horizon with semi-exogenous states and no d1 (lowmemory=0 is the only option implemented); solving as lowmemory=0')
        end
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d2, n_a, n_bothz, n_e, d2_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, ReturnFnParamsVec,0); % (d2,aprime) by a by (semiz,z) by e
    elseif refine==0
        if vfoptions.lowmemory~=0
            warning('vfoptions.lowmemory is ignored by infinite horizon with semi-exogenous states and solnmethod=purediscretization (lowmemory=0 is the only option implemented); solving as lowmemory=0')
        end
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, [n_d1,n_d2], n_a, n_bothz, n_e, d_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, ReturnFnParamsVec,0); % (d1,d2,aprime) by a by (semiz,z) by e
    else % refine
        % Refinement: calculate ReturnMatrix and 'remove' the d1 dimension (d1 does not affect the transitions, d2 does so it stays)
        if vfoptions.lowmemory==0
            ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, [n_d1,n_d2], n_a, n_bothz, n_e, d_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, ReturnFnParamsVec,0);
            ReturnMatrix=reshape(ReturnMatrix,[N_d1,N_d2*N_a,N_a,N_bothz,N_e]);
            [ReturnMatrix,d1star]=max(ReturnMatrix,[],1);
            ReturnMatrix=shiftdim(ReturnMatrix,1); % (d2,aprime) by a by (semiz,z) by e
            d1star=shiftdim(d1star,1);
        elseif vfoptions.lowmemory==1 % loop over e
            ReturnMatrix=zeros(N_d2*N_a,N_a,N_bothz,N_e,'gpuArray'); % 'refined' return matrix
            d1star=zeros(N_d2*N_a,N_a,N_bothz,N_e,'gpuArray');
            special_n_e=ones(1,length(n_e));
            for e_c=1:N_e
                e_val=vfoptions.e_gridvals(e_c,:);
                ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, [n_d1,n_d2], n_a, n_bothz, special_n_e, d_gridvals, a_grid, bothz_gridvals, e_val, ReturnFnParamsVec,0);
                ReturnMatrix_e=reshape(ReturnMatrix_e,[N_d1,N_d2*N_a,N_a,N_bothz]);
                [ReturnMatrix_e,d1star_e]=max(ReturnMatrix_e,[],1); % solve for d1star
                ReturnMatrix(:,:,:,e_c)=shiftdim(ReturnMatrix_e,1);
                d1star(:,:,:,e_c)=shiftdim(d1star_e,1);
            end
        end
    end
elseif N_d1==0
    if vfoptions.lowmemory~=0
        warning('vfoptions.lowmemory is ignored by infinite horizon with semi-exogenous states and no d1 (lowmemory=0 is the only option implemented); solving as lowmemory=0')
    end
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d2, n_a, n_bothz, d2_gridvals, a_grid, bothz_gridvals, ReturnFnParamsVec,0); % (d2,aprime) by a by (semiz,z)
elseif refine==0
    if vfoptions.lowmemory~=0
        warning('vfoptions.lowmemory is ignored by infinite horizon with semi-exogenous states and solnmethod=purediscretization (lowmemory=0 is the only option implemented); solving as lowmemory=0')
    end
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, [n_d1,n_d2], n_a, n_bothz, d_gridvals, a_grid, bothz_gridvals, ReturnFnParamsVec,0); % (d1,d2,aprime) by a by (semiz,z)
else % refine
    % Refinement: calculate ReturnMatrix and 'remove' the d1 dimension (d1 does not affect the transitions, d2 does so it stays)
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, [n_d1,n_d2], n_a, n_bothz, d_gridvals, a_grid, bothz_gridvals, ReturnFnParamsVec,0);
        ReturnMatrix=reshape(ReturnMatrix,[N_d1,N_d2*N_a,N_a,N_bothz]);
        [ReturnMatrix,d1star]=max(ReturnMatrix,[],1);
        ReturnMatrix=shiftdim(ReturnMatrix,1); % (d2,aprime) by a by (semiz,z)
        d1star=shiftdim(d1star,1);
    elseif vfoptions.lowmemory==1 % loop over (semiz,z)
        ReturnMatrix=zeros(N_d2*N_a,N_a,N_bothz,'gpuArray'); % 'refined' return matrix
        d1star=zeros(N_d2*N_a,N_a,N_bothz,'gpuArray');
        special_n_bothz=ones(1,length(n_bothz));
        for z_c=1:N_bothz
            bothz_val=bothz_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, [n_d1,n_d2], n_a, special_n_bothz, d_gridvals, a_grid, bothz_val, ReturnFnParamsVec,0);
            ReturnMatrix_z=reshape(ReturnMatrix_z,[N_d1,N_d2*N_a,N_a]);
            [ReturnMatrix_z,d1star_z]=max(ReturnMatrix_z,[],1); % solve for d1star
            ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
            d1star(:,:,z_c)=shiftdim(d1star_z,1);
        end
    end
end

if vfoptions.verbose==1
    fprintf('Starting Value Function \n')
end

%% Value fn iteration
if N_e>0
    V0=reshape(V0,[N_a,N_bothz,N_e]);

    %% Grid interpolation layer (postGI) with e
    if vfoptions.gridinterplayer==1
        % (ValueFnIter_InfHorz only lets through postGI, with one endogenous state)
        if N_d1>0 && refine==0
            error('The grid interpolation layer with semi-exogenous states and a d1 requires solnmethod=purediscretization_refinement (the default)')
        elseif N_d1>0 && vfoptions.howardsgreedy==1
            error('vfoptions.howardsgreedy=1 is not yet implemented for infinite horizon with semi-exogenous states, a d1, and the grid interpolation layer')
        end
        % (with a d1, sparse Howards is not yet implemented, so howardssparse is ignored and Howards is done by indexing)
        if ~isfield(vfoptions,'multigridswitch')
            vfoptions.multigridswitch=10;
        end
        if ~isfield(vfoptions,'maxaprimediff')
            % There is always a decision variable (d2), so use the conservatively large default of the models with d
            if n_a(1)<300
                vfoptions.maxaprimediff=ceil(n_a(1)/5);
            else
                vfoptions.maxaprimediff=ceil(n_a(1)/10);
            end
            if vfoptions.verbose_advice==1
                warning('vfoptions.postGI=1 is not guaranteed to converge globally. The default of vfoptions.maxaprimediff is set to a conservatively large value to hopefully achieve global convergence. You can try higher/lower values, and see if the solution is sensitive.')
            end
        end

        if N_d1==0
            if N_z==0
                if vfoptions.howardsgreedy==1
                    [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_HowardGreedy_nod1_noz_e_raw(V0, n_d2, n_a, n_semiz, n_e, d2_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_sparse_nod1_noz_e_raw(V0, n_d2, n_a, n_semiz, n_e, d2_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
                else
                    [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_nod1_noz_e_raw(V0, n_d2, n_a, n_semiz, n_e, d2_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
                end
            else
                if vfoptions.howardsgreedy==1
                    [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_HowardGreedy_nod1_e_raw(V0, n_d2, n_a, n_semiz, n_z, n_e, d2_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, pi_z, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_sparse_nod1_e_raw(V0, n_d2, n_a, n_semiz, n_z, n_e, d2_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, pi_z, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
                else
                    [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_nod1_e_raw(V0, n_d2, n_a, n_semiz, n_z, n_e, d2_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, pi_z, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
                end
            end
            % Policy is [d2; aprime lower; L2; L2flag]
        else
            if N_z==0
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_Refine_postGI_noz_e_raw(V0, n_d1, n_d2, n_a, n_semiz, n_e, d_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
            else
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_Refine_postGI_e_raw(V0, n_d1, n_d2, n_a, n_semiz, n_z, n_e, d_gridvals, a_grid, bothz_gridvals, vfoptions.e_gridvals, pi_z, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
            end
            % Policy is [d1; d2; aprime lower; L2; L2flag]
        end

        %% Cleaning up the output
        if vfoptions.outputkron==0
            V=reshape(V,[n_a,n_bothz,n_e]);
            % Because of how we have (semiz,z) together, use the _z_e commands to UnKron (they pass the L2 and L2flag rows through)
            if N_d1==0
                Policy=UnKronPolicyIndexes2_z_e(Policy, n_d2, n_a, n_a, n_bothz, n_e, vfoptions);
            else
                Policy=UnKronPolicyIndexes3_z_e(Policy, n_d1, n_d2, n_a, n_a, n_bothz, n_e, vfoptions);
            end
        end
        return
    end

    if N_d1==0 || refine==1
        if N_z==0
            if vfoptions.howardsgreedy==1
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_HowardGreedy_nod1_noz_e_raw(V0, N_d2, N_a, N_semiz, N_e, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_SemiExo_nod1_noz_e_raw(V0, N_d2, N_a, N_semiz, N_e, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
                elseif vfoptions.howardssparse==1
                    [V,Policy]=ValueFnIter_InfHorz_SemiExo_sparse_nod1_noz_e_raw(V0, N_d2, N_a, N_semiz, N_e, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
                end
            end
        else
            if vfoptions.howardsgreedy==1
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_HowardGreedy_nod1_e_raw(V0, N_d2, N_a, N_semiz, N_z, N_e, pi_z, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_SemiExo_nod1_e_raw(V0, N_d2, N_a, N_semiz, N_z, N_e, pi_z, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
                elseif vfoptions.howardssparse==1
                    [V,Policy]=ValueFnIter_InfHorz_SemiExo_sparse_nod1_e_raw(V0, N_d2, N_a, N_semiz, N_z, N_e, pi_z, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
                end
            end
        end
        % Policy is [d2; aprime]
        if refine==1
            % For refinement, add d1 back into Policy
            temppolicyindex=reshape(Policy(1,:,:,:)+N_d2*(Policy(2,:,:,:)-1),[1,N_a*N_bothz*N_e])+N_d2*N_a*(0:1:N_a*N_bothz*N_e-1);
            Policy=[reshape(d1star(temppolicyindex),[1,N_a,N_bothz,N_e]); Policy]; % [d1; d2; aprime]
        end
    else
        if N_z==0
            [V,Policy]=ValueFnIter_InfHorz_SemiExo_noz_e_raw(V0, N_d1, N_d2, N_a, N_semiz, N_e, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
        else
            [V,Policy]=ValueFnIter_InfHorz_SemiExo_e_raw(V0, N_d1, N_d2, N_a, N_semiz, N_z, N_e, pi_z, vfoptions.pi_semiz, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
        end
        % Policy is [d1; d2; aprime]
    end

    %% Cleaning up the output
    if vfoptions.outputkron==0
        V=reshape(V,[n_a,n_bothz,n_e]);
        % Because of how we have (semiz,z) together, use the _z_e commands to UnKron
        if N_d1==0
            Policy=UnKronPolicyIndexes2_z_e(Policy, n_d2, n_a, n_a, n_bothz, n_e, vfoptions);
        else
            Policy=UnKronPolicyIndexes3_z_e(Policy, n_d1, n_d2, n_a, n_a, n_bothz, n_e, vfoptions);
        end
    end
    return
end

V0=reshape(V0,[N_a,N_bothz]);

%% Grid interpolation layer (postGI): solve on the coarse grid, then choose (d2,aprime) on fine points around the coarse optimum
if vfoptions.gridinterplayer==1
    % (ValueFnIter_InfHorz only lets through postGI, without e, with one endogenous state)
    if N_d1>0 && refine==0
        error('The grid interpolation layer with semi-exogenous states and a d1 requires solnmethod=purediscretization_refinement (the default)')
    elseif N_d1>0 && vfoptions.howardsgreedy==1
        error('vfoptions.howardsgreedy=1 is not yet implemented for infinite horizon with semi-exogenous states, a d1, and the grid interpolation layer')
    end
    % (with a d1, sparse Howards is not yet implemented, so howardssparse is ignored and Howards is done by indexing)
    if ~isfield(vfoptions,'multigridswitch')
        vfoptions.multigridswitch=10;
    end
    if ~isfield(vfoptions,'maxaprimediff')
        % There is always a decision variable (d2), so use the conservatively large default of the models with d
        if n_a(1)<300
            vfoptions.maxaprimediff=ceil(n_a(1)/5);
        else
            vfoptions.maxaprimediff=ceil(n_a(1)/10);
        end
        if vfoptions.verbose_advice==1
            warning('vfoptions.postGI=1 is not guaranteed to converge globally. The default of vfoptions.maxaprimediff is set to a conservatively large value to hopefully achieve global convergence. You can try higher/lower values, and see if the solution is sensitive.')
        end
    end

    if N_d1==0
        if N_z==0
            if vfoptions.howardsgreedy==1
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_HowardGreedy_nod1_noz_raw(V0, n_d2, n_a, n_semiz, d2_gridvals, a_grid, bothz_gridvals, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
            elseif vfoptions.howardssparse==1
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_sparse_nod1_noz_raw(V0, n_d2, n_a, n_semiz, d2_gridvals, a_grid, bothz_gridvals, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
            else
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_nod1_noz_raw(V0, n_d2, n_a, n_semiz, d2_gridvals, a_grid, bothz_gridvals, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
            end
        else
            if vfoptions.howardsgreedy==1
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_HowardGreedy_nod1_raw(V0, n_d2, n_a, n_semiz, n_z, d2_gridvals, a_grid, bothz_gridvals, pi_z, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
            elseif vfoptions.howardssparse==1
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_sparse_nod1_raw(V0, n_d2, n_a, n_semiz, n_z, d2_gridvals, a_grid, bothz_gridvals, pi_z, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
            else
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_postGI_nod1_raw(V0, n_d2, n_a, n_semiz, n_z, d2_gridvals, a_grid, bothz_gridvals, pi_z, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
            end
        end
        % Policy is [d2; aprime lower; L2; L2flag]
    else
        if N_z==0
            [V,Policy]=ValueFnIter_InfHorz_SemiExo_Refine_postGI_noz_raw(V0, n_d1, n_d2, n_a, n_semiz, d_gridvals, a_grid, bothz_gridvals, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
        else
            [V,Policy]=ValueFnIter_InfHorz_SemiExo_Refine_postGI_raw(V0, n_d1, n_d2, n_a, n_semiz, n_z, d_gridvals, a_grid, bothz_gridvals, pi_z, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, ReturnFn, ReturnFnParamsVec, vfoptions);
        end
        % Policy is [d1; d2; aprime lower; L2; L2flag]
    end

    %% Cleaning up the output
    if vfoptions.outputkron==0
        V=reshape(V,[n_a,n_bothz]);
        % Because of how we have (semiz,z) together, use the _z commands to UnKron (they pass the L2 and L2flag rows through)
        if N_d1==0
            Policy=UnKronPolicyIndexes2_z(Policy, n_d2, n_a, n_a, n_bothz, vfoptions);
        else
            Policy=UnKronPolicyIndexes3_z(Policy, n_d1, n_d2, n_a, n_a, n_bothz, vfoptions);
        end
    end
    return
end

if N_d1==0 || refine==1
    if N_z==0
        if vfoptions.howardsgreedy==1
            [V,Policy]=ValueFnIter_InfHorz_SemiExo_HowardGreedy_nod1_noz_raw(V0, N_d2, N_a, N_semiz, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
        elseif vfoptions.howardsgreedy==0
            if vfoptions.howardssparse==0
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_nod1_noz_raw(V0, N_d2, N_a, N_semiz, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.howardssparse==1
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_sparse_nod1_noz_raw(V0, N_d2, N_a, N_semiz, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        end
    else
        if vfoptions.howardsgreedy==1
            [V,Policy]=ValueFnIter_InfHorz_SemiExo_HowardGreedy_nod1_raw(V0, N_d2, N_a, N_semiz, N_z, pi_z, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
        elseif vfoptions.howardsgreedy==0
            if vfoptions.howardssparse==0
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_nod1_raw(V0, N_d2, N_a, N_semiz, N_z, pi_z, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.howardssparse==1
                [V,Policy]=ValueFnIter_InfHorz_SemiExo_sparse_nod1_raw(V0, N_d2, N_a, N_semiz, N_z, pi_z, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        end
    end
    % Policy is [d2; aprime]
    if refine==1
        % For refinement, add d1 back into Policy
        temppolicyindex=reshape(Policy(1,:,:)+N_d2*(Policy(2,:,:)-1),[1,N_a*N_bothz])+N_d2*N_a*(0:1:N_a*N_bothz-1);
        Policy=[reshape(d1star(temppolicyindex),[1,N_a,N_bothz]); Policy]; % [d1; d2; aprime]
    end
else
    % Can't be bothered implementing HowardGreedy or sparse Howards here, as for good runtimes you should anyway be doing Refine so wouldn't get here
    if N_z==0
        [V,Policy]=ValueFnIter_InfHorz_SemiExo_noz_raw(V0, N_d1, N_d2, N_a, N_semiz, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
    else
        [V,Policy]=ValueFnIter_InfHorz_SemiExo_raw(V0, N_d1, N_d2, N_a, N_semiz, N_z, pi_z, vfoptions.pi_semiz, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
    end
    % Policy is [d1; d2; aprime]
end

%% Cleaning up the output
if vfoptions.outputkron==0
    V=reshape(V,[n_a,n_bothz]);
    % Because of how we have (semiz,z) together, use the _z commands to UnKron
    if N_d1==0
        Policy=UnKronPolicyIndexes2_z(Policy, n_d2, n_a, n_a, n_bothz, vfoptions);
    else
        Policy=UnKronPolicyIndexes3_z(Policy, n_d1, n_d2, n_a, n_a, n_bothz, vfoptions);
    end
end

end
