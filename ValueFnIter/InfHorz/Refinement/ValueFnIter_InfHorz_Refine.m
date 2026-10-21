function [V,Policy]=ValueFnIter_InfHorz_Refine(V0,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions)
% When using refinement, lowmemory is implemented in the first state (return fn) but not the second (the actual iteration).

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);


if N_z==0 && N_e==0
    %% CreateReturnFnMatrix_Disc_CPU creates a matrix of dimension (d and aprime)-by-a.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a lot of memory.

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,1);

    % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
    if n_d(1)>0
        [ReturnMatrix,dstar]=max(ReturnMatrix,[],1);
        ReturnMatrix=shiftdim(ReturnMatrix,1);
    end

    %% Refinement essentially just ends up using the 'no decision variable (nod)' case to solve the value function once we have the return matrix and refine out d
    if vfoptions.howardsgreedy==1
        if vfoptions.endogenousvfi==0
            [V,Policy_a]=ValueFnIter_InfHorz_HowardGreedy_nod_noz_raw(V0, N_a, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter);
        else % endogenous VFI (relative VFI when there is no z)
            [V,Policy_a]=ValueFnIter_InfHorz_RelativeVFI_HowardGreedy_nod_noz_raw(V0, N_a, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter);
        end
    elseif vfoptions.howardsgreedy==0
        if vfoptions.howardssparse==0
            if vfoptions.endogenousvfi==0
                [V,Policy_a]=ValueFnIter_InfHorz_nod_noz_raw(V0, N_a, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            else % endogenous VFI (relative VFI when there is no z)
                [V,Policy_a]=ValueFnIter_InfHorz_RelativeVFI_nod_noz_raw(V0, N_a, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        elseif vfoptions.howardssparse==1
            if vfoptions.endogenousvfi==0
                [V,Policy_a]=ValueFnIter_InfHorz_sparse_nod_noz_raw(V0, N_a, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            else % endogenous VFI (relative VFI when there is no z)
                [V,Policy_a]=ValueFnIter_InfHorz_RelativeVFI_sparse_nod_noz_raw(V0, N_a, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        end
    end

    %% For refinement, add d into Policy
    % Policy is currently
    if n_d(1)>0
        Policy=zeros(2,N_a,'gpuArray');
        Policy(2,:)=shiftdim(Policy_a,-1);
        temppolicyindex=reshape(Policy_a,[1,N_a])+N_a*(0:1:N_a-1);
        Policy(1,:)=reshape(dstar(temppolicyindex),[N_a,1]);
    else
        Policy=Policy_a;
    end

    %% Cleaning up the output
    if vfoptions.outputkron==0
        V=reshape(V,[n_a,1]);
        Policy=UnKronPolicyIndexes2_noz(Policy, n_d, n_a, n_a, vfoptions); % We know there is a decision variable, as otherwise wouldn't be using Refine
    else
        Policy=reshape(Policy,[size(Policy,1),N_a]);
        return
    end

elseif N_z>0 && N_e==0
    %% CreateReturnFnMatrix_Disc_CPU creates a matrix of dimension (d and aprime)-by-a-by-z.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a
    % lot of memory.

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, ReturnFnParamsVec,1);

        % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
        if n_d(1)>0
            [ReturnMatrix,dstar]=max(ReturnMatrix,[],1);
            ReturnMatrix=shiftdim(ReturnMatrix,1);
        end
    elseif vfoptions.lowmemory==1 % loop over z
        %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
        ReturnMatrix=zeros(N_a,N_a,N_z,'gpuArray'); % 'refined' return matrix
        dstar=zeros(N_a,N_a,N_z,'gpuArray');
        l_z=length(n_z);
        special_n_z=ones(1,l_z,'gpuArray');
        for z_c=1:N_z
            zvals=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn,n_d, n_a, special_n_z,d_gridvals, a_grid, zvals,ReturnFnParamsVec,1); % the 1 at the end is to output for refine
            [ReturnMatrix_z,dstar_z]=max(ReturnMatrix_z,[],1); % solve for dstar
            ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
            dstar(:,:,z_c)=shiftdim(dstar_z,1);
        end
    end

    %% Refinement essentially just ends up using the 'no decision variable (nod)' case to solve the value function once we have the return matrix and refine out d
    if vfoptions.howardsgreedy==1
        if vfoptions.endogenousvfi==0
            [V,Policy_a]=ValueFnIter_InfHorz_HowardGreedy_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter);
        else % endogenous VFI (relative VFI when there is no z)
            [V,Policy_a]=ValueFnIter_InfHorz_EndogenousVFI_HowardGreedy_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter);
        end
    elseif vfoptions.howardsgreedy==0
        if vfoptions.howardssparse==0
            if vfoptions.endogenousvfi==0
                [V,Policy_a]=ValueFnIter_InfHorz_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            else % endogenous VFI (relative VFI when there is no z)
                [V,Policy_a]=ValueFnIter_InfHorz_EndogenousVFI_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        elseif vfoptions.howardssparse==1
            if vfoptions.endogenousvfi==0
                [V,Policy_a]=ValueFnIter_InfHorz_sparse_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            else % endogenous VFI (relative VFI when there is no z)
                [V,Policy_a]=ValueFnIter_InfHorz_EndogenousVFI_sparse_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        end
    end

    %% For refinement, add d into Policy
    % Policy is currently
    if n_d(1)>0
        Policy=zeros(2,N_a,N_z,'gpuArray');
        Policy(2,:,:)=shiftdim(Policy_a,-1);
        temppolicyindex=reshape(Policy_a,[1,N_a*N_z])+(0:1:N_a*N_z-1)*N_a;
        Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]);
    else
        Policy=Policy_a;
    end

    %% Cleaning up the output
    if vfoptions.outputkron==0
        V=reshape(V,[n_a,n_z]);
        Policy=UnKronPolicyIndexes2_z(Policy, n_d, n_a, n_a, n_z, vfoptions); % We know there is a decision variable, as otherwise wouldn't be using Refine
    else
        Policy=reshape(Policy,[size(Policy,1),N_a,N_z]);
        return
    end

elseif N_z==0 && N_e>0
    %% CreateReturnFnMatrix_Disc_CPU creates a matrix of dimension (d and aprime)-by-a-by-e.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a
    % lot of memory.
    % Because no z, can treat e like z for creating the return fn matrix

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, vfoptions.n_e, d_gridvals, a_grid, vfoptions.e_gridvals, ReturnFnParamsVec,1);

        % For refinement, now we solve for d*(aprime,a,e) that maximizes the ReturnFn
        if n_d(1)>0
            [ReturnMatrix,dstar]=max(ReturnMatrix,[],1);
            ReturnMatrix=shiftdim(ReturnMatrix,1);
        end
    elseif vfoptions.lowmemory>=1 % loop over e (there is no z, so lowmemory=2 is the same as lowmemory=1)
        %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
        ReturnMatrix=zeros(N_a,N_a,N_e,'gpuArray'); % 'refined' return matrix
        dstar=zeros(N_a,N_a,N_e,'gpuArray');
        l_e=length(vfoptions.n_e);
        special_n_e=ones(1,l_e);
        for e_c=1:N_e
            evals=vfoptions.e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn,n_d, n_a, special_n_e,d_gridvals, a_grid, evals,ReturnFnParamsVec,1); % the 1 at the end is to output for refine
            [ReturnMatrix_e,dstar_e]=max(ReturnMatrix_e,[],1); % solve for dstar
            ReturnMatrix(:,:,e_c)=shiftdim(ReturnMatrix_e,1);
            dstar(:,:,e_c)=shiftdim(dstar_e,1);
        end
    end

    %% Refinement essentially just ends up using the 'no decision variable (nod)' case to solve the value function once we have the return matrix and refine out d
    if vfoptions.howardsgreedy==1
        if vfoptions.endogenousvfi==0
            [V,Policy_a]=ValueFnIter_InfHorz_HowardGreedy_nod_noz_e_raw(V0, N_a, N_e, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter);
        else % endogenous VFI (relative VFI when there is no z)
            [V,Policy_a]=ValueFnIter_InfHorz_RelativeVFI_HowardGreedy_nod_noz_e_raw(V0, N_a, N_e, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter);
        end
    elseif vfoptions.howardsgreedy==0
        if vfoptions.howardssparse==0
            if vfoptions.endogenousvfi==0
                [V,Policy_a]=ValueFnIter_InfHorz_nod_noz_e_raw(V0, N_a, N_e, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            else % endogenous VFI (relative VFI when there is no z)
                [V,Policy_a]=ValueFnIter_InfHorz_RelativeVFI_nod_noz_e_raw(V0, N_a, N_e, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        elseif vfoptions.howardssparse==1
            if vfoptions.endogenousvfi==0
                [V,Policy_a]=ValueFnIter_InfHorz_sparse_nod_noz_e_raw(V0, N_a, N_e, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            else % endogenous VFI (relative VFI when there is no z)
                [V,Policy_a]=ValueFnIter_InfHorz_RelativeVFI_sparse_nod_noz_e_raw(V0, N_a, N_e, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        end
    end

    %% For refinement, add d into Policy
    % Policy is currently
    if n_d(1)>0
        Policy=zeros(2,N_a,N_e,'gpuArray');
        Policy(2,:,:)=shiftdim(Policy_a,-1);
        temppolicyindex=reshape(Policy_a,[1,N_a*N_e])+(0:1:N_a*N_e-1)*N_a;
        Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_e]);
    else
        Policy=Policy_a;
    end

    %% Cleaning up the output
    if vfoptions.outputkron==0
        V=reshape(V,[n_a,vfoptions.n_e]);
        Policy=UnKronPolicyIndexes2_z(Policy, n_d, n_a, n_a, vfoptions.n_e, vfoptions); % We know there is a decision variable, as otherwise wouldn't be using Refine; because no z, can treat e like z for the UnKron
    else
        Policy=reshape(Policy,[size(Policy,1),N_a,N_e]);
        return
    end

elseif N_z>0 && N_e>0
    %% CreateReturnFnMatrix_Disc_e creates a matrix of dimension (d and aprime)-by-a-by-z-by-e.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a
    % lot of memory.

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, vfoptions.n_e, d_gridvals, a_grid, z_gridvals, vfoptions.e_gridvals, ReturnFnParamsVec,1);

        % For refinement, now we solve for d*(aprime,a,z,e) that maximizes the ReturnFn
        if n_d(1)>0
            [ReturnMatrix,dstar]=max(ReturnMatrix,[],1);
            ReturnMatrix=shiftdim(ReturnMatrix,1);
        end
    elseif vfoptions.lowmemory==1 % loop over e
        %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
        ReturnMatrix=zeros(N_a,N_a,N_z,N_e,'gpuArray'); % 'refined' return matrix
        dstar=zeros(N_a,N_a,N_z,N_e,'gpuArray');
        l_e=length(vfoptions.n_e);
        special_n_e=ones(1,l_e);
        for e_c=1:N_e
            evals=vfoptions.e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn,n_d, n_a, n_z, special_n_e,d_gridvals, a_grid, z_gridvals, evals,ReturnFnParamsVec,1); % the 1 at the end is to output for refine
            [ReturnMatrix_e,dstar_e]=max(ReturnMatrix_e,[],1); % solve for dstar
            ReturnMatrix(:,:,:,e_c)=shiftdim(ReturnMatrix_e,1);
            dstar(:,:,:,e_c)=shiftdim(dstar_e,1);
        end
    elseif vfoptions.lowmemory==2 % loop over z and e
        %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
        ReturnMatrix=zeros(N_a,N_a,N_z,N_e,'gpuArray'); % 'refined' return matrix
        dstar=zeros(N_a,N_a,N_z,N_e,'gpuArray');
        l_z=length(n_z);
        special_n_z=ones(1,l_z);
        l_e=length(vfoptions.n_e);
        special_n_e=ones(1,l_e);
        for z_c=1:N_z
            zvals=z_gridvals(z_c,:);
            for e_c=1:N_e
                evals=vfoptions.e_gridvals(e_c,:);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn,n_d, n_a, special_n_z, special_n_e,d_gridvals, a_grid, zvals, evals,ReturnFnParamsVec,1); % the 1 at the end is to output for refine
                [ReturnMatrix_ze,dstar_ze]=max(ReturnMatrix_ze,[],1); % solve for dstar
                ReturnMatrix(:,:,z_c,e_c)=shiftdim(ReturnMatrix_ze,1);
                dstar(:,:,z_c,e_c)=shiftdim(dstar_ze,1);
            end
        end
    end

    %% Refinement essentially just ends up using the 'no decision variable (nod)' case to solve the value function once we have the return matrix and refine out d
    if vfoptions.howardsgreedy==1
        if vfoptions.endogenousvfi==0
            [V,Policy_a]=ValueFnIter_InfHorz_HowardGreedy_nod_e_raw(V0, N_a, N_z, N_e, pi_z, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter);
        else % endogenous VFI (relative VFI when there is no z)
            [V,Policy_a]=ValueFnIter_InfHorz_EndogenousVFI_HowardGreedy_nod_e_raw(V0, N_a, N_z, N_e, pi_z, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter);
        end
    elseif vfoptions.howardsgreedy==0
        if vfoptions.howardssparse==0
            if vfoptions.endogenousvfi==0
                [V,Policy_a]=ValueFnIter_InfHorz_nod_e_raw(V0, N_a, N_z, N_e, pi_z, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            else % endogenous VFI (relative VFI when there is no z)
                [V,Policy_a]=ValueFnIter_InfHorz_EndogenousVFI_nod_e_raw(V0, N_a, N_z, N_e, pi_z, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        elseif vfoptions.howardssparse==1
            if vfoptions.endogenousvfi==0
                [V,Policy_a]=ValueFnIter_InfHorz_sparse_nod_e_raw(V0, N_a, N_z, N_e, pi_z, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            else % endogenous VFI (relative VFI when there is no z)
                [V,Policy_a]=ValueFnIter_InfHorz_EndogenousVFI_sparse_nod_e_raw(V0, N_a, N_z, N_e, pi_z, vfoptions.pi_e, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        end
    end

    %% For refinement, add d into Policy
    % Policy is currently
    if n_d(1)>0
        Policy=zeros(2,N_a,N_z,N_e,'gpuArray');
        Policy(2,:,:,:)=shiftdim(Policy_a,-1);
        temppolicyindex=reshape(Policy_a,[1,N_a*N_z*N_e])+(0:1:N_a*N_z*N_e-1)*N_a;
        Policy(1,:,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z,N_e]);
    else
        Policy=Policy_a;
    end

    %% Cleaning up the output
    if vfoptions.outputkron==0
        V=reshape(V,[n_a,n_z,vfoptions.n_e]);
        Policy=UnKronPolicyIndexes2_z_e(Policy, n_d, n_a, n_a, n_z, vfoptions.n_e, vfoptions); % We know there is a decision variable, as otherwise wouldn't be using Refine
    else
        Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_e]);
        return
    end
end

end
