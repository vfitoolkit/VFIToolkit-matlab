function [V,Policy]=ValueFnIter_InfHorz_PureDiscretization(V0,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if N_z==0

    %% CreateReturnFnMatrix_Disc_CPU creates a matrix of dimension (d and aprime)-by-a
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a lot of memory.

    if vfoptions.verbose==1
        disp('Creating return fn matrix')
    end

    if N_d==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, [], a_grid, ReturnFnParamsVec,0);
    else
        ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);
    end

    if vfoptions.verbose==1
        fprintf('Starting Value Function \n')
    end

    if N_d==0
        if vfoptions.howardsgreedy==1
            [V,Policy]=ValueFnIter_InfHorz_HowardGreedy_nod_noz_raw(V0, N_a, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
        elseif vfoptions.howardsgreedy==0
            if vfoptions.howardssparse==0
                [V,Policy]=ValueFnIter_InfHorz_nod_noz_raw(V0, N_a, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.howardssparse==1
                [V,Policy]=ValueFnIter_InfHorz_sparse_nod_noz_raw(V0, N_a, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        end
    else
        % Can't be bothered implementing HowardGreedy here, as for good runtimes you should anyway be doing Refine so wouldn't get here
        [V, Policy]=ValueFnIter_InfHorz_noz_raw(V0, n_d,n_a, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
    end
    % Note: In model without shocks there is no such thing lowmemory=1


    %% Cleaning up the output
    if vfoptions.outputkron==0
        V=reshape(V,[n_a,1]);
        if N_d==0
            Policy=UnKronPolicyIndexes1_noz(Policy, n_a, n_a, vfoptions);
        else
            Policy=UnKronPolicyIndexes2_noz(Policy, n_d, n_a, n_a, vfoptions);
        end
    else
        Policy=reshape(Policy,[1,N_a]); % no z, so N_z=0 and [1,N_a,N_z] would be empty
    end

else % N_z>0
    if vfoptions.lowmemory==0
        %% CreateReturnFnMatrix_Disc_CPU creates a matrix of dimension (d and aprime)-by-a-by-z.
        % Since the return function is independent of time creating it once and
        % then using it every iteration is good for speed, but it does use a lot of memory.

        if vfoptions.verbose==1
            disp('Creating return fn matrix')
        end

        if N_d==0
            ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_z, [], a_grid, z_gridvals, ReturnFnParamsVec,0);
        else
            ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, ReturnFnParamsVec,0);
        end

        if vfoptions.verbose==1
            fprintf('Starting Value Function \n')
        end

        if N_d==0
            if vfoptions.howardsgreedy==1
                [V,Policy]=ValueFnIter_InfHorz_HowardGreedy_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
                elseif vfoptions.howardssparse==1
                    [V,Policy]=ValueFnIter_InfHorz_sparse_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
                end
            end
        else
            % Can't be bothered implementing HowardGreedy here, as for good runtimes you should anyway be doing Refine so wouldn't get here
            [V, Policy]=ValueFnIter_InfHorz_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
        end

    elseif vfoptions.lowmemory==1
        if vfoptions.verbose==1
            disp('Starting Value Function')
        end

        if N_d==0
            if vfoptions.howardssparse==0
                [V,Policy]=ValueFnIter_InfHorz_LowMem_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.howardssparse==1
                [V,Policy]=ValueFnIter_InfHorz_LowMem_sparse_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        else
            [V, Policy]=ValueFnIter_InfHorz_LowMem_raw(V0, n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
        end
    end


    %% Cleaning up the output
    if vfoptions.outputkron==0
        V=reshape(V,[n_a,n_z]);
        if N_d==0
            Policy=UnKronPolicyIndexes1_z(Policy, n_a, n_a, n_z, vfoptions);
        else
            Policy=UnKronPolicyIndexes2_z(Policy, n_d, n_a, n_a, n_z, vfoptions);
        end
    else
        Policy=reshape(Policy,[1,N_a,N_z]);
    end
end




end