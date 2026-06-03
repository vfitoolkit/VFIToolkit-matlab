function [VKron, PolicyKron, ValtKron, PolicyaltKron]=ValueFnIter_InfHorz_QuasiHyperbolic_Refine(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_grid, pi_z, ReturnFn, ReturnFnParamsVec, DiscountFactorParamsVec, beta0, vfoptions, isNaive)
% Refinement-based purediscretization for infinite-horizon QH (Naive and Sophisticated).
% In QH the continuation values (E[V_{j+1}] for Naive, E[Vunderbar_{j+1}] for Sophisticated)
% are functions of (aprime, zprime) only, never of d. So argmax_d of the QH RHS reduces to
% argmax_d u(d, aprime, a, z), exactly as in the standard refinement -- regardless of which
% discount factor multiplies the continuation. This file is the QH analogue of
% ValueFnIter_InfHorz_Refine.m and returns Kron-form outputs for the caller to UnKron.
% PolicyaltKron is returned as [] in the Sophisticated case.

N_a=prod(n_a);
N_z=prod(n_z);

%% Build the (d*aprime)-by-a-by-z return matrix, then refine d out
ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_grid, ReturnFnParamsVec, 1);
[ReturnMatrix, dstar]=max(ReturnMatrix, [], 1);
ReturnMatrix=shiftdim(ReturnMatrix, 1);

%% Solve QH on the refined (nod-shaped) return matrix
if isNaive
    % First: standard exponential-discounting VFI to get V_std and Policyalt_a
    if vfoptions.howardsgreedy==1
        [ValtKron, Policyalt_a]=ValueFnIter_InfHorz_HowardGreedy_nod_raw(V0, N_a, N_z, pi_z, prod(DiscountFactorParamsVec), ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
    else
        if vfoptions.howardssparse==0
            [ValtKron, Policyalt_a]=ValueFnIter_InfHorz_nod_raw(V0, N_a, N_z, pi_z, prod(DiscountFactorParamsVec), ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
        elseif vfoptions.howardssparse==1
            [ValtKron, Policyalt_a]=ValueFnIter_InfHorz_sparse_nod_raw(V0, N_a, N_z, pi_z, prod(DiscountFactorParamsVec), ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
        end
    end
    % Then: one QH-Naive step
    [VKron, Policy_a]=ValueFnIter_InfHorz_QuasiHyperbolicN_nod_raw(ValtKron, n_a, n_z, pi_z, beta0, ReturnMatrix);
else
    % Sophisticated: joint Vhat / Vunderbar iteration (Vunderbar is the 3rd output)
    [VKron, Policy_a, ValtKron]=ValueFnIter_InfHorz_QuasiHyperbolicS_nod_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, beta0, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
end

%% Recover dstar at chosen aprime, packing (d_idx, aprime_idx) into (2,N_a,N_z) Policy
PolicyKron=zeros(2, N_a, N_z, 'gpuArray');
PolicyKron(2,:,:)=shiftdim(Policy_a, -1);
temppolicyindex=reshape(Policy_a, [1, N_a*N_z])+(0:1:N_a*N_z-1)*N_a;
PolicyKron(1,:,:)=reshape(dstar(temppolicyindex), [N_a, N_z]);

if isNaive
    PolicyaltKron=zeros(2, N_a, N_z, 'gpuArray');
    PolicyaltKron(2,:,:)=shiftdim(Policyalt_a, -1);
    temppolicyaltindex=reshape(Policyalt_a, [1, N_a*N_z])+(0:1:N_a*N_z-1)*N_a;
    PolicyaltKron(1,:,:)=reshape(dstar(temppolicyaltindex), [N_a, N_z]);
else
    PolicyaltKron=[];
end

end
