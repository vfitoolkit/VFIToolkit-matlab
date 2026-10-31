function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_e_raw(Vnext,n_a,n_z,n_e, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, no d, with z and iid e: aprime is restricted to a window of 2*nlocalsearch+1 grid points around
% aprimeReferencePolicy. ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_raw with an e dimension; see its header for the
% method, the sliding window and the 0-based coarseoffset.
%
% THE REFERENCE IS PER CURRENT e, [1,N_a,N_z,N_e]: one aprime index per state the search conditions on. The
% expectation is over e' and so does not depend on the current e, but the return does, and so does the optimal aprime;
% a window shared across e would have to span that whole range, the same failure as a reference that ignored z.
% Vnext is [N_a,N_z,N_e]; pi_z is [N_z,N_zprime]; pi_e is [N_e,1], the distribution of next period e.

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs; the seed sweeps are the ones that honour it.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_z,N_e]);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% The window, one per (a,z,e), sliding at the grid ends (as in every localsearch raw)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_z,N_e]
aprimeindexes=shiftdim(loweredge,-1)+(0:1:2*n)';       % [2n+1,N_a,N_z,N_e]

%% Expected value, exactly as in the unrestricted raw: first e', then z'
EV=Vnext.*shiftdim(pi_e,-2); % [N_a,N_z,N_e], aprime-by-zprime-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_z]
EV=EV.*shiftdim(pi_z',-1); % [N_a,N_zprime,N_z]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % [N_a,1,N_z]
DiscountedEV=DiscountFactorParamsVec*EV;

%% Return matrix on the window only
% Level 2 of CreateReturnFnMatrix_Disc_DC1_nod_e takes the aprime grid as given and broadcasts it, so a
% [2n+1,N_a,N_z,N_e] aprime grid returns [2n+1,N_a,N_z,N_e].
ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, a_grid(aprimeindexes), a_grid, z_gridvals, e_gridvals, ReturnFnParamsVec,2);

aprimez=aprimeindexes+N_a*shiftdim((0:1:N_z-1),-1); % linear index into DiscountedEV [N_a,1,N_z]; EV does not depend on e, so no e offset
entireRHS=ReturnMatrix+DiscountedEV(aprimez);

[V,maxindex]=max(entireRHS,[],1);
V=reshape(V,[N_a,N_z,N_e]);

coarseoffset=reshape(maxindex,[N_a,N_z,N_e])-1; % 0 to 2n, steps above loweredge
aprimeindexstar=loweredge+coarseoffset;

%% The new reference, the SAME object as the input: [1,N_a,N_z,N_e]
aprimeReferencePolicyNew=reshape(aprimeindexstar,[1,N_a,N_z,N_e]);

%% Policy in transition paths
Policy=reshape(ind2sub_vec_homemade(n_a,aprimeindexstar(:))',[length(n_a),N_a,N_z,N_e]);

end
