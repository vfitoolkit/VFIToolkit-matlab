function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_noz_e_raw(Vnext,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, no d, no z, with iid e: aprime is restricted to a window of 2*nlocalsearch+1 grid points around
% aprimeReferencePolicy. ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_raw with e in the place of z for the return (as
% the noz_e raws do), and an expectation over e' that does not depend on the current e.
% THE REFERENCE IS PER CURRENT e, [1,N_a,N_e] (see ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_e_raw for why).
% Vnext is [N_a,N_e]; pi_e is [N_e,1], the distribution of next period e.

N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs; the seed sweeps are the ones that honour it.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_e]);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% The window, one per (a,e), sliding at the grid ends (as in every localsearch raw)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_e]
aprimeindexes=shiftdim(loweredge,-1)+(0:1:2*n)';       % [2n+1,N_a,N_e]

%% Expected value, exactly as in the unrestricted raw: over e'
EV=Vnext.*shiftdim(pi_e,-1); % [N_a,N_e], aprime-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,2); % sum over e', [N_a,1]; does not depend on the current e
DiscountedEV=DiscountFactorParamsVec*EV;

%% Return matrix on the window only (e plays the role of z in the return fn matrix)
ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, a_grid(aprimeindexes), a_grid, e_gridvals, ReturnFnParamsVec,2); % [2n+1,N_a,N_e]

entireRHS=ReturnMatrix+DiscountedEV(aprimeindexes); % EV does not depend on e, so no e offset

[V,maxindex]=max(entireRHS,[],1);
V=reshape(V,[N_a,N_e]);

aprimeindexstar=loweredge+reshape(maxindex,[N_a,N_e])-1;

%% The new reference, the SAME object as the input: [1,N_a,N_e]
aprimeReferencePolicyNew=reshape(aprimeindexstar,[1,N_a,N_e]);

%% Policy in transition paths
Policy=reshape(ind2sub_vec_homemade(n_a,aprimeindexstar(:))',[length(n_a),N_a,N_e]);

end
