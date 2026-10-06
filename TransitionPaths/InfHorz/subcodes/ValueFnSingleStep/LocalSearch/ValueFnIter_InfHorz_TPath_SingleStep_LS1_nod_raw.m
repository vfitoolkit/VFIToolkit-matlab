function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_raw(Vnext,n_a,n_z, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search: aprime is restricted to a window of 2*nlocalsearch+1 grid points around
% aprimeReferencePolicy, rather than all N_a of them. The return matrix is then
% (2n+1)-by-N_a-by-N_z instead of N_a-by-N_a-by-N_z, so the cost of the step falls by a
% factor of N_a/(2n+1): at N_a=1000 and n=1 that is about 334.
%
% This is NOT the Markov-chain approximation of Bakota & Kredler (2026). It borrows their
% local-transitions requirement (the state moves only a few grid points per period) and
% their idea of an expansion point, but theirs replaces the discrete max with a transition
% matrix linear in the drift and a closed-form first order condition. This only restricts
% the max. It is an approximation unless the window happens to contain the unrestricted
% optimum, and is meant as a cheap warm start.
%
% aprimeReferencePolicy holds the aprime INDEX at the centre of each state's window, and is
% SHAPED LIKE Policy: [l_a,N_a,N_z], so [1,N_a,N_z] here. That is deliberate -- the point of
% the input is that a previously computed Policy can be fed straight back in as the
% reference, with no reshaping at the call site. When d is added later the convention stays
% the aprime channels of Policy, i.e. Policy(l_d+1:end,:,:).
% The caller supplies it; the natural default (set in the dispatcher) is the current a
% index, so the window sits around staying put.

N_a=prod(n_a);
N_z=prod(n_z);

if vfoptions.lowmemory>0
    error('vfoptions.lowmemory is not used with vfoptions.localsearch=1 (the local window is already small in memory)')
end

n=vfoptions.nlocalsearch;

% Shaped like Policy on the way in, [N_a,N_z] internally
aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_z]);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% The window
% The window SLIDES at the grid ends rather than being clamped, so it is always exactly
% 2n+1 distinct points inside [1,N_a]. Same device as divide-and-conquer's
% loweredge=min(maxindex1,n_a-maxgap).
% Note what this costs: at a reference near an end, the window is no longer centred, so
% |aprime-reference| can reach 2n rather than n. It is <=n at every interior state (one
% where reference-n>=1 and reference+n<=N_a) and <=2n everywhere. The alternative, clamping,
% would hold |aprime-reference|<=n everywhere but repeat indices and so search fewer than
% 2n+1 distinct points near the ends.
% Requires N_a>=2n+1; that is checked where nlocalsearch is validated.
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_z]
aprimeindexes=shiftdim(loweredge,-1)+(0:1:2*n)';       % [2n+1,N_a,N_z]

%% Expected value, exactly as in the unrestricted raw
EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % [N_a,1,N_z]
DiscountedEV=DiscountFactorParamsVec*EV;

%% Return matrix on the window only
% CreateReturnFnMatrix_Disc_DC1_nod needs no change: its arrayfun relies on singleton
% expansion, so a [2n+1,N_a,N_z] aprime_grid broadcasts against [1,N_a] and [1,1,N_z].
ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid(aprimeindexes), a_grid, z_gridvals, ReturnFnParamsVec,2);

aprimez=aprimeindexes+N_a*shiftdim((0:1:N_z-1),-1); % linear index into DiscountedEV
entireRHS=ReturnMatrix+DiscountedEV(aprimez);

[V,maxindex]=max(entireRHS,[],1);
V=reshape(V,[N_a,N_z]);

% aprimeindexes(k,a,z)=loweredge(a,z)+(k-1), so the decode is just the offset from loweredge
aprimeindexstar=loweredge+reshape(maxindex,[N_a,N_z])-1;

%% The new reference, which is the SAME object as the input: one aprime index per state, [1,N_a,N_z]
% With no d it is just the chosen aprime, so it coincides with Policy here. It is returned all the
% same, so that a caller never has to branch on whether the model has a d.
aprimeReferencePolicyNew=reshape(aprimeindexstar,[1,N_a,N_z]);

%% Policy in transition paths
Policy=reshape(ind2sub_vec_homemade(n_a,aprimeindexstar(:))',[length(n_a),N_a,N_z]);

end
