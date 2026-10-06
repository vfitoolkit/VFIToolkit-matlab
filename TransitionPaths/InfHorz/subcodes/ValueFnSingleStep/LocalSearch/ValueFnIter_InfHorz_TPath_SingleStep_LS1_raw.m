function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_raw(Vnext,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable. aprime is restricted to a window of 2*nlocalsearch+1
% grid points around aprimeReferencePolicy, and d is searched in FULL.
%
% THE REFERENCE IS CONDITIONAL ON d, [N_d,N_a,N_z]: one aprime index per (d,a,z). It has to be.
% d enters the budget, so the optimal aprime moves with d, and a reference that did not condition
% on d would need a window spanning the whole range of aprime*(d) -- the same failure as a
% reference that did not condition on z. Divide-and-conquer with d already does this, its
% loweredge being n_d-by-1-by-n_z.
%
% NOTE this is NOT shaped like Policy, unlike the no-d case. Policy holds (d*,aprime*) and so
% gives aprime* only AT the optimal d, which is not what a search that keeps d free needs. The
% no-d raw being Policy-shaped is a coincidence of l_d=0, not the convention. The convention is:
% the reference holds the aprime index at the centre of each window, indexed by everything the
% search conditions on.
%
% d IS NOT WINDOWED. d is often unordered -- an occupation or a sector -- and nothing here can
% tell which kind of d the user has, so a window in the d index would be unsound. Searching d in
% full costs nothing in assumptions and still gives the whole N_a/(2n+1) saving.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if vfoptions.lowmemory>0
    error('vfoptions.lowmemory is not used with vfoptions.localsearch=1 (the local window is already small in memory)')
end

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_z]);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% The window, one per (d,a,z)
% The window SLIDES at the grid ends rather than being clamped, so it is always exactly 2n+1
% distinct points inside [1,N_a]. Same line as the no-d raw, now with a d dimension.
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n);         % [N_d,N_a,N_z]
aprimeindexes=reshape(loweredge,[N_d,1,N_a,N_z])+(0:1:2*n);    % [N_d,2n+1,N_a,N_z], d fastest

%% Expected value, exactly as in the unrestricted raw
EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % [N_a,1,N_z]
DiscountedEV=DiscountFactorParamsVec*EV;

%% Return matrix on the window only
% Level 3 rather than Level 2, because Level 3 returns [N_d,N_aprime,N_a,N_z] without folding d and
% aprime together, which is the shape the two maxima below want.
ReturnMatrix=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid, z_gridvals, ReturnFnParamsVec,3);

aprimez=aprimeindexes+N_a*shiftdim((0:1:N_z-1),-2); % linear index into DiscountedEV, which is [N_a,1,N_z]
entireRHS=ReturnMatrix+DiscountedEV(aprimez);

%% Two maxima, not one
% Stage 1 gives the best aprime in the window FOR EACH d, which IS the new reference. Stage 2 then
% picks the d. Doing it this way rather than as one max over a folded (d,aprime) dimension is what
% makes the new reference free, and it also removes the decode the folded version needs -- a
% rem/floor split of the index followed by a loweredge lookup at the chosen d, which is exactly the
% construction that has drifted before in this family.
% It does change tie-breaking: a single folded max breaks a tie by linear order (d fastest), this
% breaks it by lowest d among the per-d maxima. Two names for one value, like the DC tie artefact.
[Vd,offsetd]=max(entireRHS,[],2);                               % [N_d,1,N_a,N_z]
aprimeReferencePolicyNew=loweredge+reshape(offsetd,[N_d,N_a,N_z])-1;

[V,dstar]=max(reshape(Vd,[N_d,N_a,N_z]),[],1);                  % [1,N_a,N_z]
V=reshape(V,[N_a,N_z]);
dstar=reshape(dstar,[N_a,N_z]);

% aprime at the chosen d. The lookup must use dstar and not the state index, which is the one thing
% to get wrong here.
aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1));  % [N_a,N_z]

%% Policy in transition paths
% d fastest in the folded index, matching the unrestricted raw's repelem(EV,N_d,1,1) layout
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],reshape(dstar+N_d*(aprimestar-1),[],1))',[length(n_d)+length(n_a),N_a,N_z]);

end
