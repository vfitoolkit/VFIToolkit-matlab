function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, fastOLG (all ages at once), with a decision variable, no z. The V input
% is next period value fn (across all ages), the V output is this period.
% The fastOLG version of slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LS1_raw: aprime is
% restricted to a window of 2*nlocalsearch+1 grid points around aprimeReferencePolicy, d is searched in
% FULL, the reference is CONDITIONAL ON d, and the max is taken in two stages (over aprime per d, which
% is the new reference, then over d). See that raw (and the InfHorz original) for the reasoning.
%
% THE REFERENCE is [N_d,N_a,N_j].
% Policy comes out in the SAME Kron form as the unrestricted raw, [1,N_a,N_j], the folded (d,aprime)
% index with d fastest, so the dispatcher's UnKron applies unchanged.
% Tie-breaking differs from the folded raw (lowest d among the per-d maxima, rather than linear order);
% two names for one value, V is unaffected.
%
% EV is built EXACTLY as ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_raw does: the j=N_j slice is left
% as zeros, so the terminal age needs no special case. That raw has no EVpre branch; local search is only
% reachable from a transition path, where EVpre=0 anyway.

N_d=prod(n_d);
N_a=prod(n_a);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_j]);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=V(:,2:end);

% The unrestricted raw repelems this over d; the gather below does the same job
DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV; % [N_a,1,N_j]

%% The window, one per (d,a,j)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_j]
aprimeindexes=reshape(loweredge,[N_d,1,N_a,N_j])+(0:1:2*n); % [N_d,2n+1,N_a,N_j]

%% Return matrix on the window only
% Level 3 returns [N_d,N_aprime,N_a,N_j] without folding d and aprime together
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_noz(ReturnFn, n_d, N_j, d_gridvals, a_grid(aprimeindexes), a_grid, ReturnFnParamsAgeMatrix,3);

aprimej=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-2); % linear index into DiscountedEV, which is [N_a,1,N_j]
entireRHS=ReturnMatrix+DiscountedEV(aprimej);

%% Two maxima: over aprime per d (the new reference), then over d
[Vd,offset1]=max(entireRHS,[],2);                               % [N_d,1,N_a,N_j]
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(offset1,[N_d,N_a,N_j])-1;                  % 0 to 2n, steps above loweredge, per d
aprimeReferencePolicyNew=loweredge+coarseoffset;                % [N_d,N_a,N_j]

[Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j]),[],1);              % [1,N_a,N_j]
V=reshape(Vtemp,[N_a,N_j]);
dstar=reshape(dstar,[N_a,N_j]);

% aprime at the chosen d. The lookup must use dstar and not the state index.
aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)); % [N_a,N_j]

%% Output shape for policy, the same as the unrestricted raw
% The folded (d,aprime) index, d fastest, as the unrestricted raw's repelem(...,N_d,1,1) layout
Policy=reshape(dstar+N_d*(aprimestar-1),[1,N_a,N_j]);

end
