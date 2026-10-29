function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, fastOLG (all ages at once), with a decision variable, no z, with e. The
% V input is next period value fn, (a,j)-by-e, the V output is this period.
% The restricted version of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_e_raw, done exactly as
% slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LS1_raw: aprime is restricted to a window of
% 2*nlocalsearch+1 grid points around aprimeReferencePolicy, d is searched in FULL, the reference is
% CONDITIONAL ON d, and the max is taken in two stages (over aprime per d, which is the new reference,
% then over d). See that raw for the reasoning.
%
% THE REFERENCE is [N_d,N_a,N_j,N_e], (a,j,e) being the order of the unrestricted raw's Policy.
% Policy comes out as the unrestricted raw's, [1,N_a,N_j,N_e], the folded (d,aprime) index with d
% fastest, so the dispatcher's UnKron applies unchanged.
% Tie-breaking differs from the folded raw (lowest d among the per-d maxima, rather than linear order);
% two names for one value, V is unaffected.
%
% THE TERMINAL AGE needs nothing: as in the unrestricted raw, EV is zero at j=N_j, so R+0 is R exactly.
% Only the vfoptions.EVpre==0 expectation is implemented: local search is only reachable from a
% transition path, where EVpre=0.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_j,N_e]);

e_gridvals_J=shiftdim(e_gridvals_J,-3); % [1,1,1,N_j,N_e,l_e]

jind=shiftdim(gpuArray(0:1:N_j-1),-2);

%% First, create the big 'next period (of transition path) expected value fn.
% Exactly as the unrestricted raw, except that it is not repelem'd over d: the gather below is per d.
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EV=[sum(V(N_a+1:end,:).*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_e_J to create expectations

discountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*reshape(EV,[N_a,1,N_j]); % [N_aprime,1,N_j], the same products the unrestricted raw repelems

%% The window, one per (d,a,j,e)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n);                    % [N_d,N_a,N_j,N_e]
aprimeindexes=reshape(loweredge,[N_d,1,N_a,N_j,N_e])+(0:1:2*n);           % [N_d,2n+1,N_a,N_j,N_e], d fastest

%% Return matrix on the window only
% Level 3 returns [N_d,N_aprime,N_a,N_j,N_e] without folding d and aprime together (e plays the z role)
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_e, N_j, d_gridvals, a_grid(aprimeindexes), a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,3);

aprimej=aprimeindexes+N_a*jind; % linear index into discountedEV, which is [N_a,1,N_j]
entireRHS=ReturnMatrix+discountedEV(aprimej);

%% Two maxima: over aprime per d (the new reference), then over d
[Vd,offset1]=max(entireRHS,[],2);                                         % [N_d,1,N_a,N_j,N_e]
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(offset1,[N_d,N_a,N_j,N_e])-1;                        % 0 to 2n, steps above loweredge, per d
aprimeReferencePolicyNew=loweredge+coarseoffset;

[Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j,N_e]),[],1);                    % [1,N_a,N_j,N_e]
V=reshape(Vtemp,[N_a*N_j,N_e]);
dstar=reshape(dstar,[N_a,N_j,N_e]);

% aprime at the chosen d. The lookup must use dstar and not the state index.
aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)+N_d*N_a*N_j*shiftdim(0:1:N_e-1,-1)); % [N_a,N_j,N_e]

%% Output shape for policy, the same as the unrestricted raw
% The folded (d,aprime) index, d fastest, as the unrestricted raw's repelem(EV,N_d,1,1) layout
Policy=reshape(dstar+N_d*(aprimestar-1),[1,N_a,N_j,N_e]);

end
