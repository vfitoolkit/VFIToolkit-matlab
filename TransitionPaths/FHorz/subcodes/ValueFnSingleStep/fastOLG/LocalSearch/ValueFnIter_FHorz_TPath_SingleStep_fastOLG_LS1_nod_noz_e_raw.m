function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid,e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, fastOLG (all ages at once), no d, no z, with e. The V input is next
% period value fn, (a,j)-by-e, the V output is this period.
% The restricted version of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_noz_e_raw, done exactly as
% slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_raw: aprime is restricted to a window
% of 2*nlocalsearch+1 grid points around aprimeReferencePolicy, the window SLIDES at the grid ends, and
% coarseoffset is 0-BASED. See that raw (and the InfHorz original) for the reasoning.
%
% THE REFERENCE is [1,N_a,N_j,N_e], (a,j,e) being the order of the unrestricted raw's Policy.
% Policy comes out as the unrestricted raw's, [1,N_a,N_j,N_e], the aprime index.
%
% THE TERMINAL AGE needs nothing: as in the unrestricted raw, EV is zero at j=N_j, so R+0 is R exactly.
% Only the vfoptions.EVpre==0 expectation is implemented: local search is only reachable from a
% transition path, where EVpre=0.

N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs.

n=vfoptions.nlocalsearch;
nC=2*n+1; % the window

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_j,N_e]);

e_gridvals_J=shiftdim(e_gridvals_J,-2);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);

%% First, create the big 'next period (of transition path) expected value fn.
% Exactly as the unrestricted raw
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EV=[sum(V(N_a+1:end,:).*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_e_J to create expectations

discountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*reshape(EV,[N_a,1,N_j]); % [N_aprime,1,N_j]

%% The window, for every (a,j,e) at once
% Requires N_a>=2n+1, which is checked where nlocalsearch is validated.
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_j,N_e]
aprimeindexes=shiftdim(loweredge,-1)+(0:1:2*n)';     % [nC,N_a,N_j,N_e]

%% Return matrix on the window only
% Level 2 takes aprime as [nC,N_a,N_j,N_e] and returns [nC,N_a,N_j,N_e]
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,n_e,N_j,a_grid(aprimeindexes),a_grid,e_gridvals_J,ReturnFnParamsAgeMatrix,2);

aprimej=aprimeindexes+N_a*jind; % linear index into discountedEV, which is [N_a,1,N_j]
entireRHS=ReturnMatrix+reshape(discountedEV(aprimej(:)),[nC,N_a,N_j,N_e]);

[Vtemp,maxindex]=max(entireRHS,[],1);
V=reshape(Vtemp,[N_a*N_j,N_e]);

% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(maxindex,[N_a,N_j,N_e])-1; % 0 to 2n, steps above loweredge
aprimeindexstar=loweredge+coarseoffset;

%% Output shape for policy, the same as the unrestricted raw
Policy=reshape(aprimeindexstar,[1,N_a,N_j,N_e]);
% With no d the new reference is just the chosen aprime. Returned all the same, so that a caller
% never has to branch on whether the model has a d.
aprimeReferencePolicyNew=Policy;

end
