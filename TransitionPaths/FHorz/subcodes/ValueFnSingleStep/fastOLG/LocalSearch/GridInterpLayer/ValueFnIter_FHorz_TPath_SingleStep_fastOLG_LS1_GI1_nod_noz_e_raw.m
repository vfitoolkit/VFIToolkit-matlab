function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid,e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with the grid interpolation layer, finite horizon, fastOLG (all ages at once), no d, no
% z, with e. The V input is next period value fn, (a,j)-by-e, the V output is this period.
% The restricted version of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_nod_noz_e_raw, done exactly
% as slowOLG/LocalSearch/GridInterpLayer/ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_nod_raw, in TWO
% LAYERS: a coarse max over the window, then the interpolation layer one coarse cell either side of that
% argmax CLAMPED to [loweredge+1,loweredge+2n-1], which covers exactly the window. See that raw for the
% reasoning.
%
% THE REFERENCE is a COARSE aprime index, [1,N_a,N_j,N_e]: (a,j,e) is the order of the unrestricted
% raw's FINAL Policy (its first line allocates (3,N_a,N_e,N_j), but both lowmemory branches reallocate
% it as (3,N_a,N_j,N_e) before filling it). Layer 1's answer IS the new reference, the UNCLAMPED coarse
% argmax.
% Policy comes out in the SAME form as the unrestricted GI raw, (lower coarse point, L2, L2flag) by
% [N_a,N_j,N_e], so the dispatcher's UnKron applies unchanged.
%
% THE TERMINAL AGE needs nothing: as in the unrestricted raw, EV and EVinterp are zero at j=N_j.
% Only the vfoptions.EVpre==0 expectation is implemented: local search is only reachable from a
% transition path, where EVpre=0.

N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;
nC=2*n+1; % the coarse window

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_j,N_e]);

e_gridvals_J=shiftdim(e_gridvals_J,-2);

Policy=zeros(3,N_a,N_j,N_e,'gpuArray'); % lower coarse point, L2, L2flag

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);

%% First, create the big 'next period (of transition path) expected value fn.
% Exactly as the unrestricted raw
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

% pi_e_J is (a,j)-by-e
EV=[sum(V(N_a+1:end,:).*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_e_J to create expectations
EV=reshape(EV,[N_a,1,N_j]); % (aprime,1,j)

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

% Discounting applied BEFORE the gather, the same order the unrestricted raw uses
DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV;
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,N_j]).*EVinterp;

%% The window, in COARSE grid points, for every (a,j,e) at once
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_j,N_e], a coarse index

%% LAYER 1: the coarse max over the window, on the COARSE EV
coarseindexes=shiftdim(loweredge,-1)+(0:1:2*n)'; % [nC,N_a,N_j,N_e], coarse indices
ReturnMatrix1=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,n_e,N_j,a_grid(coarseindexes),a_grid,e_gridvals_J,ReturnFnParamsAgeMatrix,2);
coarsej=coarseindexes+N_a*jind;
entireRHS1=ReturnMatrix1+reshape(DiscountedEV(coarsej(:)),[nC,N_a,N_j,N_e]);
[~,maxindex1]=max(entireRHS1,[],1);
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(maxindex1,[N_a,N_j,N_e])-1; % 0 to 2n, steps above loweredge

% The new reference: THE LAYER 1 ANSWER, taken straight, UNCLAMPED
aprimeReferencePolicyNew=reshape(loweredge+coarseoffset,[1,N_a,N_j,N_e]);

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint
midpoint=shiftdim(loweredge+min(max(coarseoffset,1),2*n-1),-1); % [1,N_a,N_j,N_e]
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % [n2long,N_a,N_j,N_e]
ReturnMatrix2=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,n_e,N_j,aprime_grid(aprimeindexes),a_grid,e_gridvals_J,ReturnFnParamsAgeMatrix,2);
aprimej=aprimeindexes+n2aprime*jind;
entireRHS2=ReturnMatrix2+reshape(DiscountedEVinterp(aprimej(:)),[n2long,N_a,N_j,N_e]);
[Vtempii,maxindexL2]=max(entireRHS2,[],1);
V=reshape(Vtempii,[N_a*N_j,N_e]);
Policy(1,:,:,:)=midpoint; % midpoint
Policy(2,:,:,:)=maxindexL2; % aprimeL2ind

%% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
% (1=all to lower, 2=usual, 3=all to upper.) The unrestricted raw's rule, computed BEFORE the re-spelling.
isInfLower    = (ReturnMatrix2(1,     :,:,:) == -Inf);
isInfUpper    = (ReturnMatrix2(n2long,:,:,:) == -Inf);
inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
Policy(3,:,:,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
% The unrestricted raw's lines. L2 ends up in [1,n2short+2].
adjust=(Policy(2,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust; % lower grid point
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% NOTE ON THE ENCODING, inherited from the standard GI raw. The window's top fine point can come out
% as either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use
% the DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
