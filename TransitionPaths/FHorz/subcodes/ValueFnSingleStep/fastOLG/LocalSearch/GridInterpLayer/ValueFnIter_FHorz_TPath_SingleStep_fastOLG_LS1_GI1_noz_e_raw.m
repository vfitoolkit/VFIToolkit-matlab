function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable AND the grid interpolation layer, finite horizon, fastOLG (all
% ages at once), no z, with e. The V input is next period value fn, (a,j)-by-e, the V output is this
% period.
% The restricted version of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_noz_e_raw, done exactly as
% slowOLG/LocalSearch/GridInterpLayer/ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_raw, in TWO LAYERS,
% BOTH CONDITIONAL ON d: a coarse max over the window per d (whose answer IS the new reference,
% unclamped), then the interpolation layer per d around that argmax clamped to
% [loweredge+1,loweredge+2n-1], and only then the max over d. See that raw for the reasoning.
%
% THE REFERENCE is a COARSE aprime index per (d,a,j,e), [N_d,N_a,N_j,N_e], (a,j,e) being the order of
% the unrestricted raw's Policy.
% Policy comes out in the SAME form as the unrestricted GI raw: (d, lower coarse point, L2, L2flag) by
% [N_a,N_j,N_e], so the dispatcher's UnKron applies unchanged.
%
% THE TERMINAL AGE needs nothing: as in the unrestricted raw, EV and EVinterp are zero at j=N_j.
% Only the vfoptions.EVpre==0 expectation is implemented: local search is only reachable from a
% transition path, where EVpre=0.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_j,N_e]);

Policy=zeros(4,N_a,N_j,N_e,'gpuArray'); % d, lower coarse point, L2, L2flag

e_gridvals_J=shiftdim(e_gridvals_J,-3); % [1,1,1,N_j,N_e,l_e]

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-2);

%% First, create the big 'next period (of transition path) expected value fn.
% Exactly as the unrestricted raw, except that it is not repelem'd over d: the gathers below are per d.
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
DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV;             % [N_a,1,N_j]
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,N_j]).*EVinterp; % [n2aprime,1,N_j]

%% The window, in COARSE grid points, one per (d,a,j,e)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_j,N_e], a coarse index

%% LAYER 1: the coarse max over the window, per d, on the COARSE EV
coarseindexes=reshape(loweredge,[N_d,1,N_a,N_j,N_e])+(0:1:2*n); % [N_d,nC,N_a,N_j,N_e], coarse indices
% Level 3 returns [N_d,N_aprime,N_a,N_j,N_e] without folding d and aprime together (e plays the z role)
ReturnMatrix1=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_e, N_j, d_gridvals, a_grid(coarseindexes), a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,3);
coarsej=coarseindexes+N_a*jind;
entireRHS1=ReturnMatrix1+DiscountedEV(coarsej);
[~,offset1]=max(entireRHS1,[],2);                      % [N_d,1,N_a,N_j,N_e]
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(offset1,[N_d,N_a,N_j,N_e])-1;     % 0 to 2n, steps above loweredge, per d

% The new reference: THE LAYER 1 ANSWER per d, taken straight, UNCLAMPED
aprimeReferencePolicyNew=loweredge+coarseoffset;

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, per d
midpoint=loweredge+min(max(coarseoffset,1),2*n-1);     % [N_d,N_a,N_j,N_e]
aprimeindexes=reshape(midpoint+(midpoint-1)*n2short,[N_d,1,N_a,N_j,N_e])+(-n2short-1:1:1+n2short); % [N_d,n2long,N_a,N_j,N_e]
ReturnMatrix2=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_e, N_j, d_gridvals, aprime_grid(aprimeindexes), a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,3);
aprimej=aprimeindexes+n2aprime*jind;
entireRHS2=ReturnMatrix2+DiscountedEVinterp(aprimej);
[Vd,offsetL2]=max(entireRHS2,[],2);                    % [N_d,1,N_a,N_j,N_e]
L2d=reshape(offsetL2,[N_d,N_a,N_j,N_e]);               % 1 to n2long, per d

%% Then the max over d
[Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j,N_e]),[],1); % [1,N_a,N_j,N_e]
V=reshape(Vtemp,[N_a*N_j,N_e]);
dstar=reshape(dstar,[N_a,N_j,N_e]);

% Everything below is read AT THE CHOSEN d
lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)+N_d*N_a*N_j*shiftdim(0:1:N_e-1,-1); % [N_a,N_j,N_e], into an [N_d,N_a,N_j,N_e] array
L2star=L2d(lin);
midstar=midpoint(lin);

Policy(1,:,:,:)=reshape(dstar,[1,N_a,N_j,N_e]); % d
Policy(2,:,:,:)=reshape(midstar,[1,N_a,N_j,N_e]); % midpoint
Policy(3,:,:,:)=reshape(L2star,[1,N_a,N_j,N_e]); % aprimeL2ind

%% L2 flag, for the cell the chosen d landed in
% The unrestricted raw's rule, with the two cell endpoints at window positions 1 and n2long in the
% chosen d's row of ReturnMatrix2. Computed from L2star BEFORE the re-spelling below.
base=dstar+N_d*n2long*(0:1:N_a-1)'+N_d*n2long*N_a*(0:1:N_j-1)+N_d*n2long*N_a*N_j*shiftdim(0:1:N_e-1,-1); % window position 1, at the chosen d
linidx_lower=base;
linidx_upper=base+N_d*(n2long-1);
isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
inLowerStrict=(L2star>=2)         & (L2star<=n2short+1);
inUpperStrict=(L2star>=n2short+3) & (L2star<=n2long-1);
Policy(4,:,:,:)=reshape(2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper),[1,N_a,N_j,N_e]);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
% The unrestricted raw's lines. L2 ends up in [1,n2short+2].
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% NOTE ON THE ENCODING, inherited from the standard GI raw: the window's top fine point can come out as
% either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use the
% DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
