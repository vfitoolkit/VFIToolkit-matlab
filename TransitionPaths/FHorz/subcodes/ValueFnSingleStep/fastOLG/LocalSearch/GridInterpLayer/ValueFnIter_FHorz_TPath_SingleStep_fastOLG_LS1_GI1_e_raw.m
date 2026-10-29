function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable AND the grid interpolation layer, finite horizon, fastOLG (all
% ages at once), with z and e. fastOLG is done as (a,j,z,e): V is (a,j)-by-z-by-e.
% This is EXACTLY slowOLG/LocalSearch/GridInterpLayer/ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_raw,
% with every age in one kernel and e added, in TWO LAYERS, BOTH CONDITIONAL ON d: a coarse max over the
% window per d (whose answer IS the new reference, unclamped), then the interpolation layer per d around
% that argmax clamped to [loweredge+1,loweredge+2n-1], and only then the max over d. See that raw for
% the reasoning.
%
% THE REFERENCE is a COARSE aprime index per (d,a,j,z,e), [N_d,N_a,N_j,N_z,N_e].
% EV, EVinterp and the discounting are copied from the unrestricted raw
% ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_e_raw, which discounts BEFORE the gather; the only
% change is that its repelem over d is dropped (it copies each value across d, so gathering from the
% un-repeated array gives the same numbers). j=N_j: EV is zero there via the zero slice appended to
% pi_z_J. ONLY THE EVpre=0 PATH, which is all that raw has.
%
% Policy comes out in the SAME form as the unrestricted GI raw: (d, lower coarse point, L2, L2flag),
% [4,N_a,N_j,N_z,N_e], so the dispatcher's UnKron applies unchanged.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_j,N_z,N_e]);

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]); % [1,1,1,N_j,1,N_e,l_e]

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);

%% First, create the big 'next period (of transition path) expected value fn.
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV;             % [N_a,1,N_j,N_z]
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,N_j]).*EVinterp; % [n2aprime,1,N_j,N_z]

%% The window, in COARSE grid points, one per (d,a,j,z,e)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_j,N_z,N_e], a coarse index

%% LAYER 1: the coarse max over the window, per d, on the COARSE EV
coarseindexes=reshape(loweredge,[N_d,1,N_a,N_j,N_z,N_e])+(0:1:2*n); % [N_d,2n+1,N_a,N_j,N_z,N_e], coarse indices
% Level 3 returns [N_d,N_aprime,N_a,N_j,N_z,N_e] without folding d and aprime together (aprime not shifted)
ReturnMatrix1=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, N_j, d_gridvals, a_grid(coarseindexes), a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,3);
coarsejz=coarseindexes+N_a*jind+N_a*N_j*zind; % linear index into DiscountedEV; e autofills
entireRHS1=ReturnMatrix1+DiscountedEV(coarsejz);
[~,offset1]=max(entireRHS1,[],2);                          % [N_d,1,N_a,N_j,N_z,N_e]
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(offset1,[N_d,N_a,N_j,N_z,N_e])-1;     % 0 to 2n, steps above loweredge, per d

% The new reference: THE LAYER 1 ANSWER per d, taken straight, UNCLAMPED
aprimeReferencePolicyNew=loweredge+coarseoffset;           % [N_d,N_a,N_j,N_z,N_e]

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, per d
midpoint=loweredge+min(max(coarseoffset,1),2*n-1);         % [N_d,N_a,N_j,N_z,N_e]
aprimeindexes=reshape(midpoint+(midpoint-1)*n2short,[N_d,1,N_a,N_j,N_z,N_e])+(-n2short-1:1:1+n2short); % [N_d,n2long,N_a,N_j,N_z,N_e]
ReturnMatrix2=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, N_j, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,3);
aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind;    % linear index into DiscountedEVinterp; e autofills
entireRHS2=ReturnMatrix2+DiscountedEVinterp(aprimejz);
[Vd,offsetL2]=max(entireRHS2,[],2);                        % [N_d,1,N_a,N_j,N_z,N_e]
L2d=reshape(offsetL2,[N_d,N_a,N_j,N_z,N_e]);               % 1 to n2long, per d

%% Then the max over d
[Vtemp,dstar]=max(Vd,[],1);                                % [1,1,N_a,N_j,N_z,N_e]
V=reshape(Vtemp,[N_a*N_j,N_z,N_e]);
dstar=reshape(dstar,[1,N_a,N_j,N_z,N_e]);

% Everything below is read AT THE CHOSEN d
stateind=reshape(gpuArray(0:1:N_a*N_j*N_z*N_e-1),[1,N_a,N_j,N_z,N_e]);
lin=dstar+N_d*stateind;                                    % into an [N_d,N_a,N_j,N_z,N_e] array
L2star=L2d(lin);
midstar=midpoint(lin);

%% L2 flag, for the cell the chosen d landed in
% The standard GI raw's rule, with the two cell endpoints at window positions 1 and n2long in the
% chosen d's row of ReturnMatrix2. Computed from L2star BEFORE the re-spelling below.
base=dstar+N_d*n2long*stateind; % window position 1, at the chosen d
isInfLower=(ReturnMatrix2(base)==-Inf);
isInfUpper=(ReturnMatrix2(base+N_d*(n2long-1))==-Inf);
inLowerStrict=(L2star>=2)         & (L2star<=n2short+1);
inUpperStrict=(L2star>=n2short+3) & (L2star<=n2long-1);
L2flag=2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
% The standard GI raw's lines. L2 ends up in [1,n2short+2].
adjust=(L2star<1+n2short+1);
Policy=cat(1, dstar, midstar-adjust, adjust.*L2star+(1-adjust).*(L2star-n2short-1), L2flag); % [4,N_a,N_j,N_z,N_e]

% NOTE ON THE ENCODING, inherited from the standard GI raw: the window's top fine point can come out as
% either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use the
% DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
