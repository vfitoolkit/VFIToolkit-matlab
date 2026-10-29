function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_nod_e_raw(V,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J,e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with the grid interpolation layer, finite horizon, fastOLG (all ages at once), with z
% and e, no d. fastOLG is done as (a,j,z,e): V is (a,j)-by-z-by-e.
% This is EXACTLY slowOLG/LocalSearch/GridInterpLayer/ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_nod_raw,
% with every age in one kernel and e added, in TWO LAYERS: a coarse max over the window, then the
% interpolation layer one coarse cell either side of that argmax CLAMPED to [loweredge+1,loweredge+2n-1].
% See that raw for the reasoning.
%
% THE REFERENCE is a COARSE aprime index, [1,N_a,N_j,N_z,N_e]. Layer 1's answer IS the new reference,
% the UNCLAMPED coarse argmax.
% EV, EVinterp and the discounting are copied unchanged from the unrestricted raw
% ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_nod_e_raw, which discounts BEFORE the gather
% (DiscountedEV, DiscountedEVinterp); this does the same so the cap is bit-identical. j=N_j: EV is
% zero there via the zero slice appended to pi_z_J. ONLY THE EVpre=0 PATH, which is all that raw has.
%
% Policy comes out in the SAME form as the unrestricted GI raw: (lower coarse point, L2, L2flag),
% [3,N_a,N_j,N_z,N_e], so the dispatcher's UnKron applies unchanged.

N_a=prod(n_a);
N_z=prod(n_z);
pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[1,N_a,N_j,N_z,N_e]);

z_gridvals_J=shiftdim(z_gridvals_J,-2); % [1,1,N_j,N_z,l_z]
e_gridvals_J=reshape(e_gridvals_J,[1,1,N_j,1,N_e,length(n_e)]); % [1,1,N_j,1,N_e,l_e]

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-2);

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

%% The window, in COARSE grid points, one per (a,j,z,e)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [1,N_a,N_j,N_z,N_e], a coarse index

%% LAYER 1: the coarse max over the window, on the COARSE EV
coarseindexes=loweredge+(0:1:2*n)'; % [2n+1,N_a,N_j,N_z,N_e], coarse indices
ReturnMatrix1=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, n_e, N_j, a_grid(coarseindexes), a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
coarsejz=coarseindexes+N_a*jind+N_a*N_j*zind; % linear index into DiscountedEV; e autofills
entireRHS1=ReturnMatrix1+DiscountedEV(coarsejz);
[~,maxindex1]=max(entireRHS1,[],1);
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=maxindex1-1; % [1,N_a,N_j,N_z,N_e], 0 to 2n, steps above loweredge

% The new reference: THE LAYER 1 ANSWER, taken straight, UNCLAMPED
aprimeReferencePolicyNew=loweredge+coarseoffset; % [1,N_a,N_j,N_z,N_e]

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint
midpoint=loweredge+min(max(coarseoffset,1),2*n-1); % [1,N_a,N_j,N_z,N_e]
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % [n2long,N_a,N_j,N_z,N_e]
ReturnMatrix2=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, n_e, N_j, aprime_grid(aprimeindexes), a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind;
entireRHS2=ReturnMatrix2+reshape(DiscountedEVinterp(aprimejz(:)),[n2long,N_a,N_j,N_z,N_e]);
[Vtemp,maxindexL2]=max(entireRHS2,[],1);
V=reshape(Vtemp,[N_a*N_j,N_z,N_e]);
% maxindexL2 is deliberately LEFT as [1,N_a,N_j,N_z,N_e], the shape the flag's slices are in

%% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
% (1=all to lower, 2=usual, 3=all to upper.) The standard GI raw's rule; computed from maxindexL2
% BEFORE the re-spelling below.
isInfLower=(ReturnMatrix2(1,     :,:,:,:)==-Inf);
isInfUpper=(ReturnMatrix2(n2long,:,:,:,:)==-Inf);
inLowerStrict=(maxindexL2>=2)         & (maxindexL2<=n2short+1);
inUpperStrict=(maxindexL2>=n2short+3) & (maxindexL2<=n2long-1);
L2flag=2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
% The standard GI raw's lines. L2 ends up in [1,n2short+2].
adjust=(maxindexL2<1+n2short+1);
Policy=cat(1, midpoint-adjust, adjust.*maxindexL2+(1-adjust).*(maxindexL2-n2short-1), L2flag); % [3,N_a,N_j,N_z,N_e]

% NOTE ON THE ENCODING, inherited from the standard GI raw. The window's top fine point can come out
% as either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use
% the DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
