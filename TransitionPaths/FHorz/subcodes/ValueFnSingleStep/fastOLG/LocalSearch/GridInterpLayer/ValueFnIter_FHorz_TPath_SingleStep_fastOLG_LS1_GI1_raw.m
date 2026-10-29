function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable AND the grid interpolation layer, finite horizon, fastOLG (all
% ages at once), with z. fastOLG is done as (a,j,z): V is (a,j)-by-z, Policy is (4,a,j,z).
% This is EXACTLY slowOLG/LocalSearch/GridInterpLayer/ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_raw with
% every age in one kernel, in TWO LAYERS, BOTH CONDITIONAL ON d: a coarse max over the window per d
% (whose answer IS the new reference, unclamped), then the interpolation layer per d around that argmax
% clamped to [loweredge+1,loweredge+2n-1], and only then the max over d. See that raw for the reasoning.
%
% THE REFERENCE is a COARSE aprime index per (d,a,j,z), [N_d,N_a,N_j,N_z].
% j=N_j: EV and EVinterp are zero there via the zero slice appended to pi_z_J, as in the unrestricted
% fastOLG GI raw.
%
% Policy comes out in the SAME form as ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_raw: (d, lower
% coarse point, L2, L2flag), so the dispatcher's UnKron applies unchanged. That raw takes ONE max over the
% folded (d,L2); this takes the per-d max then the max over d. Same V; ties can name a different d.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

% ONLY THE vfoptions.EVpre==0 PATH. Local search is reachable only from a transition path, where
% EVpre=0 (EVpre=1 is the Matched Expectations Path, which never sets localsearch).
if vfoptions.EVpre~=0
    error('vfoptions.localsearch requires vfoptions.EVpre=0 (ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_raw)')
end

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_j,N_z]);

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

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

pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
EVpre=zeros(N_a,1,N_j,N_z);
EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z)

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

% Discounting is applied to the whole EV and EVinterp and THEN gathered, the same order the unrestricted
% fastOLG GI raw uses. No repelem over d: the gathers below index aprime directly.
DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV;             % [N_a,1,N_j,N_z]
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,N_j]).*EVinterp; % [n2aprime,1,N_j,N_z]

Policy=zeros(4,N_a,N_j,N_z,'gpuArray'); % d, lower coarse point, L2, L2flag

%% The window, in COARSE grid points, one per (d,a,j,z)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_j,N_z], a coarse index

%% LAYER 1: the coarse max over the window, per d, on the COARSE EV
coarseindexes=reshape(loweredge,[N_d,1,N_a,N_j,N_z])+(0:1:2*n); % [N_d,nC,N_a,N_j,N_z], coarse indices
% Level 3 returns [N_d,N_aprime,N_a,N_j,N_z] without folding d and aprime together (aprime not shifted)
ReturnMatrix1=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_z, N_j, d_gridvals, a_grid(coarseindexes), a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,3);
coarsejz=coarseindexes+N_a*jind+N_a*N_j*zind; % linear index into DiscountedEV, [N_a,1,N_j,N_z]
entireRHS1=ReturnMatrix1+DiscountedEV(coarsejz);
[~,offset1]=max(entireRHS1,[],2);                          % [N_d,1,N_a,N_j,N_z]
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(offset1,[N_d,N_a,N_j,N_z])-1;         % 0 to 2n, steps above loweredge, per d

% The new reference: THE LAYER 1 ANSWER per d, taken straight, UNCLAMPED
aprimeReferencePolicyNew=loweredge+coarseoffset;           % [N_d,N_a,N_j,N_z]

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, per d
midpoint=loweredge+min(max(coarseoffset,1),2*n-1);         % [N_d,N_a,N_j,N_z]
aprimeindexes=reshape(midpoint+(midpoint-1)*n2short,[N_d,1,N_a,N_j,N_z])+(-n2short-1:1:1+n2short); % [N_d,n2long,N_a,N_j,N_z]
ReturnMatrix2=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_z, N_j, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,3);
aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind; % linear index into DiscountedEVinterp
entireRHS2=ReturnMatrix2+DiscountedEVinterp(aprimejz);
[Vd,offsetL2]=max(entireRHS2,[],2);                        % [N_d,1,N_a,N_j,N_z]
L2d=reshape(offsetL2,[N_d,N_a,N_j,N_z]);                   % 1 to n2long, per d

%% Then the max over d
[Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j,N_z]),[],1);     % [1,N_a,N_j,N_z]
V=reshape(Vtemp,[N_a*N_j,N_z]);
dstar=reshape(dstar,[N_a,N_j,N_z]);

% Everything below is read AT THE CHOSEN d
lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)+N_d*N_a*N_j*shiftdim((0:1:N_z-1),-1); % [N_a,N_j,N_z], into an [N_d,N_a,N_j,N_z] array
L2star=L2d(lin);
midstar=midpoint(lin);

%% L2 flag, for the cell the chosen d landed in
% The standard GI raw's rule, with the two cell endpoints at window positions 1 and n2long in the
% chosen d's row of ReturnMatrix2. Computed from L2star BEFORE the re-spelling below.
base=dstar+N_d*n2long*(0:1:N_a-1)'+N_d*n2long*N_a*(0:1:N_j-1)+N_d*n2long*N_a*N_j*shiftdim((0:1:N_z-1),-1); % window position 1, at the chosen d
linidx_lower=base;
linidx_upper=base+N_d*(n2long-1);
isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
inLowerStrict=(L2star>=2)         & (L2star<=n2short+1);
inUpperStrict=(L2star>=n2short+3) & (L2star<=n2long-1);
Policy(4,:,:,:)=shiftdim(2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper),-1);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
% The standard GI raw's lines. L2 ends up in [1,n2short+2].
adjust=(L2star<1+n2short+1);
Policy(1,:,:,:)=shiftdim(dstar,-1);
Policy(2,:,:,:)=shiftdim(midstar-adjust,-1);
Policy(3,:,:,:)=shiftdim(adjust.*L2star+(1-adjust).*(L2star-n2short-1),-1);

% NOTE ON THE ENCODING, inherited from the standard GI raw: the window's top fine point can come out as
% either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use the
% DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
