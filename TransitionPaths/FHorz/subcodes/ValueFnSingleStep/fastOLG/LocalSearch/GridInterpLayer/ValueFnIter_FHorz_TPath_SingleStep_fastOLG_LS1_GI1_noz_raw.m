function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable AND the grid interpolation layer, finite horizon, fastOLG (all
% ages at once), no z. The V input is next period value fn (across all ages), the V output is this period.
% The fastOLG version of slowOLG/LocalSearch/GridInterpLayer/ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_raw,
% in TWO LAYERS, BOTH CONDITIONAL ON d: a coarse max over the window per d (whose answer IS the new
% reference, unclamped), then the interpolation layer per d around that argmax clamped to
% [loweredge+1,loweredge+2n-1], and only then the max over d. See that raw for the reasoning.
%
% THE REFERENCE is a COARSE aprime index per (d,a,j), [N_d,N_a,N_j].
% Policy comes out in the SAME form as the unrestricted GI raw, [4,N_a,N_j]: (d, lower coarse point, L2,
% L2flag), so the dispatcher's UnKron applies unchanged.
%
% EV and EVinterp are built EXACTLY as ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_noz_raw does: the
% j=N_j slice is left as zeros, so the terminal age needs no special case. That raw has no EVpre branch;
% local search is only reachable from a transition path, where EVpre=0 anyway.

N_d=prod(n_d);
N_a=prod(n_a);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_j]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

Policy=zeros(4,N_a,N_j,'gpuArray'); % d, lower coarse point, L2, L2flag

jind=shiftdim(gpuArray(0:1:N_j-1),-2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=V(:,2:end);
% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

% Discounted BEFORE the gather, as the GI raw does. The GI raw also repelems these over d; the gathers
% below index the same values without that copy.
DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV;             % [N_a,1,N_j]
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,N_j]).*EVinterp; % [n2aprime,1,N_j]

%% The window, in COARSE grid points, one per (d,a,j)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_j], a coarse index

%% LAYER 1: the coarse max over the window, per d, on the COARSE EV
coarseindexes=reshape(loweredge,[N_d,1,N_a,N_j])+(0:1:2*n); % [N_d,nC,N_a,N_j], coarse indices
% Level 3 returns [N_d,N_aprime,N_a,N_j] without folding d and aprime together
ReturnMatrix1=CreateReturnFnMatrix_fastOLG_Disc_DC1_noz(ReturnFn, n_d, N_j, d_gridvals, a_grid(coarseindexes), a_grid, ReturnFnParamsAgeMatrix,3);
coarsej=coarseindexes+N_a*jind;
entireRHS1=ReturnMatrix1+DiscountedEV(coarsej);
[~,offset1]=max(entireRHS1,[],2);                      % [N_d,1,N_a,N_j]
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(offset1,[N_d,N_a,N_j])-1;         % 0 to 2n, steps above loweredge, per d

% The new reference: THE LAYER 1 ANSWER per d, taken straight, UNCLAMPED
aprimeReferencePolicyNew=loweredge+coarseoffset;       % [N_d,N_a,N_j]

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, per d
midpoint=loweredge+min(max(coarseoffset,1),2*n-1);     % [N_d,N_a,N_j]
aprimeindexes=reshape(midpoint+(midpoint-1)*n2short,[N_d,1,N_a,N_j])+(-n2short-1:1:1+n2short); % [N_d,n2long,N_a,N_j]
% Level 3 here (the GI raw uses the folded Level 2 of the same creator, same values)
ReturnMatrix2=CreateReturnFnMatrix_fastOLG_Disc_DC1_noz(ReturnFn, n_d, N_j, d_gridvals, aprime_grid(aprimeindexes), a_grid, ReturnFnParamsAgeMatrix,3);
aprimej=aprimeindexes+n2aprime*jind;
entireRHS2=ReturnMatrix2+DiscountedEVinterp(aprimej);
[Vd,offsetL2]=max(entireRHS2,[],2);                    % [N_d,1,N_a,N_j]
L2d=reshape(offsetL2,[N_d,N_a,N_j]);                   % 1 to n2long, per d

%% Then the max over d
[Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j]),[],1);     % [1,N_a,N_j]
V=reshape(Vtemp,[N_a,N_j]);
dstar=reshape(dstar,[N_a,N_j]);

% Everything below is read AT THE CHOSEN d
lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1);        % [N_a,N_j], into an [N_d,N_a,N_j] array
L2star=L2d(lin);
midstar=midpoint(lin);

%% L2 flag, for the cell the chosen d landed in
% The standard GI raw's rule, with the two cell endpoints at window positions 1 and n2long in the
% chosen d's row of ReturnMatrix2. Computed from L2star BEFORE the re-spelling below.
base=dstar+N_d*n2long*(0:1:N_a-1)'+N_d*n2long*N_a*(0:1:N_j-1); % window position 1, at the chosen d
linidx_lower=base;
linidx_upper=base+N_d*(n2long-1);
isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
inLowerStrict=(L2star>=2)         & (L2star<=n2short+1);
inUpperStrict=(L2star>=n2short+3) & (L2star<=n2long-1);
Policy(4,:,:)=shiftdim(2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper),-1);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
% The standard GI raw's lines. L2 ends up in [1,n2short+2].
adjust=(L2star<1+n2short+1);
Policy(1,:,:)=shiftdim(dstar,-1);
Policy(2,:,:)=shiftdim(midstar-adjust,-1);
Policy(3,:,:)=shiftdim(adjust.*L2star+(1-adjust).*(L2star-n2short-1),-1);

% NOTE ON THE ENCODING, inherited from the standard GI raw: the window's top fine point can come out as
% either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use the
% DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
