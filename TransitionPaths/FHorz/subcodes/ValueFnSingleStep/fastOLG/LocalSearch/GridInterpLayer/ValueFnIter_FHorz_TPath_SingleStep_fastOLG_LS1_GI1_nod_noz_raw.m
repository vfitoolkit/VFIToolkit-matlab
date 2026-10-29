function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with the grid interpolation layer, finite horizon, fastOLG (all ages at once), no d, no z.
% The V input is next period value fn (across all ages), the V output is this period.
% The fastOLG version of slowOLG/LocalSearch/GridInterpLayer/ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_nod_raw,
% in TWO LAYERS: a coarse max over the window, then the interpolation layer one coarse cell either side
% of that argmax CLAMPED to [loweredge+1,loweredge+2n-1]. See that raw for the reasoning.
%
% THE REFERENCE is a COARSE aprime index, [1,N_a,N_j]. Layer 1's answer IS the new reference, the
% UNCLAMPED coarse argmax.
% Policy comes out in the SAME form as the unrestricted GI raw, [3,N_a,N_j]: (lower coarse point, L2,
% L2flag), so the dispatcher's UnKron applies unchanged.
%
% EV and EVinterp are built EXACTLY as ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_nod_noz_raw does: the
% j=N_j slice is left as zeros, so the terminal age needs no special case. That raw has no EVpre branch;
% local search is only reachable from a transition path, where EVpre=0 anyway.

N_a=prod(n_a);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_j]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);
nC=2*n+1; % the coarse window

Policy=zeros(3,N_a,N_j,'gpuArray'); % lower coarse point, L2, L2flag

jind=shiftdim(gpuArray(0:1:N_j-1),-1);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=V(:,2:end);
% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

%% The window, in COARSE grid points, for every (a,j) at once
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_j], a coarse index
loweredge=reshape(loweredge,[1,N_a,N_j]);

%% LAYER 1: the coarse max over the window, on the COARSE EV
coarseindexes=loweredge+(0:1:2*n)'; % [nC,N_a,N_j], coarse indices
ReturnMatrix1=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn,N_j,a_grid(coarseindexes),a_grid,ReturnFnParamsAgeMatrix,2);
coarsej=coarseindexes+N_a*jind;
% beta_j*EV(aprime,j), the same products as the GI raw's reshape(DiscountFactor_J,[1,1,N_j]).*EV
entireRHS1=ReturnMatrix1+reshape(DiscountFactor_J,[1,1,N_j]).*reshape(EV(coarsej(:)),[nC,N_a,N_j]);
[~,maxindex1]=max(entireRHS1,[],1);
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=maxindex1-1; % [1,N_a,N_j], 0 to 2n, steps above loweredge

% The new reference: THE LAYER 1 ANSWER, taken straight, UNCLAMPED
aprimeReferencePolicyNew=loweredge+coarseoffset; % [1,N_a,N_j]

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint
midpoint=loweredge+min(max(coarseoffset,1),2*n-1); % [1,N_a,N_j]
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % [n2long,N_a,N_j]
ReturnMatrix2=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn,N_j,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsAgeMatrix,2);
aprimej=aprimeindexes+n2aprime*jind;
% Discounting applied AFTER the gather, the same order the GI raw uses
entireRHS2=ReturnMatrix2+reshape(DiscountFactor_J,[1,1,N_j]).*reshape(EVinterp(aprimej(:)),[n2long,N_a,N_j]);
[Vtempii,maxindexL2]=max(entireRHS2,[],1);
V=shiftdim(Vtempii,1);

%% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
% (1=all to lower, 2=usual, 3=all to upper.) The standard GI raw's rule; computed from maxindexL2
% BEFORE the re-spelling below.
isInfLower    = (ReturnMatrix2(1,     :,:) == -Inf);
isInfUpper    = (ReturnMatrix2(n2long,:,:) == -Inf);
inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
Policy(3,:,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
% The standard GI raw's lines. L2 ends up in [1,n2short+2].
adjust=(maxindexL2<1+n2short+1);
Policy(1,:,:)=midpoint-adjust;
Policy(2,:,:)=adjust.*maxindexL2+(1-adjust).*(maxindexL2-n2short-1);

% NOTE ON THE ENCODING, inherited from the standard GI raw: the window's top fine point can come out as
% either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2). Anything comparing these raws must use the
% DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
