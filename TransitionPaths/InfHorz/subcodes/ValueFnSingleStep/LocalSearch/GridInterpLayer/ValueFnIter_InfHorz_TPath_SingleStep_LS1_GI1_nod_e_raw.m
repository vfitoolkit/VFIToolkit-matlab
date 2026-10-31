function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_nod_e_raw(Vnext,n_a,n_z,n_e, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with the grid interpolation layer, no d, with z and iid e: a coarse max over the window, then the
% interpolation layer around the winner. ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_nod_raw with an e dimension; see
% its header for the two layers, the interior clamp (which keeps the reachable set exactly the window), and the
% encoding note. The reference is a COARSE index per (a,z,e), [1,N_a,N_z,N_e] (per current e, see
% ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_e_raw).
% Vnext is [N_a,N_z,N_e]; pi_z is [N_z,N_zprime]; pi_e is [N_e,1], the distribution of next period e.

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs; the seed sweeps are the ones that honour it.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_z,N_e]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);
nC=2*n+1; % the coarse window

Policy=zeros(3,N_a,N_z,N_e,'gpuArray'); % lower coarse point, L2, L2flag

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% Expected value (first e', then z'), and its interpolation onto the fine grid, as in the GI e raw
EV=Vnext.*shiftdim(pi_e,-2); % [N_a,N_z,N_e], aprime-by-zprime-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_z]
EV=EV.*shiftdim(pi_z',-1); % [N_a,N_zprime,N_z]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % [N_a,1,N_z]

EVinterp=interp1(a_grid,EV,aprime_grid); % [n2aprime,1,N_z]

%% The window, in COARSE grid points, one per (a,z,e)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_z,N_e], a coarse index

%% LAYER 1: the coarse max over the window
coarseindexes=shiftdim(loweredge,-1)+(0:1:2*n)'; % [nC,N_a,N_z,N_e], coarse indices
ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,n_e,a_grid(coarseindexes),a_grid,z_gridvals,e_gridvals,ReturnFnParamsVec,2);
coarsez=coarseindexes+N_a*shiftdim((0:1:N_z-1),-1); % EV does not depend on e, so no e offset
entireRHS1=ReturnMatrix1+DiscountFactorParamsVec*reshape(EV(coarsez(:)),[nC,N_a,N_z,N_e]);
[~,maxindex1]=max(entireRHS1,[],1);
coarseoffset=reshape(maxindex1,[N_a,N_z,N_e])-1; % 0 to 2n, steps above loweredge (0-based, as in every localsearch raw)

%% The new reference: the layer 1 answer, unclamped, [1,N_a,N_z,N_e]
aprimeReferencePolicyNew=reshape(loweredge+coarseoffset,[1,N_a,N_z,N_e]);

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint
midpoint=loweredge+min(max(coarseoffset,1),2*n-1); % [N_a,N_z,N_e]
aprimeindexes=shiftdim(midpoint+(midpoint-1)*n2short,-1)+(-n2short-1:1:1+n2short)'; % [n2long,N_a,N_z,N_e]
ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,n_e,aprime_grid(aprimeindexes),a_grid,z_gridvals,e_gridvals,ReturnFnParamsVec,2);
aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-1); % index into EVinterp [n2aprime,1,N_z]
entireRHS2=ReturnMatrix2+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z,N_e]);
[V,maxindexL2]=max(entireRHS2,[],1); % maxindexL2 is left as [1,N_a,N_z,N_e], the shape everything below is in
V=reshape(V,[N_a,N_z,N_e]);

%% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
% (1=all to lower, 2=usual, 3=all to upper.) The standard GI raw's rule.
aind=0:1:N_a-1;
zind=shiftdim(0:1:N_z-1,-1);
eind=shiftdim(0:1:N_e-1,-2);
linidx_lower=1      +n2long*aind+n2long*N_a*zind+n2long*N_a*N_z*eind;
linidx_upper=n2long +n2long*aind+n2long*N_a*zind+n2long*N_a*N_z*eind;
isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
inLowerStrict=(maxindexL2>=2)         & (maxindexL2<=n2short+1);
inUpperStrict=(maxindexL2>=n2short+3) & (maxindexL2<=n2long-1);
Policy(3,:,:,:)=2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2), as in the standard GI raw
Policy(1,:,:,:)=shiftdim(midpoint,-1);
Policy(2,:,:,:)=maxindexL2;
adjust=(Policy(2,:,:,:)<1+n2short+1);
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust;
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1);

end
