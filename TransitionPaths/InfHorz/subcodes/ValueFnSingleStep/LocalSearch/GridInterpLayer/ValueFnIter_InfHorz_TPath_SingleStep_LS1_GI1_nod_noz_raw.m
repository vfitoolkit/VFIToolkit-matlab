function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_nod_noz_raw(Vnext,n_a, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with the grid interpolation layer, no d, no z, no e: a coarse max over the window, then the
% interpolation layer around the winner. ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_nod_raw without the z dimension;
% see its header for the two layers, the interior clamp (which keeps the reachable set exactly the window), and the
% encoding note. The reference is a COARSE index per a, [1,N_a].
% Vnext is [N_a,1].

N_a=prod(n_a);

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[1,N_a]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

Policy=zeros(3,N_a,'gpuArray'); % lower coarse point, L2, L2flag

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

EV=Vnext; % [N_a,1], no exogenous state so nothing to take expectations over
EVinterp=interp1(a_grid,EV,aprime_grid);

%% The window, in COARSE grid points, one per a
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [1,N_a], a coarse index

%% LAYER 1: the coarse max over the window
coarseindexes=loweredge+(0:1:2*n)'; % [nC,N_a], coarse indices
ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,a_grid(coarseindexes),a_grid,ReturnFnParamsVec); % [nC,N_a]
entireRHS1=ReturnMatrix1+DiscountFactorParamsVec*EV(coarseindexes);
[~,maxindex1]=max(entireRHS1,[],1);
coarseoffset=maxindex1-1; % [1,N_a], 0 to 2n, steps above loweredge (0-based, as in every localsearch raw)

%% The new reference: the layer 1 answer, unclamped, [1,N_a]
aprimeReferencePolicyNew=loweredge+coarseoffset;

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint
midpoint=loweredge+min(max(coarseoffset,1),2*n-1); % [1,N_a]
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % [n2long,N_a]
ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec); % [n2long,N_a]
entireRHS2=ReturnMatrix2+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
[V,maxindexL2]=max(entireRHS2,[],1); % [1,N_a]
V=reshape(V,[N_a,1]);

%% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
% (1=all to lower, 2=usual, 3=all to upper.) The standard GI raw's rule.
aind=0:1:N_a-1;
linidx_lower=1      +n2long*aind;
linidx_upper=n2long +n2long*aind;
isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
inLowerStrict=(maxindexL2>=2)         & (maxindexL2<=n2short+1);
inUpperStrict=(maxindexL2>=n2short+3) & (maxindexL2<=n2long-1);
Policy(3,:)=2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2), as in the standard GI raw
Policy(1,:)=midpoint;
Policy(2,:)=maxindexL2;
adjust=(Policy(2,:)<1+n2short+1);
Policy(1,:)=Policy(1,:)-adjust;
Policy(2,:)=adjust.*Policy(2,:)+(1-adjust).*(Policy(2,:)-n2short-1);

end
