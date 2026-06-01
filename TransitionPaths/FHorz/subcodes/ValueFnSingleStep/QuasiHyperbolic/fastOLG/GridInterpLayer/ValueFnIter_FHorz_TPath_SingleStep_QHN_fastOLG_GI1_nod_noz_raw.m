function [V, Policy, Policyalt, Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_GI1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions)
% fastOLG just means parallelize over "age" (j)
% V is (a,j) (V carries Valt for Naive)

N_a=prod(n_a);

Vtilde=zeros(N_a,N_j,'gpuArray'); % QH-optimal value (max after L2 refinement on beta0beta-step)
Policy=zeros(3,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime (midpoint, aprimeL2ind, L2flag)
Policyalt=zeros(3,N_a,N_j,'gpuArray');

%%

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,{vfoptions.QHadditionaldiscount},N_j);
beta0beta_J=beta0_J.*beta_J; % Discount factor between today and tomorrow.

EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=V(:,2:end);
% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV_alt=reshape(beta_J,[1,1,N_j]).*EV;
DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;
DiscountedEVinterp_alt=reshape(beta_J,[1,1,N_j]).*EVinterp;
DiscountedEVinterp=reshape(beta0beta_J,[1,1,N_j]).*EVinterp;

ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn, N_j, a_grid, a_grid, ReturnFnParamsAgeMatrix,2);
% fastOLG: ReturnMatrix is [aprime,a,j]

%% Valt-step (beta) -- writes V and Policyalt
entireRHS_alt=ReturnMatrix+DiscountedEV_alt; % [aprime,a,j]
[~,maxindexalt]=max(entireRHS_alt,[],1);
midpointalt=max(min(maxindexalt,n_a-1),2);
aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
ReturnMatrix_iialt=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn,N_j,aprime_grid(aprimeindexesalt),a_grid,ReturnFnParamsAgeMatrix,2);
aprimejalt=aprimeindexesalt+n2aprime*jind;
entireRHS_iialt=ReturnMatrix_iialt+reshape(DiscountedEVinterp_alt(aprimejalt(:)),[n2long,N_a,N_j]);
[Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
V=shiftdim(Vtempii,1);
Policyalt(1,:,:)=shiftdim(squeeze(midpointalt),-1); % midpoint
Policyalt(2,:,:)=shiftdim(maxindexL2alt,-1); % aprimeL2ind
isInfLoweralt    = (ReturnMatrix_iialt(1,     :,:) == -Inf);
isInfUpperalt    = (ReturnMatrix_iialt(n2long,:,:) == -Inf);
inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
Policyalt(3,:,:) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

%% beta0beta-step -- writes Policy
entireRHS=ReturnMatrix+DiscountedEV; % [aprime,a,j]
[~,maxindex]=max(entireRHS,[],1);
midpoint=max(min(maxindex,n_a-1),2);
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn,N_j,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsAgeMatrix,2);
aprimej=aprimeindexes+n2aprime*jind;
entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimej(:)),[n2long,N_a,N_j]);
[Vtildeii,maxindexL2]=max(entireRHS_ii,[],1);
Vtilde=shiftdim(Vtildeii,1);
Policy(1,:,:)=shiftdim(squeeze(midpoint),-1); % midpoint
Policy(2,:,:)=shiftdim(maxindexL2,-1); % aprimeL2ind
isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
Policy(3,:,:) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:)=Policy(1,:,:)-adjust; % lower grid point
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

adjustalt=(Policyalt(2,:,:)<1+n2short+1);
Policyalt(1,:,:)=Policyalt(1,:,:)-adjustalt;
Policyalt(2,:,:)=adjustalt.*Policyalt(2,:,:)+(1-adjustalt).*(Policyalt(2,:,:)-n2short-1);


end
