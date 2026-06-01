function [V,Policy,Policyalt,Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_GI1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions)
% fastOLG just means parallelize over "age" (j)
% V is (a,j) (V carries Valt for Naive)

N_d=prod(n_d);
N_a=prod(n_a);

Vtilde=zeros(N_a,N_j,'gpuArray'); % QH-optimal value (max after L2 refinement on beta0beta-step)
Policy=zeros(4,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for d & aprime (d, midpoint, aprimeL2ind, L2flag)
Policyalt=zeros(4,N_a,N_j,'gpuArray');

%% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

aBind=gpuArray(0:1:N_a-1);
jind=shiftdim(gpuArray(0:1:N_j-1),-2);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

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

DiscountedEV_alt=repelem(shiftdim(reshape(beta_J,[1,1,N_j]).*EV,-1),N_d,1,1); % [1,N_a,1,N_j], singular first dimension for d
DiscountedEV=repelem(shiftdim(reshape(beta0beta_J,[1,1,N_j]).*EV,-1),N_d,1,1);
DiscountedEVinterp_alt=repelem(reshape(beta_J,[1,1,N_j]).*EVinterp,N_d,1,1); % [N_d*n2aprime,1,N_j]
DiscountedEVinterp=repelem(reshape(beta0beta_J,[1,1,N_j]).*EVinterp,N_d,1,1);

ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_noz(ReturnFn, n_d, N_j, d_gridvals, a_grid, a_grid, ReturnFnParamsAgeMatrix,1);

%% Valt-step (beta) -- writes V and Policyalt
entireRHS_alt=ReturnMatrix+DiscountedEV_alt; %(d,aprime)-by-(a,j)
[~,maxindex1alt]=max(entireRHS_alt,[],2);
midpointalt=max(min(maxindex1alt,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
ReturnMatrix_iialt=CreateReturnFnMatrix_fastOLG_Disc_DC1_noz(ReturnFn,n_d,N_j,d_gridvals,aprime_grid(aprimeindexesalt),a_grid,ReturnFnParamsAgeMatrix,2);
daprimejalt=(1:1:N_d)'+N_d*(aprimeindexesalt-1)+N_d*n2aprime*jind;
entireRHS_iialt=ReturnMatrix_iialt+reshape(DiscountedEVinterp_alt(daprimejalt(:)),[N_d*n2long,N_a,N_j]);
[V,maxindexL2alt]=max(entireRHS_iialt,[],1);
V=squeeze(V);
d_indalt=rem(maxindexL2alt-1,N_d)+1;
allindalt=d_indalt+N_d*aBind+N_d*N_a*jBind;
Policyalt(1,:,:)=d_indalt; % d
Policyalt(2,:,:)=shiftdim(squeeze(midpointalt(allindalt)),-1); % midpoint
Policyalt(3,:,:)=shiftdim(ceil(maxindexL2alt/N_d),-1); % aprimeL2ind
L2offsetalt=ceil(maxindexL2alt/N_d);
linidx_loweralt=d_indalt                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
linidx_upperalt=d_indalt+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
isInfLoweralt=(ReturnMatrix_iialt(linidx_loweralt)==-Inf);
isInfUpperalt=(ReturnMatrix_iialt(linidx_upperalt)==-Inf);
inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
Policyalt(4,:,:)=shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

%% beta0beta-step -- writes Policy
entireRHS=ReturnMatrix+DiscountedEV; %(d,aprime)-by-(a,j)
[~,maxindex1]=max(entireRHS,[],2);
midpoint=max(min(maxindex1,n_a-1),2);
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_noz(ReturnFn,n_d,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsAgeMatrix,2);
daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind;
entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j]);
[Vtilde,maxindexL2]=max(entireRHS_ii,[],1);
Vtilde=squeeze(Vtilde);
d_ind=rem(maxindexL2-1,N_d)+1;
allind=d_ind+N_d*aBind+N_d*N_a*jBind;
Policy(1,:,:)=d_ind; % d
Policy(2,:,:)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
Policy(3,:,:)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
L2offset=ceil(maxindexL2/N_d);
linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
Policy(4,:,:)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);


%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

adjustalt=(Policyalt(3,:,:)<1+n2short+1);
Policyalt(2,:,:)=Policyalt(2,:,:)-adjustalt;
Policyalt(3,:,:)=adjustalt.*Policyalt(3,:,:)+(1-adjustalt).*(Policyalt(3,:,:)-n2short-1);


end
