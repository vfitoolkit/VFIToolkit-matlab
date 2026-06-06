function [V, Policy, Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_GI1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions)
% fastOLG just means parallelize over "age" (j)
% V is (a,j) (V carries Vunderbar for Sophisticated)

N_a=prod(n_a);

Vhat=zeros(N_a,N_j,'gpuArray'); % pre-Vunderbar value (snapshot of V before the beta*EV-at-policy correction)
Policy=zeros(3,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime (midpoint, aprimeL2ind, L2flag)

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

DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;
DiscountedEVinterp=reshape(beta0beta_J,[1,1,N_j]).*EVinterp;

ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn, N_j, a_grid, a_grid, ReturnFnParamsAgeMatrix,2);
% fastOLG: ReturnMatrix is [aprime,a,j]

%% beta0beta-step -- writes Policy (QH-optimal choice) and Vhat
entireRHS=ReturnMatrix+DiscountedEV; % [aprime,a,j]
[~,maxindex]=max(entireRHS,[],1);
midpoint=max(min(maxindex,n_a-1),2);
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn,N_j,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsAgeMatrix,2);
aprimej=aprimeindexes+n2aprime*jind;
EVfine=reshape(EVinterp(aprimej(:)),[n2long,N_a,N_j]);
entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimej(:)),[n2long,N_a,N_j]);
[Vhatii,maxindexL2]=max(entireRHS_ii,[],1);
Policy(1,:,:)=shiftdim(squeeze(midpoint),-1); % midpoint
Policy(2,:,:)=shiftdim(maxindexL2,-1); % aprimeL2ind
isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
Policy(3,:,:) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

%% Vunderbar: re-evaluate at Policy's aprime with beta
linidx=reshape(maxindexL2,[1,N_a*N_j])+n2long*(0:N_a*N_j-1);
EV_at_policy=reshape(EVfine(linidx),[N_a,N_j]);
V=shiftdim(Vhatii,1)+reshape(beta_J-beta0beta_J,[1,N_j]).*EV_at_policy;
Vhat=shiftdim(Vhatii,1); % snapshot pre-Vunderbar

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1);
Policy(1,:,:)=Policy(1,:,:)-adjust;
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1);


end
