function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_GI1_noz_raw(V,n_d1,n_d2,n_a1,n_a2,N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;

Policy3=zeros(4,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for d and a1prime rest of dimensions a (d, midpoint, a1primeL2ind, L2flag)

%% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

d2ind=repelem(gpuArray(1:1:N_d2)',N_d1,1);
aind=shiftdim(gpuArray(0:1:N_a-1),-2);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
jind=shiftdim(gpuArray(0:1:N_j-1),-3);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-4);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_J(aprimeFn, n_d2, n_a2, N_j, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_j], whereas aprimeProbs is [N_d2,N_a2,N_j]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat((a2primeIndex-1),N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_j]

    EVpre=[V(N_a+1:end); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations

    % Need to add the indexes for j to the aprimeIndex, remember fastOLG so V is (a,j)-by-1
    Vlower=reshape(EVpre(aprimeIndex+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    Vupper=reshape(EVpre(aprimeplus1Index+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,N_j)
    % Already applied the probabilities from interpolating onto grid

    EV=reshape(sum(EV,4),[N_d2*N_a1,N_a2,N_j]); % (aprime,1,j), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_J(aprimeFn, n_d2, n_a2, N_j, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_j], whereas aprimeProbs is [N_d2,N_a2,N_j]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat((a2primeIndex-1),N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_j]

    % Need to add the indexes for j to the aprimeIndex, remember fastOLG so V is (a,j)-by-1
    Vlower=reshape(V(aprimeIndex+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    Vupper=reshape(V(aprimeplus1Index+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,N_j)
    % Already applied the probabilities from interpolating onto grid

    EV=reshape(sum(EV,4),[N_d2*N_a1,N_a2,N_j]); % (aprime,1,j), 2nd dim will be autofilled with a
end


DiscountedEV=DiscountFactorParamsVec.*reshape(EV,[N_d2,N_a1,1,N_a2,N_j]);
% Interpolate EV over a1prime_grid
DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);   % [N_d2,N_a1prime,1,N_a2,N_j]

ReturnMatrix=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_noz(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2,N_j, d_gridvals, a1_gridvals, a1_gridvals, a2_grid, ReturnFnParamsAgeMatrix,1,0); % Level=1, Refine=0
% fastOLG: ReturnMatrix is [N_d,N_a1prime,N_a1,N_a2,N_j]

entireRHS=ReturnMatrix+DiscountedEV;

% First, we want a1prime conditional on (d,1,a1,a2,j)
[~,maxindex1]=max(entireRHS,[],2);

% Turn this into the 'midpoint'
midpoint=max(min(maxindex1,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
% midpoint is N_d-by-1-by-N_a1-by-N_a2-by-N_j
a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
% aprime possibilities are N_d-by-n2long-by-N_a1-by-N_a2-by-N_j
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_noz(ReturnFn, n_d1, n_d2, n2long, n_a1,n_a2,N_j, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_grid, ReturnFnParamsAgeMatrix,2,0); % Level=2, Refine=0
d2aprimej=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*jind;
entireRHS_ii=ReturnMatrix_ii+DiscountedEVinterp(reshape(d2aprimej,[N_d*n2long,N_a1*N_a2,N_j]));
[Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
V=shiftdim(Vtempii,1);
dind=rem(maxindexL2-1,N_d)+1;
allind=reshape(dind,[1,1,1,N_a1*N_a2,N_j])+N_d*aind+N_d*N_a*jind; % midpoint is N_d-by-1-by-N_a1-by-N_a2-by-N_j
allind=reshape(allind,[1,N_a1*N_a2,N_j]);
Policy3(1,:,:)=dind; % d
Policy3(2,:,:)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
Policy3(3,:,:)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
L2offset=ceil(maxindexL2/N_d);
linidx_lower=reshape(dind,[1,N_a1*N_a2,N_j])                  +N_d*n2long*shiftdim(gpuArray(0:1:N_a-1),-1)+N_d*n2long*N_a*shiftdim(gpuArray(0:1:N_j-1),-2);
linidx_upper=reshape(dind,[1,N_a1*N_a2,N_j])+N_d*(n2long-1)   +N_d*n2long*shiftdim(gpuArray(0:1:N_a-1),-1)+N_d*n2long*N_a*shiftdim(gpuArray(0:1:N_j-1),-2);
isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
Policy3(4,:,:)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

%% fastOLG without z, so need to output to take certain shapes
V=reshape(V,[N_a*N_j,1]);


%% With grid interpolation, switch from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy3(3,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy3(2,:,:)=Policy3(2,:,:)-adjust; % lower grid point
Policy3(3,:,:)=adjust.*Policy3(3,:,:)+(1-adjust).*(Policy3(3,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=Policy3;



end
