function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG GI2A (no z): parallelize over "age" (j), grid-interpolation layer
% on the first endogenous state a1, iterate (via broadcasting) over a2.
% fastOLG layout (no z): V is (a,j), Policy is (channels,a,j)

N_d=prod(n_d);
N_a=prod(n_a);

% Policy: 5 channels (d, a1prime_lower, a2prime, L2ind, L2flag) — separate a1/a2
% channels so UnKronPolicyIndexes3_FHorz_noz can unpack via n_d/n_a1/n_a2 divisors.
Policy=zeros(5,N_a,N_j,'gpuArray');

%% a-split
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

%% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% precompute index helpers (all 0-based)
% For broadcasting against a1primeindexes of shape (N_d, n2long, N_a2, N_a1, N_a2, N_j):
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);  % (1,1,N_a2)             -> a2prime axis (dim 3)
jind =shiftdim(gpuArray(0:1:N_j-1),-4);   % (1,1,1,1,1,N_j)        -> j axis (dim 6)

% (a1,a2) flat index for picking midpoint per (a1,a2)
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);

% Policy-time (a,j) flat indices: Policy layout is (channel, N_a, N_j).
aBind=gpuArray(0:1:N_a-1);                % (1,N_a)
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);   % (1,1,N_j)

%% Age-matrix params and discount
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

%% Build next-period expected value (V-shift trick, no reverse_j loop, no z)
EV=zeros(N_a1,N_a2,1,1,N_j,'gpuArray');
EV(:,:,1,1,1:N_j-1)=reshape(V(:,2:end),[N_a1,N_a2,1,1,N_j-1]); % zeros at j=N_j

% Interpolate EV over a1prime_grid (interp1 acts on the first dim)
EVinterp=interp1(a1_grid,EV,a1prime_grid); % (N_a1fine, N_a2, 1, 1, N_j)

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV;             % (N_a1, N_a2, 1, 1, N_j)
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EVinterp; % (N_a1fine, N_a2, 1, 1, N_j)

%% Level 1: coarse search to get a1prime midpoint
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC2A_noz(ReturnFn, n_d, N_j, d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, ReturnFnParamsAgeMatrix, 1);
% Shape: (d, a1prime, a2prime, a1, a2, j)

entireRHS=ReturnMatrix+shiftdim(DiscountedEV,-1); % shiftdim(-1) puts a leading singleton-d; (1, N_a1, N_a2, 1, 1, N_j) broadcasts against ReturnMatrix

% argmax over a1prime: result shape (d, 1, a2prime, a1, a2, j)
[~,maxindex]=max(entireRHS,[],2);

% Turn this into the 'midpoint'
midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
% midpoint is (n_d, 1, n_a2, n_a1, n_a2, N_j)
a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % a1prime points either side of midpoint
% a1primeindexes shape: (n_d, n2long, n_a2, n_a1, n_a2, N_j)

%% Level 2: fine search across the n2long neighbourhood
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_noz(ReturnFn, n_d, N_j, d_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, ReturnFnParamsAgeMatrix, 2);
% Shape: (N_d*n2long*N_a2, N_a, N_j)

% Index into EVinterp (N_a1fine, N_a2, 1, 1, N_j) — strides: 1, N_a1fine, _, _, N_a1fine*N_a2
aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind;
entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_j]);

[Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
maxindexL2d=rem(maxindexL2-1,N_d)+1;
maxindexL2a=ceil(maxindexL2/N_d);
maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
maxindexL2a2=ceil(maxindexL2a/n2long);

V=reshape(Vtempii,[N_a,N_j]);

% Map back to midpoint per (a,j): linear index into midpoint(d,1,a2prime,a1,a2,j) treated as (d,a2prime,a,j)
midpoint_pick=maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*jBind;
% midpoint_pick shape: (1, N_a, N_j)
a1prime_midpoint=midpoint(midpoint_pick); % (1, N_a, N_j)

% L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d, a2prime)
linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aBind + N_d*n2long*N_a2*N_a*jBind;
linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aBind + N_d*n2long*N_a2*N_a*jBind;
isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
PolicyL2flag  = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper); % shape (1, N_a, N_j)

%% Fold a1prime midpoint and L2ind into 'lower grid point + offset' convention
adjust=(maxindexL2a1<1+n2short+1);
a1prime_lower=a1prime_midpoint-adjust;
a1prime_L2  =adjust.*maxindexL2a1+(1-adjust).*(maxindexL2a1-n2short-1);

%% Pack Policy (5 channels: d, a1prime_lower, a2prime, L2ind, L2flag)
Policy(1,:,:)=maxindexL2d;
Policy(2,:,:)=a1prime_lower;
Policy(3,:,:)=maxindexL2a2;
Policy(4,:,:)=a1prime_L2;
Policy(5,:,:)=PolicyL2flag;


end
