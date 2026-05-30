function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG GI2A: parallelize over "age" (j), grid-interpolation layer on the
% first endogenous state a1 (midpoint + n2long fine search), iterate (via
% broadcasting) over the second endogenous state a2.
% fastOLG layout: V is (a,j)-by-z, Policy is (channels,a,j,z)
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG

if vfoptions.lowmemory>0
    error('vfoptions.lowmemory>0 not supported for ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_raw')
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% Policy: 4 channels (d, aprime_kron_lower, L2ind, L2flag), matching the
% fastOLG_GI1 convention so the downstream UnKron sees a uniform shape.
Policy=zeros(4,N_a,N_j,N_z,'gpuArray');

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

% Pre-shift z_gridvals_J so the fastOLG DC2A return-fn helper can index it directly.
z_gridvals_J=shiftdim(z_gridvals_J,-5); % [1,1,1,1,1,N_j,N_z,l_z]

% precompute index helpers (all 0-based)
% For broadcasting against a1primeindexes of shape (N_d, n2long, N_a2, N_a1, N_a2, N_j, N_z):
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);  % (1,1,N_a2)             -> a2prime axis (dim 3)
jind =shiftdim(gpuArray(0:1:N_j-1),-4);   % (1,1,1,1,1,N_j)        -> j axis (dim 6)
zBind=shiftdim(gpuArray(0:1:N_z-1),-5);   % (1,1,1,1,1,1,N_z)      -> z axis (dim 7)

% (a1,a2) flat index for picking midpoint per (a1,a2)
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);

% Policy-time (a,j,z) flat indices: Policy layout is (channel, N_a, N_j, N_z), so
% these need to broadcast against tensors of shape (1, N_a, N_j, N_z).
aBind=gpuArray(0:1:N_a-1);                % (1,N_a)
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);   % (1,1,N_j)
zind =shiftdim(gpuArray(0:1:N_z-1),-2);   % (1,1,1,N_z)

%% Age-matrix params and discount
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

%% Build next-period expected value (V-shift trick, no reverse_j loop)
if vfoptions.EVpre==0
    EVpre=zeros(N_a,1,N_j,N_z);
    EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % zeros at j=N_j (terminal age has no continuation in TPath)
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; % -Inf*0 = NaN, replace with 0 (the 0 comes from transition prob)
    EV=reshape(sum(EV,4),[N_a1,N_a2,1,1,N_j,N_z]); % (a1prime,a2prime,1,1,j,z); singletons broadcast against (a1,a2)
elseif vfoptions.EVpre==1
    % 'Matched Expectations Path': input V is already E[V'|.] across z'
    EV=reshape(V,[N_a,1,N_j,N_z]).*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,4),[N_a1,N_a2,1,1,N_j,N_z]);
end

% Interpolate EV over a1prime_grid (interp1 acts on the first dim)
EVinterp=interp1(a1_grid,EV,a1prime_grid); % (N_a1fine, N_a2, 1, 1, N_j, N_z)

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV;             % (N_a1, N_a2, 1, 1, N_j, N_z)
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EVinterp; % (N_a1fine, N_a2, 1, 1, N_j, N_z)

%% Level 1: coarse search to get a1prime midpoint
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, n_z, N_j, d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix, 1);
% Shape: (d, a1prime, a2prime, a1, a2, j, z)

entireRHS=ReturnMatrix+shiftdim(DiscountedEV,-1); % shiftdim(-1) puts a leading singleton-d; (1, N_a1, N_a2, 1, 1, N_j, N_z) broadcasts against ReturnMatrix

% argmax over a1prime: result shape (d, 1, a2prime, a1, a2, j, z)
[~,maxindex]=max(entireRHS,[],2);

% Turn this into the 'midpoint'
midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
% midpoint is (n_d, 1, n_a2, n_a1, n_a2, N_j, n_z)
a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % a1prime points either side of midpoint
% a1primeindexes shape: (n_d, n2long, n_a2, n_a1, n_a2, N_j, n_z)

%% Level 2: fine search across the n2long neighbourhood
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, n_z, N_j, d_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix, 2);
% Shape: (N_d*n2long*N_a2, N_a, N_j, N_z)

% Indices into EVinterp: EVinterp is (N_a1fine, N_a2, 1, 1, N_j, N_z) but the d-broadcast is implicit.
aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind+N_a1fine*N_a2*N_j*zBind;
entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_j,N_z]);

[Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
% maxindexL2 indexes into (d, a1prime fine, a2prime) within the n2long window
maxindexL2d=rem(maxindexL2-1,N_d)+1;
maxindexL2a=ceil(maxindexL2/N_d);
maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
maxindexL2a2=ceil(maxindexL2a/n2long);

V=reshape(Vtempii,[N_a*N_j,N_z]);

% Map back to midpoint per (a,j,z): linear index into midpoint(d,1,a2prime,a1,a2,j,z) treated as (d,a2prime,a,j,z)
midpoint_pick=maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*jBind+N_d*N_a2*N_a*N_j*zind;
% midpoint_pick shape: (1, N_a, N_j, N_z)
a1prime_midpoint=midpoint(midpoint_pick); % (1, N_a, N_j, N_z)

% L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d, a2prime)
linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aBind + N_d*n2long*N_a2*N_a*jBind + N_d*n2long*N_a2*N_a*N_j*zind;
linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aBind + N_d*n2long*N_a2*N_a*jBind + N_d*n2long*N_a2*N_a*N_j*zind;
isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
PolicyL2flag  = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper); % shape (1, N_a, N_j, N_z)

%% Fold a1prime midpoint and L2ind into the same 'lower grid point + offset' convention as fastOLG_GI1
% Currently the midpoint is the inner-point label and maxindexL2a1 indexes the n2long fine window.
% Switch a1prime midpoint to 'lower coarse a1 grid point' and remap maxindexL2a1 to 1..(n2short+2) counting up from the lower point.
adjust=(maxindexL2a1<1+n2short+1); % if second layer is choosing below midpoint
a1prime_lower=a1prime_midpoint-adjust;                                      % lower coarse a1 grid point (1..N_a1)
a1prime_L2  =adjust.*maxindexL2a1+(1-adjust).*(maxindexL2a1-n2short-1);     % 1 (lower) to n2short+2 (upper)

%% Pack Policy into the 4-channel form expected by UnKronPolicyIndexes_Case1_FHorz (with L2flag pilot)
Policy(1,:,:,:)=maxindexL2d; % d
Policy(2,:,:,:)=a1prime_lower+N_a1*(maxindexL2a2-1); % aprime
Policy(3,:,:,:)=a1prime_L2; % a1primeL2
Policy(4,:,:,:)=PolicyL2flag; % L2flag


end
