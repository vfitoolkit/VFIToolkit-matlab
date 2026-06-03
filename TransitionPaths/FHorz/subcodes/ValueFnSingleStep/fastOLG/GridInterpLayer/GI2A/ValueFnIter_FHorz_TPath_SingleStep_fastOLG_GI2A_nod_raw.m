function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG GI2A (no d): parallelize over "age" (j), grid-interpolation layer
% on the first endogenous state a1, iterate (via broadcasting) over a2.
% fastOLG layout: V is (a,j)-by-z, Policy is (channels,a,j,z)
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG

N_a=prod(n_a);
N_z=prod(n_z);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported')
end

% Policy: 4 channels (a1prime_lower, a2prime, L2ind, L2flag) — separate a1/a2
% channels so UnKronPolicyIndexes2_FHorz can unpack via n_a1/n_a2 divisors.
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
z_gridvals_J=shiftdim(z_gridvals_J,-4); % [1,1,1,1,N_j,N_z,l_z]

% precompute index helpers (all 0-based)
% For broadcasting against a1primeindexes of shape (n2long, N_a2, N_a1, N_a2, N_j, N_z):
a2ind=gpuArray(0:1:N_a2-1);               % (1,N_a2)              -> a2prime axis (dim 2)
jind =shiftdim(gpuArray(0:1:N_j-1),-3);   % (1,1,1,1,N_j)         -> j axis (dim 5)
zBind=shiftdim(gpuArray(0:1:N_z-1),-4);   % (1,1,1,1,1,N_z)       -> z axis (dim 6)

% (a1,a2) flat index for picking midpoint per (a1,a2)
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);

% Policy-time (a,j,z) flat indices: Policy layout is (channel, N_a, N_j, N_z).
aBind=gpuArray(0:1:N_a-1);                % (1,N_a)
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);   % (1,1,N_j)
zind =shiftdim(gpuArray(0:1:N_z-1),-2);   % (1,1,1,N_z)

%% Age-matrix params and discount
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

%% Build next-period expected value (V-shift trick, no reverse_j loop)
if vfoptions.EVpre==0
    EVpre=zeros(N_a,1,N_j,N_z);
    EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % zeros at j=N_j
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,4),[N_a1,N_a2,1,1,N_j,N_z]); % (a1prime,a2prime,1,1,j,z)
elseif vfoptions.EVpre==1
    EV=reshape(V,[N_a,1,N_j,N_z]).*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,4),[N_a1,N_a2,1,1,N_j,N_z]);
end

% Interpolate EV over a1prime_grid (interp1 acts on the first dim)
EVinterp=interp1(a1_grid,EV,a1prime_grid); % (N_a1fine, N_a2, 1, 1, N_j, N_z)

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV;             % (N_a1, N_a2, 1, 1, N_j, N_z)
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EVinterp; % (N_a1fine, N_a2, 1, 1, N_j, N_z)

if vfoptions.lowmemory==0
    %% Level 1: coarse search to get a1prime midpoint
    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, n_z, N_j, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix, 1);
    % Shape: (a1prime, a2prime, a1, a2, j, z)

    entireRHS=ReturnMatrix+DiscountedEV; % EV is (a1prime,a2prime,1,1,j,z) — same shape (with broadcasting on dims 3,4)

    % argmax over a1prime: result shape (1, a2prime, a1, a2, j, z)
    [~,maxindex]=max(entireRHS,[],1);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is (1, n_a2, n_a1, n_a2, N_j, n_z)
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % a1prime points either side of midpoint (column vector)
    % a1primeindexes shape: (n2long, n_a2, n_a1, n_a2, N_j, n_z)

    %% Level 2: fine search across the n2long neighbourhood
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, n_z, N_j, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix, 2);
    % Shape: (n2long*N_a2, N_a, N_j, N_z)

    % Index into EVinterp (N_a1fine, N_a2, 1, 1, N_j, N_z) — strides: 1, N_a1fine, _, _, N_a1fine*N_a2, N_a1fine*N_a2*N_j
    aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind+N_a1fine*N_a2*N_j*zBind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a,N_j,N_z]);

    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    % maxindexL2 indexes into (a1prime fine, a2prime) within the n2long window
    maxindexL2a1=rem(maxindexL2-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2/n2long);

    V=reshape(Vtempii,[N_a*N_j,N_z]);

    % Map back to midpoint per (a,j,z): linear index into midpoint(1,a2prime,a1,a2,j,z) treated as (a2prime,a,j,z)
    midpoint_pick=maxindexL2a2+N_a2*a12ind+N_a2*N_a*jBind+N_a2*N_a*N_j*zind;
    % midpoint_pick shape: (1, N_a, N_j, N_z)
    a1prime_midpoint=midpoint(midpoint_pick); % (1, N_a, N_j, N_z)

    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
    % ReturnMatrix_ii row index for (a1prime fine=1 or n2long, a2prime) = 1+n2long*(a2prime-1)  or  n2long*a2prime
    linidx_lower = 1                  + n2long*(maxindexL2a2-1) + n2long*N_a2*aBind + n2long*N_a2*N_a*jBind + n2long*N_a2*N_a*N_j*zind;
    linidx_upper = n2long*maxindexL2a2                          + n2long*N_a2*aBind + n2long*N_a2*N_a*jBind + n2long*N_a2*N_a*N_j*zind;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    PolicyL2flag  = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper); % shape (1, N_a, N_j, N_z)

    %% Fold a1prime midpoint and L2ind into 'lower grid point + offset' convention (fastOLG_GI1_nod style)
    adjust=(maxindexL2a1<1+n2short+1);
    a1prime_lower=a1prime_midpoint-adjust;
    a1prime_L2  =adjust.*maxindexL2a1+(1-adjust).*(maxindexL2a1-n2short-1);

    %% Pack Policy (no d): 4 channels = a1prime_lower, a2prime, L2ind, L2flag
    Policy(1,:,:,:)=a1prime_lower;
    Policy(2,:,:,:)=maxindexL2a2;
    Policy(3,:,:,:)=a1prime_L2;
    Policy(4,:,:,:)=PolicyL2flag;
elseif vfoptions.lowmemory==1
    V=zeros(N_a*N_j,N_z,'gpuArray');
    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,1,:,z_c,:); % [1,1,1,1,N_j,1,l_z]
        DiscountedEV_z=DiscountedEV(:,:,:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,z_c);

        ReturnMatrix_z=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_z, N_j, a1_grid, a2_grid, a1_grid, a2_grid, z_vals, ReturnFnParamsAgeMatrix, 1);
        entireRHS_z=ReturnMatrix_z+DiscountedEV_z;
        [~,maxindex]=max(entireRHS_z,[],1);
        midpoint=max(min(maxindex,n_a1-1),2);
        % midpoint is (1, n_a2, n_a1, n_a2, N_j)
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        % a1primeindexes shape: (n2long, n_a2, n_a1, n_a2, N_j)

        ReturnMatrix_ii_z=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_z, N_j, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_vals, ReturnFnParamsAgeMatrix, 2);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind;
        entireRHS_ii_z=ReturnMatrix_ii_z+reshape(DiscountedEVinterp_z(aprime),[n2long*N_a2,N_a,N_j]);

        [Vtempii_z,maxindexL2]=max(entireRHS_ii_z,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V(:,z_c)=reshape(Vtempii_z,[N_a*N_j,1]);

        midpoint_pick=maxindexL2a2+N_a2*a12ind+N_a2*N_a*jBind;
        a1prime_midpoint=midpoint(midpoint_pick); % (1, N_a, N_j)

        linidx_lower = 1                  + n2long*(maxindexL2a2-1) + n2long*N_a2*aBind + n2long*N_a2*N_a*jBind;
        linidx_upper = n2long*maxindexL2a2                          + n2long*N_a2*aBind + n2long*N_a2*N_a*jBind;
        isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag  = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        adjust=(maxindexL2a1<1+n2short+1);
        a1prime_lower=a1prime_midpoint-adjust;
        a1prime_L2  =adjust.*maxindexL2a1+(1-adjust).*(maxindexL2a1-n2short-1);

        Policy(1,:,:,z_c)=a1prime_lower;
        Policy(2,:,:,z_c)=maxindexL2a2;
        Policy(3,:,:,z_c)=a1prime_L2;
        Policy(4,:,:,z_c)=PolicyL2flag;
    end
end


end
