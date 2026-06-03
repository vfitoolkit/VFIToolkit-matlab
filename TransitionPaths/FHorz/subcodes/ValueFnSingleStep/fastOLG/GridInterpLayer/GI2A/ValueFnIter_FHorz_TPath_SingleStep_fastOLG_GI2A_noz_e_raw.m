function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG GI2A (no z, has e): parallelize over "age" (j), grid-interpolation
% layer on the first endogenous state a1, iterate (via broadcasting) over a2.
% fastOLG layout (noz_e): V is (a*j)-by-e, Policy is (channels, a, j, e)
% pi_e_J is (a*j, e) for fastOLG (i.i.d. e)
% e_gridvals_J is (j, N_e, l_e) for fastOLG

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported')
end

% Policy: 5 channels (d, a1prime_lower, a2prime, L2ind, L2flag) — separate a1/a2
% channels so UnKronPolicyIndexes3_FHorz_z can unpack via n_d/n_a1/n_a2 divisors.
Policy=zeros(5,N_a,N_j,N_e,'gpuArray');

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

% Pre-shift e_gridvals_J so the fastOLG DC2A return-fn helper (treating e as
% its 'z') can index it directly: helper expects [1,1,1,1,1,N_j,N_z=N_e,l_z=l_e].
e_gridvals_J=shiftdim(e_gridvals_J,-5); % [1,1,1,1,1,N_j,N_e,l_e]

% precompute index helpers (all 0-based)
% For broadcasting against a1primeindexes of shape (N_d, n2long, N_a2, N_a1, N_a2, N_j, N_e):
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);  % (1,1,N_a2)             -> a2prime axis (dim 3)
jind =shiftdim(gpuArray(0:1:N_j-1),-4);   % (1,1,1,1,1,N_j)        -> j axis (dim 6)
eBind=shiftdim(gpuArray(0:1:N_e-1),-5);   % (1,1,1,1,1,1,N_e)      -> e axis (dim 7)

% (a1,a2) flat index for picking midpoint per (a1,a2)
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);

% Policy-time (a,j,e) flat indices: Policy layout is (channel, N_a, N_j, N_e).
aBind=gpuArray(0:1:N_a-1);                % (1,N_a)
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);   % (1,1,N_j)
eind =shiftdim(gpuArray(0:1:N_e-1),-2);   % (1,1,1,N_e)

%% Age-matrix params and discount
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

%% Build next-period expected value (V-shift trick, no reverse_j loop)
% V is (N_a*N_j, N_e); e is i.i.d., so EV depends only on (a',j) once we
% integrate over e' using pi_e_J.
if vfoptions.EVpre==0
    EVpre=[sum(V(N_a+1:end,:).*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % zeros at j=N_j (terminal age has no continuation in TPath)
    EV=reshape(EVpre,[N_a1,N_a2,1,1,N_j]); % (a1prime,a2prime,1,1,j)
elseif vfoptions.EVpre==1
    % 'Matched Expectations Path': input V is already E[V'|.] across e'
    EV=reshape(V,[N_a1,N_a2,1,1,N_j]);
end

% Interpolate EV over a1prime_grid (interp1 acts on the first dim)
EVinterp=interp1(a1_grid,EV,a1prime_grid); % (N_a1fine, N_a2, 1, 1, N_j)

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV;             % (N_a1, N_a2, 1, 1, N_j)
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EVinterp; % (N_a1fine, N_a2, 1, 1, N_j)

if vfoptions.lowmemory==0
    %% Level 1: coarse search to get a1prime midpoint
    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, n_e, N_j, d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, e_gridvals_J, ReturnFnParamsAgeMatrix, 1);
    % Shape: (d, a1prime, a2prime, a1, a2, j, e)

    entireRHS=ReturnMatrix+shiftdim(DiscountedEV,-1); % shiftdim(-1) puts a leading singleton-d; (1, N_a1, N_a2, 1, 1, N_j) broadcasts against ReturnMatrix (trailing e dim auto-broadcasts since iid)

    % argmax over a1prime: result shape (d, 1, a2prime, a1, a2, j, e)
    [~,maxindex]=max(entireRHS,[],2);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is (n_d, 1, n_a2, n_a1, n_a2, N_j, n_e)
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % a1prime points either side of midpoint
    % a1primeindexes shape: (n_d, n2long, n_a2, n_a1, n_a2, N_j, n_e)

    %% Level 2: fine search across the n2long neighbourhood
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, n_e, N_j, d_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, e_gridvals_J, ReturnFnParamsAgeMatrix, 2);
    % Shape: (N_d*n2long*N_a2, N_a, N_j, N_e)

    % Index into EVinterp (N_a1fine, N_a2, 1, 1, N_j) — strides: 1, N_a1fine, _, _, N_a1fine*N_a2.
    % E[V] does not depend on e' (i.i.d.), but aprime does (midpoint inherits e from maxindex), so the lookup is per-e.
    aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_j,N_e]);

    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);

    V=reshape(Vtempii,[N_a*N_j,N_e]);

    % Map back to midpoint per (a,j,e): linear index into midpoint(d,1,a2prime,a1,a2,j,e) treated as (d,a2prime,a,j,e)
    midpoint_pick=maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*jBind+N_d*N_a2*N_a*N_j*eind;
    % midpoint_pick shape: (1, N_a, N_j, N_e)
    a1prime_midpoint=midpoint(midpoint_pick); % (1, N_a, N_j, N_e)

    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d, a2prime)
    linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aBind + N_d*n2long*N_a2*N_a*jBind + N_d*n2long*N_a2*N_a*N_j*eind;
    linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aBind + N_d*n2long*N_a2*N_a*jBind + N_d*n2long*N_a2*N_a*N_j*eind;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    PolicyL2flag  = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper); % shape (1, N_a, N_j, N_e)

    %% Fold a1prime midpoint and L2ind into 'lower grid point + offset' convention
    adjust=(maxindexL2a1<1+n2short+1);
    a1prime_lower=a1prime_midpoint-adjust;
    a1prime_L2  =adjust.*maxindexL2a1+(1-adjust).*(maxindexL2a1-n2short-1);

    %% Pack Policy (5 channels: d, a1prime_lower, a2prime, L2ind, L2flag)
    Policy(1,:,:,:)=maxindexL2d;
    Policy(2,:,:,:)=a1prime_lower;
    Policy(3,:,:,:)=maxindexL2a2;
    Policy(4,:,:,:)=a1prime_L2;
    Policy(5,:,:,:)=PolicyL2flag;
elseif vfoptions.lowmemory==1
    V=zeros(N_a*N_j,N_e,'gpuArray');
    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,1,1,:,e_c,:); % [1,1,1,1,1,N_j,1,l_e]
        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, special_n_e, N_j, d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, e_vals, ReturnFnParamsAgeMatrix, 1);
        entireRHS_e=ReturnMatrix_e+shiftdim(DiscountedEV,-1);
        [~,maxindex]=max(entireRHS_e,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);
        % midpoint is (n_d, 1, n_a2, n_a1, n_a2, N_j)
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        % a1primeindexes shape: (n_d, n2long, n_a2, n_a1, n_a2, N_j)

        ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, special_n_e, N_j, d_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, e_vals, ReturnFnParamsAgeMatrix, 2);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind;
        entireRHS_ii_e=ReturnMatrix_ii_e+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_j]);

        [Vtempii_e,maxindexL2]=max(entireRHS_ii_e,[],1);
        maxindexL2d=rem(maxindexL2-1,N_d)+1;
        maxindexL2a=ceil(maxindexL2/N_d);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        V(:,e_c)=reshape(Vtempii_e,[N_a*N_j,1]);

        midpoint_pick=maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*jBind;
        a1prime_midpoint=midpoint(midpoint_pick); % (1, N_a, N_j)

        linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aBind + N_d*n2long*N_a2*N_a*jBind;
        linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aBind + N_d*n2long*N_a2*N_a*jBind;
        isInfLower    = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag  = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        adjust=(maxindexL2a1<1+n2short+1);
        a1prime_lower=a1prime_midpoint-adjust;
        a1prime_L2  =adjust.*maxindexL2a1+(1-adjust).*(maxindexL2a1-n2short-1);

        Policy(1,:,:,e_c)=maxindexL2d;
        Policy(2,:,:,e_c)=a1prime_lower;
        Policy(3,:,:,e_c)=maxindexL2a2;
        Policy(4,:,:,e_c)=a1prime_L2;
        Policy(5,:,:,e_c)=PolicyL2flag;
    end
end


end
