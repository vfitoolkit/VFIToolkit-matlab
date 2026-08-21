function [V,Policy2]=ValueFnIter_InfHorz_TPath_SingleStep_GI2A_raw(Vnext,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Grid interpolation layer on the first endogenous state, without divide-and-conquer.
% Everything after the first endogenous state is handled together as one Kron'd block, so
% this covers length(n_a)>=2 (not just exactly two).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%%
if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported for ValueFnIter_InfHorz_TPath_SingleStep_GI2A_raw')
end

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=prod(n_a2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of a1prime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% precompute
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1
zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
zBind=shiftdim(gpuArray(0:1:N_z-1),-4); % already includes -1
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);

Policy=zeros(5,N_a,N_z,'gpuArray'); % first dim: d,a1prime midpoint,a2prime,a1prime L2,L2flag (pilot)
% When ReturnFn is -Inf on one of the coarse grid points, we allow the fine index between that
% and the neighbouring coarse grid point, but record it in L2flag so that the simulation and the
% agent dist iteration can later avoid putting weight on that -Inf point.

%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over z', leaving a singular second dimension

EV=reshape(EV,[N_a1,N_a2,1,1,N_z]);
% Interpolate EV over a1prime_grid (only the first endogenous state is interpolated)
EVinterp=interp1(a1_grid,EV,a1prime_grid);

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn,n_d,n_z,d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParamsVec,1,0);
    entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);

    % Calc the max and it's index: a1prime(d,1,a2prime,a1,a2,z)
    [~,maxindex]=max(entireRHS,[],2);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-n_z
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % a1prime points either side of midpoint
    % a1prime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn,n_d,n_z,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParamsVec,2,0);
    aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d*n2long*N_a2,N_a,N_z]);
    [V,maxindexL2]=max(entireRHS_ii,[],1);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);

    V=shiftdim(V,1);
    Policy(1,:,:)=maxindexL2d; % d
    Policy(2,:,:)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*zind); % a1prime midpoint
    Policy(3,:,:)=maxindexL2a2; % a2prime
    Policy(4,:,:)=maxindexL2a1; % a1primeL2ind
    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d, a2prime)
    linidx_lower  = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
    linidx_upper  = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*zind;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    Policy(5,:,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

elseif vfoptions.lowmemory==1
    V=zeros(N_a,N_z,'gpuArray'); % preallocate

    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        EV_z=EV(:,:,:,:,z_c);
        EVinterp_z=EVinterp(:,:,:,:,z_c);

        ReturnMatrix_z=CreateReturnFnMatrix_Disc_DC2A(ReturnFn,n_d,special_n_z,d_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec,1,0);
        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*shiftdim(EV_z,-1);

        [~,maxindex]=max(entireRHS_z,[],2);

        midpoint=max(min(maxindex,n_a1-1),2);
        % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        % a1prime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2
        ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A(ReturnFn,n_d,special_n_z,d_gridvals,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec,2,0);
        aprime=a1primeindexes+N_a1fine*a2ind;
        entireRHS_ii_z=ReturnMatrix_ii_z+DiscountFactorParamsVec*reshape(EVinterp_z(aprime),[N_d*n2long*N_a2,N_a]);
        [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
        maxindexL2d=rem(maxindexL2-1,N_d)+1;
        maxindexL2a=ceil(maxindexL2/N_d);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        V(:,z_c)=shiftdim(Vtempii,1);
        Policy(1,:,z_c)=maxindexL2d; % d
        Policy(2,:,z_c)=midpoint(maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind); % a1prime midpoint
        Policy(3,:,z_c)=maxindexL2a2; % a2prime
        Policy(4,:,z_c)=maxindexL2a1; % a1primeL2ind
        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_lower  = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
        linidx_upper  = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind;
        isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        Policy(5,:,z_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    end
end

%% Currently Policy(2,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(4,:,:)=adjust.*Policy(4,:,:)+(1-adjust).*(Policy(4,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%% Policy in transition paths
l_d=length(n_d);
Policy2=zeros(l_d+4,N_a,N_z); % +4 = a1 midpoint, a2prime, a1prime L2index, L2flag (pilot)
% sort d variables
Policy2(1,:,:)=rem(Policy(1,:,:)-1,n_d(1))+1;
if l_d>1
    if l_d>2
        for ii=2:l_d-1
            Policy2(ii,:,:)=rem(ceil(Policy(1,:,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
        end
    end
    Policy2(l_d,:,:)=ceil(Policy(1,:,:)/prod(n_d(1:l_d-1)));
end
% rest are already in right shape
Policy2(l_d+1:end,:,:)=Policy(2:5,:,:);

end
