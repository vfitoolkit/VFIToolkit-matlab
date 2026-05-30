function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_GI2A_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG
% DC2A_GI2A: divide-and-conquer in a1 (iterate a2), grid interpolation on a1

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% fastOLG, so a-j-z
Policy=zeros(4,N_a,N_j,N_z,'gpuArray'); % first dim is (d, a1prime midpoint, a2prime, a1prime L2)
PolicyL2flag=2*ones(1,N_a,N_j,N_z,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%% Split endogenous state into a1 (DC) and a2 (iterate)
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity (DC on a1 only)
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation (GI on a1)
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% Pre-shift z_gridvals_J to the shape expected by the DC2A fastOLG helper:
% [1,1,1,1,1,N_j,N_z,l_z] (age in dim 6, z in dim 7)
z_gridvals_J=shiftdim(z_gridvals_J,-5);

% precompute index arrays
% ReturnMatrix_ii Level=1/3 shape: [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_j, N_z] (a2prime in dim 3, j in dim 6, z in dim 7)
% ReturnMatrix_ii Level=2 shape after reshape: [N_d*n2long*N_a2, N_a, N_j, N_z]
% DiscountedEV shape: [1, N_a1, N_a2, 1, 1, N_j, N_z]; DiscountedEVinterp shape: [1, N_a1fine, N_a2, 1, 1, N_j, N_z]
a2Bind=shiftdim(gpuArray(0:1:N_a2-1),-1); % [1,1,N_a2] -> a2prime offset in dim 3 for linear-index into DiscountedEV
a2ind =shiftdim(gpuArray(0:1:N_a2-1),-1); % [1,1,N_a2] -> a2prime offset in dim 3 for linear-index into DiscountedEVinterp
jind  =shiftdim(gpuArray(0:1:N_j-1),-4);  % [1,1,1,1,1,N_j] -> age offset in dim 6
zind_a=shiftdim(gpuArray(0:1:N_z-1),-5);  % [1,1,1,1,1,1,N_z] -> shock offset in dim 7

% Final Policy linear indexers (target shape [1, N_a, N_j, N_z]):
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1); % size [1,N_a]; provides a1+N_a1*a2
jBind =shiftdim(gpuArray(0:1:N_j-1),-1); % [1,1,N_j]
zBind =shiftdim(gpuArray(0:1:N_z-1),-2); % [1,1,1,N_z]

if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d,1,N_a2,N_a1,N_a2,N_j,N_z,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(N_d,1,N_a2,N_a1,N_a2,N_j,'gpuArray');
    special_n_z=ones(1,length(n_z));
end

%% First, create the big 'next period (of transition path) expected value fn.

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.EVpre==0
    % V input is (N_a*N_j, N_z); shift one age to the left (next-period continuation)
    EVpre=zeros(N_a,1,N_j,N_z,'gpuArray');
    EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % zeros at j=N_j so pi_z_J builds the expectation
    EV=EVpre.*shiftdim(pi_z_J,-2); % [N_a,1,N_j,N_z(zprime)] .* [1,1,N_j,N_z(zprime),N_z(z)]
    EV(isnan(EV))=0; % multiplications of -Inf with 0 give NaN; transition probabilities zero them out
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (a,1,j,z)
elseif vfoptions.EVpre==1
    % 'Matched Expectations Path': input V is already of size [N_a,N_j,N_z]
    EV=reshape(V,[N_a,1,N_j,N_z]).*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]);
end
% Reshape to (a1,a2,1,1,j,z) for DC2A-style broadcast against ReturnMatrix dims (a1prime,a2prime,a1,a2,j,z)
EV=reshape(EV,[N_a1,N_a2,1,1,N_j,N_z]);
% Interpolate over a1prime_grid (along dim 1)
EVinterp=interp1(a1_grid,EV,a1prime_grid);

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV; % age broadcasts across dim 5 (singleton in EV)
DiscountedEV=shiftdim(DiscountedEV,-1); % [1, N_a1, N_a2, 1, 1, N_j, N_z]; will autoexpand d in dim 1
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EVinterp; % [N_a1fine, N_a2, 1, 1, N_j, N_z]
DiscountedEVinterp=shiftdim(DiscountedEVinterp,-1); % [1, N_a1fine, N_a2, 1, 1, N_j, N_z]

if vfoptions.lowmemory==0

    % n-Monotonicity (DC on a1 only)
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, n_z, N_j, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % shape: [N_d, N_a1, N_a2, level1n, N_a2, N_j, N_z]

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % broadcasts d (dim 1) and a (dims 4-5) for the continuation

    % First, we want a1prime conditional on (d,1,a2prime,a1=level1ii,a2,j,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);
    % maxindex1 shape: [N_d, 1, N_a2, level1n, N_a2, N_j, N_z]

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(:,1,:,level1ii,:,:,:)=maxindex1;

    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, n_z, N_j, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,3);
            % Linear-index DiscountedEV: stride 1 for a1prime (dim 2), N_a1 for a2prime (dim 3), N_a for j (dim 6), N_a*N_j for z (dim 7)
            aprimejz=a1primeindexes+N_a1*a2Bind+N_a*jind+N_a*N_j*zind_a;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimejz(:)),[N_d,(maxgap(ii)+1),N_a2,1,N_a2,N_j,N_z]); % autoexpand level1iidiff(ii) in 4th-dim
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:,:);
            midpoints_jj(:,1,:,curraindex,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1,1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-N_j-by-n_z
    a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-N_j-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn,n_d,n_z,N_j,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);
    % ReturnMatrix_ii shape after reshape: [N_d*n2long*N_a2, N_a, N_j, N_z]
    aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind+N_a1fine*N_a2*N_j*zind_a;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_j,N_z]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    % maxindexL2 shape: [1, N_a, N_j, N_z]
    V=reshape(Vtempii,[N_a*N_j,N_z]);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);
    Policy(1,:,:,:)=maxindexL2d; % d
    % midpoints_jj is [N_d, 1, N_a2, N_a1, N_a2, N_j, N_z]; strides: d=1, a2prime=N_d, a1=N_d*N_a2, a2=N_d*N_a2*N_a1, j=N_d*N_a2*N_a, z=N_d*N_a2*N_a*N_j
    midindex=maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*jBind+N_d*N_a2*N_a*N_j*zBind;
    Policy(2,:,:,:)=midpoints_jj(midindex); % a1prime midpoint
    Policy(3,:,:,:)=maxindexL2a2; % a2prime
    Policy(4,:,:,:)=maxindexL2a1; % a1primeL2ind

    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind + N_d*n2long*N_a2*N_a*N_j*zBind;
    linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind + N_d*n2long*N_a2*N_a*N_j*zBind;
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    PolicyL2flag(1,:,:,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

elseif vfoptions.lowmemory==1

    V=zeros(N_a*N_j,N_z,'gpuArray'); % V is over (a,j,z)

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,1,1,N_j,N_z,l_z)
        DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, ReturnFnParamsAgeMatrix,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsAgeMatrix,3);
                aprimej=a1primeindexes+N_a1*a2Bind+N_a*jind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_z(aprimej(:)),[N_d,(maxgap(ii)+1),N_a2,1,N_a2,N_j]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:);
                midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn,n_d,special_n_z,N_j,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_vals, ReturnFnParamsAgeMatrix,2);
        aprimej=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(aprimej(:)),[N_d*n2long*N_a2,N_a,N_j]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,z_c)=reshape(Vtempii,[N_a*N_j,1]);
        maxindexL2d=rem(maxindexL2-1,N_d)+1;
        maxindexL2a=ceil(maxindexL2/N_d);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);
        Policy(1,:,:,z_c)=maxindexL2d;
        midindex=maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*jBind;
        Policy(2,:,:,z_c)=midpoints_jj(midindex);
        Policy(3,:,:,z_c)=maxindexL2a2;
        Policy(4,:,:,z_c)=maxindexL2a1;

        % L2 flag
        linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind;
        linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,z_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    end
end


%% Currently Policy(2,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy;PolicyL2flag];


end
