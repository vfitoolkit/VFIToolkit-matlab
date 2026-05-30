function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_GI2A_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG
% DC2A_GI2A (no d): divide-and-conquer in a1 (iterate a2), grid interpolation on a1

N_a=prod(n_a);
N_z=prod(n_z);

% fastOLG, so a-j-z
Policy=zeros(3,N_a,N_j,N_z,'gpuArray'); % first dim is (a1prime midpoint, a2prime, a1prime L2)
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

% Grid interpolation (GI on a1)
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% Pre-shift z_gridvals_J to the shape expected by the DC2A_nod fastOLG helper:
% [1,1,1,1,N_j,N_z,l_z] (age in dim 5, z in dim 6)
z_gridvals_J=shiftdim(z_gridvals_J,-4);

% precompute index arrays
% ReturnMatrix_ii Level=1 shape: [N_a1prime, N_a2prime, N_a1, N_a2, N_j, N_z] (a2prime in dim 2, j in dim 5, z in dim 6)
% ReturnMatrix_ii Level=2 shape after reshape: [N_a1prime*N_a2prime, N_a, N_j, N_z]
% DiscountedEV shape: [N_a1, N_a2, 1, 1, N_j, N_z]; DiscountedEVinterp shape: [N_a1fine, N_a2, 1, 1, N_j, N_z]
a2Bind=gpuArray(0:1:N_a2-1); % [1,N_a2] -> a2prime offset in dim 2 for linear-index into DiscountedEV
a2ind =gpuArray(0:1:N_a2-1); % [1,N_a2] -> a2prime offset in dim 2 for linear-index into DiscountedEVinterp
jind  =shiftdim(gpuArray(0:1:N_j-1),-3); % [1,1,1,1,N_j] -> age offset in dim 5
zind_a=shiftdim(gpuArray(0:1:N_z-1),-4); % [1,1,1,1,1,N_z] -> shock offset in dim 6

% Final Policy linear indexers (target shape [1, N_a, N_j, N_z]):
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1); % [1,N_a]
jBind =shiftdim(gpuArray(0:1:N_j-1),-1); % [1,1,N_j]
zBind =shiftdim(gpuArray(0:1:N_z-1),-2); % [1,1,1,N_z]

if vfoptions.lowmemory==0
    midpoints_jj=zeros(1,N_a2,N_a1,N_a2,N_j,N_z,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(1,N_a2,N_a1,N_a2,N_j,'gpuArray');
    special_n_z=ones(1,length(n_z));
end

%% First, create the big 'next period (of transition path) expected value fn.

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.EVpre==0
    EVpre=zeros(N_a,1,N_j,N_z,'gpuArray');
    EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]);
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]);
elseif vfoptions.EVpre==1
    EV=reshape(V,[N_a,1,N_j,N_z]).*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]);
end
% Reshape to (a1,a2,1,1,j,z) for DC2A-style broadcast against ReturnMatrix dims (a1prime,a2prime,a1,a2,j,z)
EV=reshape(EV,[N_a1,N_a2,1,1,N_j,N_z]);
% Interpolate over a1prime_grid (along dim 1)
EVinterp=interp1(a1_grid,EV,a1prime_grid);

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV; % [N_a1, N_a2, 1, 1, N_j, N_z]
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EVinterp; % [N_a1fine, N_a2, 1, 1, N_j, N_z]

if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, n_z, N_j, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % shape: [N_a1, N_a2, level1n, N_a2, N_j, N_z]

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % broadcasts a (dims 3-4) for the continuation

    % First, we want a1prime conditional on (1,a2prime,a1=level1ii,a2,j,z)
    [~,maxindex1]=max(entireRHS_ii,[],1);
    % maxindex1 shape: [1, N_a2, level1n, N_a2, N_j, N_z]

    midpoints_jj(1,:,level1ii,:,:,:)=maxindex1;

    maxgap=squeeze(max(max(max(max(maxindex1(1,:,2:end,:,:,:)-maxindex1(1,:,1:end-1,:,:,:),[],6),[],5),[],4),[],2));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,:,ii,:,:,:),N_a1-maxgap(ii));
            % loweredge is 1-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
            a1primeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are (maxgap(ii)+1)-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, n_z, N_j, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,1);
            % Linear-index DiscountedEV: stride 1 for a1prime (dim 1), N_a1 for a2prime (dim 2), N_a for j (dim 5), N_a*N_j for z (dim 6)
            aprimejz=a1primeindexes+N_a1*a2Bind+N_a*jind+N_a*N_j*zind_a;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimejz(:)),[(maxgap(ii)+1),N_a2,1,N_a2,N_j,N_z]); % autoexpand level1iidiff(ii) in 3rd-dim
            [~,maxindex]=max(entireRHS_ii,[],1);
            midpoints_jj(1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(1,:,ii,:,:,:);
            midpoints_jj(1,:,curraindex,:,:,:)=repelem(loweredge,1,1,length(curraindex),1,1,1);
        end
    end

    midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
    % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-N_j-by-n_z
    a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-N_j-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn,n_z,N_j,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);
    % shape after reshape: [n2long*N_a2, N_a, N_j, N_z]
    aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind+N_a1fine*N_a2*N_j*zind_a;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a,N_j,N_z]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V=reshape(Vtempii,[N_a*N_j,N_z]);
    maxindexL2a1=rem(maxindexL2-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2/n2long);
    % midpoints_jj is [1, N_a2, N_a1, N_a2, N_j, N_z]; strides: a2prime=1, a1=N_a2, a2=N_a2*N_a1, j=N_a2*N_a, z=N_a2*N_a*N_j
    midindex=maxindexL2a2+N_a2*a12ind+N_a2*N_a*jBind+N_a2*N_a*N_j*zBind;
    Policy(1,:,:,:)=midpoints_jj(midindex); % a1prime midpoint
    Policy(2,:,:,:)=maxindexL2a2; % a2prime
    Policy(3,:,:,:)=maxindexL2a1; % a1primeL2ind

    % L2 flag
    linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*jBind + n2long*N_a2*N_a*N_j*zBind;
    linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*jBind + n2long*N_a2*N_a*N_j*zBind;
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    PolicyL2flag(1,:,:,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

elseif vfoptions.lowmemory==1

    V=zeros(N_a*N_j,N_z,'gpuArray');

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,1,N_j,N_z,l_z)
        DiscountedEV_z=DiscountedEV(:,:,:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,z_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_z, N_j, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, ReturnFnParamsAgeMatrix,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

        [~,maxindex1]=max(entireRHS_ii,[],1);
        midpoints_jj(1,:,level1ii,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_z, N_j, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsAgeMatrix,1);
                aprimej=a1primeindexes+N_a1*a2Bind+N_a*jind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_z(aprimej(:)),[(maxgap(ii)+1),N_a2,1,N_a2,N_j]);
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                midpoints_jj(1,:,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_z, N_j,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid, z_vals, ReturnFnParamsAgeMatrix,2);
        aprimej=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(aprimej(:)),[n2long*N_a2,N_a,N_j]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,z_c)=reshape(Vtempii,[N_a*N_j,1]);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);
        midindex=maxindexL2a2+N_a2*a12ind+N_a2*N_a*jBind;
        Policy(1,:,:,z_c)=midpoints_jj(midindex);
        Policy(2,:,:,z_c)=maxindexL2a2;
        Policy(3,:,:,z_c)=maxindexL2a1;

        % L2 flag
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*jBind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*jBind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,z_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    end
end


%% Currently Policy(1,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy;PolicyL2flag];


end
