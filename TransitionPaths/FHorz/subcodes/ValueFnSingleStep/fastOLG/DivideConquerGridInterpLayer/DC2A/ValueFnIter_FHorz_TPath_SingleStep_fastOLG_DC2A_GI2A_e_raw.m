function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_GI2A_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% DC2A_GI2A (has d, z, e): divide-and-conquer in a1 (iterate a2), grid interpolation on a1.
% fastOLG layout: V is (a*j)-by-z-by-e, Policy is (channels, a, j, z, e)
% pi_z_J is (j,z',z) for fastOLG
% pi_e_J is (a*j, z, e) for fastOLG (i.i.d. e, broadcast across a/j/z)
% z_gridvals_J is (j,N_z,l_z) and e_gridvals_J is (j,N_e,l_e) for fastOLG

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
end
if vfoptions.lowmemory>=3
    error('vfoptions.lowmemory>=3 not supported')
end

% Policy: 4 channels (d, a1prime midpoint, a2prime, a1prime L2); L2flag appended at end.
Policy=zeros(4,N_a,N_j,N_z,N_e,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_j,N_z,N_e,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

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
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% Pre-shift z/e grids so the fastOLG DC2A_e return-fn helper can index directly:
% helper expects z at dim 7, e at dim 8, with l-component on dim 9.
z_gridvals_J=reshape(z_gridvals_J,[1,1,1,1,1,N_j,N_z,1,length(n_z)]); % [1,1,1,1,1,N_j,N_z,1,l_z]
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,1,1,N_j,1,N_e,length(n_e)]); % [1,1,1,1,1,N_j,1,N_e,l_e]

% precompute index arrays
% ReturnMatrix_ii Level=1/3 shape: [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_j, N_z, N_e]
% ReturnMatrix_ii Level=2 shape after reshape: [N_d*n2long*N_a2, N_a, N_j, N_z, N_e]
% DiscountedEV shape: [1, N_a1, N_a2, 1, 1, N_j, N_z]; DiscountedEVinterp shape: [1, N_a1fine, N_a2, 1, 1, N_j, N_z]
% (EV does not carry an e dim — e is i.i.d. so the continuation depends only on a',j,z)
a2Bind=shiftdim(gpuArray(0:1:N_a2-1),-1); % [1,1,N_a2] -> a2prime offset in dim 3
a2ind =shiftdim(gpuArray(0:1:N_a2-1),-1); % [1,1,N_a2]
jind  =shiftdim(gpuArray(0:1:N_j-1),-4);  % [1,1,1,1,1,N_j] -> age offset in dim 6
zind_a=shiftdim(gpuArray(0:1:N_z-1),-5);  % [1,1,1,1,1,1,N_z] -> shock offset in dim 7

% Final Policy linear indexers (target shape [1, N_a, N_j, N_z, N_e]):
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1); % [1,N_a]
jBind =shiftdim(gpuArray(0:1:N_j-1),-1); % [1,1,N_j]
zBind =shiftdim(gpuArray(0:1:N_z-1),-2); % [1,1,1,N_z]
eind  =shiftdim(gpuArray(0:1:N_e-1),-3); % [1,1,1,1,N_e]

%% First, create the big 'next period (of transition path) expected value fn.

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

% V is (N_a*N_j, N_z, N_e). First integrate over e' using pi_e_J (i.i.d.), then
% over z' using pi_z_J (Markov). The resulting EV depends on (a',j,z), not on e.
if vfoptions.EVpre==0
    EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % zeros at j=N_j (terminal age has no continuation in TPath)
    EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; % -Inf*0 = NaN, replace with 0 (the 0 comes from transition prob)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]);
elseif vfoptions.EVpre==1
    % 'Matched Expectations Path': input V is already E[V'|.] across z' and e'
    EV=reshape(V,[N_a,1,N_j,N_z]).*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]);
end
% Reshape to (a1,a2,1,1,j,z) for DC2A-style broadcast against ReturnMatrix dims (a1prime,a2prime,a1,a2,j,z)
EV=reshape(EV,[N_a1,N_a2,1,1,N_j,N_z]);
% Interpolate over a1prime_grid (along dim 1)
EVinterp=interp1(a1_grid,EV,a1prime_grid);

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV; % [N_a1, N_a2, 1, 1, N_j, N_z]
DiscountedEV=shiftdim(DiscountedEV,-1); % [1, N_a1, N_a2, 1, 1, N_j, N_z]; will autoexpand d in dim 1
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EVinterp; % [N_a1fine, N_a2, 1, 1, N_j, N_z]
DiscountedEVinterp=shiftdim(DiscountedEVinterp,-1); % [1, N_a1fine, N_a2, 1, 1, N_j, N_z]

if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d,1,N_a2,N_a1,N_a2,N_j,N_z,N_e,'gpuArray');

    % n-Monotonicity (Level 1 coarse)
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, N_j, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % shape: [N_d, N_a1, N_a2, level1n, N_a2, N_j, N_z, N_e]

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % trailing N_e broadcasts in

    % First, we want a1prime conditional on (d,1,a2prime,a1=level1ii,a2,j,z,e)
    [~,maxindex1]=max(entireRHS_ii,[],2);
    % maxindex1 shape: [N_d, 1, N_a2, level1n, N_a2, N_j, N_z, N_e]

    midpoints_jj(:,1,:,level1ii,:,:,:,:)=maxindex1;

    maxgap=squeeze(max(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:),[],8),[],7),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z-by-n_e
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, N_j, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,3);
            % Linear-index DiscountedEV: stride 1 for a1prime (dim 2), N_a1 for a2prime (dim 3), N_a for j (dim 6), N_a*N_j for z (dim 7).
            % EV does not depend on e, but aprimejz does (loweredge inherits e from maxindex1), so the lookup is per-e.
            aprimejz=a1primeindexes+N_a1*a2Bind+N_a*jind+N_a*N_j*zind_a;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimejz(:)),[N_d,(maxgap(ii)+1),N_a2,1,N_a2,N_j,N_z,N_e]);
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,:,curraindex,:,:,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
            midpoints_jj(:,1,:,curraindex,:,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1,1,1);
        end
    end

    midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
    % midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-N_j-by-n_z-by-n_e
    a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    % aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-N_j-by-n_z-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_e(ReturnFn,n_d,n_z,n_e,N_j,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    % ReturnMatrix_ii shape after reshape: [N_d*n2long*N_a2, N_a, N_j, N_z, N_e]
    % EV does not depend on e — reuse same a1prime index, trailing N_e broadcasts in.
    aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind+N_a1fine*N_a2*N_j*zind_a;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_j,N_z,N_e]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    % maxindexL2 shape: [1, N_a, N_j, N_z, N_e]
    V=reshape(Vtempii,[N_a*N_j,N_z,N_e]);
    maxindexL2d=rem(maxindexL2-1,N_d)+1;
    maxindexL2a=ceil(maxindexL2/N_d);
    maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
    maxindexL2a2=ceil(maxindexL2a/n2long);
    Policy(1,:,:,:,:)=maxindexL2d; % d
    % midpoints_jj is [N_d, 1, N_a2, N_a1, N_a2, N_j, N_z, N_e]; strides: d=1, a2prime=N_d, a1=N_d*N_a2, a2=N_d*N_a2*N_a1, j=N_d*N_a2*N_a, z=N_d*N_a2*N_a*N_j, e=N_d*N_a2*N_a*N_j*N_z
    midindex=maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*jBind+N_d*N_a2*N_a*N_j*zBind+N_d*N_a2*N_a*N_j*N_z*eind;
    Policy(2,:,:,:,:)=midpoints_jj(midindex); % a1prime midpoint
    Policy(3,:,:,:,:)=maxindexL2a2; % a2prime
    Policy(4,:,:,:,:)=maxindexL2a1; % a1primeL2ind

    % L2 flag
    linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind + N_d*n2long*N_a2*N_a*N_j*zBind + N_d*n2long*N_a2*N_a*N_j*N_z*eind;
    linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind + N_d*n2long*N_a2*N_a*N_j*zBind + N_d*n2long*N_a2*N_a*N_j*N_z*eind;
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
    PolicyL2flag(1,:,:,:,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
elseif vfoptions.lowmemory==1
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,1,1,:,1,e_c,:);
        midpoints_jj=zeros(N_d,1,N_a2,N_a1,N_a2,N_j,N_z,'gpuArray');

        ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, N_j, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,1);
        % shape: [N_d, N_a1, N_a2, level1n, N_a2, N_j, N_z]

        entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV;
        [~,maxindex1]=max(entireRHS_ii_e,[],2);
        % maxindex1 shape: [N_d, 1, N_a2, level1n, N_a2, N_j, N_z]
        midpoints_jj(:,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, N_j, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,3);
                aprimejz=a1primeindexes+N_a1*a2Bind+N_a*jind+N_a*N_j*zind_a;
                entireRHS_ii_e=ReturnMatrix_ii_e+reshape(DiscountedEV(aprimejz(:)),[N_d,(maxgap(ii)+1),N_a2,1,N_a2,N_j,N_z]);
                [~,maxindex]=max(entireRHS_ii_e,[],2);
                midpoints_jj(:,1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoints_jj(:,1,:,curraindex,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC2A_e(ReturnFn,n_d,n_z,special_n_e,N_j,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind+N_a1fine*N_a2*N_j*zind_a;
        entireRHS_ii_e=ReturnMatrix_ii_e+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_j,N_z]);
        [Vtempii_e,maxindexL2]=max(entireRHS_ii_e,[],1);
        V(:,:,e_c)=reshape(Vtempii_e,[N_a*N_j,N_z]);
        maxindexL2d=rem(maxindexL2-1,N_d)+1;
        maxindexL2a=ceil(maxindexL2/N_d);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);
        Policy(1,:,:,:,e_c)=maxindexL2d; % d
        midindex=maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*jBind+N_d*N_a2*N_a*N_j*zBind;
        Policy(2,:,:,:,e_c)=midpoints_jj(midindex);
        Policy(3,:,:,:,e_c)=maxindexL2a2;
        Policy(4,:,:,:,e_c)=maxindexL2a1;

        linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind + N_d*n2long*N_a2*N_a*N_j*zBind;
        linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind + N_d*n2long*N_a2*N_a*N_j*zBind;
        isInfLower = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,:,e_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    end
elseif vfoptions.lowmemory==2
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,1,1,:,1,e_c,:);
        for z_c=1:N_z
            z_vals=z_gridvals_J(1,1,1,1,1,:,z_c,1,:);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
            midpoints_jj=zeros(N_d,1,N_a2,N_a1,N_a2,N_j,'gpuArray');

            ReturnMatrix_ii_ze=CreateReturnFnMatrix_fastOLG_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, N_j, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix,1);
            % shape: [N_d, N_a1, N_a2, level1n, N_a2, N_j]

            entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_z;
            [~,maxindex1]=max(entireRHS_ii_ze,[],2);
            midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_fastOLG_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, N_j, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix,3);
                    aprimejz=a1primeindexes+N_a1*a2Bind+N_a*jind;
                    entireRHS_ii_ze=ReturnMatrix_ii_ze+reshape(DiscountedEV_z(aprimejz(:)),[N_d,(maxgap(ii)+1),N_a2,1,N_a2,N_j]);
                    [~,maxindex]=max(entireRHS_ii_ze,[],2);
                    midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
                end
            end

            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_ze=CreateReturnFnMatrix_fastOLG_Disc_DC2A_e(ReturnFn,n_d,special_n_z,special_n_e,N_j,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix,2);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind;
            entireRHS_ii_ze=ReturnMatrix_ii_ze+reshape(DiscountedEVinterp_z(aprime),[N_d*n2long*N_a2,N_a,N_j]);
            [Vtempii_ze,maxindexL2]=max(entireRHS_ii_ze,[],1);
            V(:,z_c,e_c)=reshape(Vtempii_ze,[N_a*N_j,1]);
            maxindexL2d=rem(maxindexL2-1,N_d)+1;
            maxindexL2a=ceil(maxindexL2/N_d);
            maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2a/n2long);
            Policy(1,:,:,z_c,e_c)=maxindexL2d;
            midindex=maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*jBind;
            Policy(2,:,:,z_c,e_c)=midpoints_jj(midindex);
            Policy(3,:,:,z_c,e_c)=maxindexL2a2;
            Policy(4,:,:,z_c,e_c)=maxindexL2a1;

            linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind;
            linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind;
            isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,:,z_c,e_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end
end


%% Currently Policy(2,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy;PolicyL2flag];


end
