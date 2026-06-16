function [V,Policy,Policyalt,Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_GI1_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z,e), rather than standard (a,z,e,j)
% V is (a,j)-by-z-by-e (carries Valt for Naive QH)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% fastOLG, so a-j-z-e
Policy=zeros(4,N_a,N_j,N_z,N_e,'gpuArray'); % first dim indexes the optimal choice for d and aprime (d, midpoint, L2, L2 flag)
Policyalt=zeros(4,N_a,N_j,N_z,N_e,'gpuArray'); % exponential discounter optimal (d, midpoint, L2, L2 flag)
Vtilde=zeros(N_a*N_j,N_z,N_e,'gpuArray');

z_gridvals_J=shiftdim(z_gridvals_J,-3);
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]);

%% Grid interpolation

% Preallocate
if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d,1,N_a,N_j,N_z,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(N_d,1,N_a,N_j,N_z,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoints_jj=zeros(N_d,1,N_a,N_j,'gpuArray');
end

% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);
aBind=gpuArray(0:1:N_a-1);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
zBind=shiftdim(gpuArray(0:1:N_z-1),-2);
eBind=shiftdim(gpuArray(0:1:N_e-1),-3);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z*N_e (note: N_aprime is just equal to N_a)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
beta0beta_J=beta0_J.*beta_J;

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(1:end-N_a,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
    EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EVpre=sum(reshape(V,[N_a*N_j,N_z,N_e]).*pi_e_J,3);  % input V is already of size [N_a,N_j] and we want to use the whole thing
    EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
end

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV_alt=reshape(beta_J,[1,1,N_j]).*EV;
DiscountedEV_alt=repelem(shiftdim(DiscountedEV_alt,-1),N_d,1,1,1); % [d,aprime,1,j,z]
DiscountedEVinterp_alt=reshape(beta_J,[1,1,N_j]).*EVinterp; % [n2aprime fine,1,j,z]

DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;
DiscountedEV=repelem(shiftdim(DiscountedEV,-1),N_d,1,1,1); % [d,aprime,1,j,z]
DiscountedEVinterp=reshape(beta0beta_J,[1,1,N_j]).*EVinterp; % [n2aprime fine,1,j,z]

if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z,n_e,N_j, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,1);

    %% Valt (beta) -> V output
    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_alt;

    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoints_jj(:,1,level1ii,:,:,:)=maxindex1;

    maxgap=max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:,:),n_a-maxgap(:,1,ii,:,:,:));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z,n_e,N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,3);
            daprimejz=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*N_a*jind+N_d*N_a*N_j*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_alt(daprimejz(:)),[N_d,(maxgap(ii)+1),1,N_j,N_z,N_e]);
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:,:,:);
            midpoints_jj(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end

    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d, n_z,n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind;
    entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt(aprimejz(:)),[N_d*n2long,N_a,N_j,N_z,N_e]);
    [V,maxindexL2alt]=max(entireRHS_L2,[],1);
    V=reshape(V,[N_a*N_j,N_z,N_e]);
    d_indalt=rem(maxindexL2alt-1,N_d)+1;
    allindalt=d_indalt+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind+N_d*N_a*N_j*N_z*eBind;
    Policyalt(1,:,:,:,:)=d_indalt;
    Policyalt(2,:,:,:,:)=shiftdim(squeeze(midpoints_jj(allindalt)),-1);
    Policyalt(3,:,:,:,:)=shiftdim(ceil(maxindexL2alt/N_d),-1);

    L2offsetalt=ceil(maxindexL2alt/N_d);
    linidx_loweralt=d_indalt                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind+N_d*n2long*N_a*N_j*N_z*eBind;
    linidx_upperalt=d_indalt+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind+N_d*n2long*N_a*N_j*N_z*eBind;
    isInfLoweralt=(ReturnMatrix_L2(linidx_loweralt)==-Inf);
    isInfUpperalt=(ReturnMatrix_L2(linidx_upperalt)==-Inf);
    inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
    inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
    Policyalt(4,:,:,:,:)=shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

    %% Policy (beta0*beta)
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z,n_e,N_j, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,1); % recompute at level-1 anchors
    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoints_jj(:,1,level1ii,:,:,:)=maxindex1;

    maxgap=max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:,:),n_a-maxgap(:,1,ii,:,:,:));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z,n_e,N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,3);
            daprimejz=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*N_a*jind+N_d*N_a*N_j*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(daprimejz(:)),[N_d,(maxgap(ii)+1),1,N_j,N_z,N_e]);
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:,:,:);
            midpoints_jj(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end

    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d, n_z,n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind;
    entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp(aprimejz(:)),[N_d*n2long,N_a,N_j,N_z,N_e]);
    [Vtilde,maxindexL2]=max(entireRHS_L2,[],1);
    Vtilde=reshape(Vtilde,[N_a*N_j,N_z,N_e]);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind+N_d*N_a*N_j*N_z*eBind;
    Policy(1,:,:,:,:)=d_ind;
    Policy(2,:,:,:,:)=shiftdim(squeeze(midpoints_jj(allind)),-1);
    Policy(3,:,:,:,:)=shiftdim(ceil(maxindexL2/N_d),-1);

    L2offset=ceil(maxindexL2/N_d);
    linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind+N_d*n2long*N_a*N_j*N_z*eBind;
    linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind+N_d*n2long*N_a*N_j*N_z*eBind;
    isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
    inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
    inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
    Policy(4,:,:,:,:)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z,special_n_e,N_j, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J,e_vals, ReturnFnParamsAgeMatrix,1);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_alt;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:,:)=maxindex1;

        maxgap=max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(:,1,ii,:,:));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z,special_n_e,N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J,e_vals, ReturnFnParamsAgeMatrix,3);
                daprimejz=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*N_a*jind+N_d*N_a*N_j*zind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_alt(daprimejz(:)),[N_d,(maxgap(ii)+1),1,N_j,N_z]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d, n_z,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_gridvals_J,e_vals, ReturnFnParamsAgeMatrix,2);
        aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind;
        entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt(aprimejz(:)),[N_d*n2long,N_a,N_j,N_z]);
        [Vtemp,maxindexL2alt]=max(entireRHS_L2,[],1);
        V(:,:,e_c)=reshape(Vtemp,[N_a*N_j,N_z]);
        d_indalt=rem(maxindexL2alt-1,N_d)+1;
        allindalt=d_indalt+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind;
        Policyalt(1,:,:,:,e_c)=d_indalt;
        Policyalt(2,:,:,:,e_c)=shiftdim(squeeze(midpoints_jj(allindalt)),-1);
        Policyalt(3,:,:,:,e_c)=shiftdim(ceil(maxindexL2alt/N_d),-1);

        L2offsetalt=ceil(maxindexL2alt/N_d);
        linidx_loweralt=d_indalt                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
        linidx_upperalt=d_indalt+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
        isInfLoweralt=(ReturnMatrix_L2(linidx_loweralt)==-Inf);
        isInfUpperalt=(ReturnMatrix_L2(linidx_upperalt)==-Inf);
        inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
        inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
        Policyalt(4,:,:,:,e_c)=shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

        %% Policy (beta0*beta)
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z,special_n_e,N_j, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J,e_vals, ReturnFnParamsAgeMatrix,1); % recompute at level-1 anchors
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:,:)=maxindex1;

        maxgap=max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(:,1,ii,:,:));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z,special_n_e,N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J,e_vals, ReturnFnParamsAgeMatrix,3);
                daprimejz=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*N_a*jind+N_d*N_a*N_j*zind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(daprimejz(:)),[N_d,(maxgap(ii)+1),1,N_j,N_z]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d, n_z,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_gridvals_J,e_vals, ReturnFnParamsAgeMatrix,2);
        aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind;
        entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp(aprimejz(:)),[N_d*n2long,N_a,N_j,N_z]);
        [Vtilde_e,maxindexL2]=max(entireRHS_L2,[],1);
        Vtilde(:,:,e_c)=reshape(Vtilde_e,[N_a*N_j,N_z]);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind;
        Policy(1,:,:,:,e_c)=d_ind;
        Policy(2,:,:,:,e_c)=shiftdim(squeeze(midpoints_jj(allind)),-1);
        Policy(3,:,:,:,e_c)=shiftdim(ceil(maxindexL2/N_d),-1);

        L2offset=ceil(maxindexL2/N_d);
        linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
        linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
        isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
        inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
        inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
        Policy(4,:,:,:,e_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
    end

elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:);
        DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,z_c);
        DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,z_c);
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,z_c);

        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,1,:,1,e_c,:);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, special_n_z,special_n_e,N_j, d_gridvals, a_grid, a_grid(level1ii), z_vals,e_vals, ReturnFnParamsAgeMatrix,1);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_alt_z;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,level1ii,:)=maxindex1;

            maxgap=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(:,1,ii,:));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, special_n_z,special_n_e,N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals,e_vals, ReturnFnParamsAgeMatrix,3);
                    daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*N_a*jind;
                    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_alt_z(daprimej(:)),[N_d,(maxgap(ii)+1),1,N_j]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d, special_n_z,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_vals,e_vals, ReturnFnParamsAgeMatrix,2);
            aprimej=aprimeindexes+n2aprime*jind;
            entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt_z(aprimej(:)),[N_d*n2long,N_a,N_j]);
            [Vtemp,maxindexL2alt]=max(entireRHS_L2,[],1);
            V(:,z_c,e_c)=reshape(Vtemp,[N_a*N_j,1]);
            d_indalt=rem(maxindexL2alt-1,N_d)+1;
            allindalt=d_indalt+N_d*aBind+N_d*N_a*jBind;
            Policyalt(1,:,:,z_c,e_c)=d_indalt;
            Policyalt(2,:,:,z_c,e_c)=shiftdim(squeeze(midpoints_jj(allindalt)),-1);
            Policyalt(3,:,:,z_c,e_c)=shiftdim(ceil(maxindexL2alt/N_d),-1);

            L2offsetalt=ceil(maxindexL2alt/N_d);
            linidx_loweralt=d_indalt                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
            linidx_upperalt=d_indalt+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
            isInfLoweralt=(ReturnMatrix_L2(linidx_loweralt)==-Inf);
            isInfUpperalt=(ReturnMatrix_L2(linidx_upperalt)==-Inf);
            inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
            inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
            Policyalt(4,:,:,z_c,e_c)=shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

            %% Policy (beta0*beta)
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, special_n_z,special_n_e,N_j, d_gridvals, a_grid, a_grid(level1ii), z_vals,e_vals, ReturnFnParamsAgeMatrix,1); % recompute at level-1 anchors
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,level1ii,:)=maxindex1;

            maxgap=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(:,1,ii,:));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, special_n_z,special_n_e,N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals,e_vals, ReturnFnParamsAgeMatrix,3);
                    daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*N_a*jind;
                    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_z(daprimej(:)),[N_d,(maxgap(ii)+1),1,N_j]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d, special_n_z,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_vals,e_vals, ReturnFnParamsAgeMatrix,2);
            aprimej=aprimeindexes+n2aprime*jind;
            entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_z(aprimej(:)),[N_d*n2long,N_a,N_j]);
            [Vtilde_ze,maxindexL2]=max(entireRHS_L2,[],1);
            Vtilde(:,z_c,e_c)=reshape(Vtilde_ze,[N_a*N_j,1]);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aBind+N_d*N_a*jBind;
            Policy(1,:,:,z_c,e_c)=d_ind;
            Policy(2,:,:,z_c,e_c)=shiftdim(squeeze(midpoints_jj(allind)),-1);
            Policy(3,:,:,z_c,e_c)=shiftdim(ceil(maxindexL2/N_d),-1);

            L2offset=ceil(maxindexL2/N_d);
            linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
            linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
            isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
            Policy(4,:,:,z_c,e_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
        end
    end
end



%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1);
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust;
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1);

adjustalt=(Policyalt(3,:,:,:,:)<1+n2short+1);
Policyalt(2,:,:,:,:)=Policyalt(2,:,:,:,:)-adjustalt;
Policyalt(3,:,:,:,:)=adjustalt.*Policyalt(3,:,:,:,:)+(1-adjustalt).*(Policyalt(3,:,:,:,:)-n2short-1);

end
