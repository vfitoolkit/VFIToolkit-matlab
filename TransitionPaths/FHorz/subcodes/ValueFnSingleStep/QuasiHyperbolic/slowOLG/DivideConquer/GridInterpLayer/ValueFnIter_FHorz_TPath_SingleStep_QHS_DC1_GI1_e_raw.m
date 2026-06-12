function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_GI1_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Sophisticated QH: V carries Vunderbar; Policy = QH choice; Vhat is the agent's-perspective (beta0*beta) value.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); % [d_ind; midpoint; aprimeL2ind; L2flag]
Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');

% e is start-of-period: precompute the expectation of V over e for use as continuation
Vnext=sum(V.*shiftdim(pi_e_J(:,[1,1:end-1]),-2),3); % Take expectations over e: Vnext(...,jj+1) is read for current age jj, so weight V at age jj+1 by pi_e_J(:,jj) [same timing as standard ValueFnIter commands]; first column is padding, never read

%%
if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
end
if vfoptions.lowmemory>=3
    error('vfoptions.lowmemory=K not supported for ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_GI1_e_raw')
end

if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d,1,N_a,N_z,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(N_d,1,N_a,N_z,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoints_jj=zeros(N_d,1,N_a,1,'gpuArray');
end

aind=gpuArray(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-1);
eind=shiftdim(gpuArray(0:1:N_e-1),-2);
zBind=shiftdim(gpuArray(0:1:N_z-1),-2);

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);


%% j=N_j: terminal age has no continuation in TPath
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    midpoints_jj(:,1,level1ii,:,:)=maxindex1;

    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,6);
            [~,maxindex]=max(ReturnMatrix_ii,[],2);
            midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:,:);
            midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end

    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,:,:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind+N_d*N_a*zind+N_d*N_a*N_z*eind;
    Policy(1,:,:,:,N_j)=d_ind;
    Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
    Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
    L2offset = ceil(maxindexL2/N_d);
    linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*zind + N_d*n2long*N_a*N_z*eind;
    linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zind + N_d*n2long*N_a*N_z*eind;
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    Policy(4,:,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoints_jj(:,1,level1ii,:)=maxindex1;

        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,6);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind;
        Policy(1,:,:,e_c,N_j)=d_ind;
        Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
        Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
        L2offset = ceil(maxindexL2/N_d);
        linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*zind;
        linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(4,:,:,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
    end
elseif vfoptions.lowmemory==2
    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);

            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoints_jj(:,1,level1ii)=maxindex1;

            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,6);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind;
            Policy(1,:,z_c,e_c,N_j)=d_ind;
            Policy(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
            Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
            L2offset = ceil(maxindexL2/N_d);
            linidx_lower = d_ind                  + N_d*n2long*aind;
            linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(4,:,z_c,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
        end
    end
end
Vhat(:,:,:,N_j)=V(:,:,:,N_j); % terminal: Vhat coincides with V (Vunderbar)


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    VKronNext_j=Vnext(:,:,1,jj+1);

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        %% Vhat (beta0*beta) — find QH-optimal Policy
        entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,6);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez(:)),[N_d,(maxgap(ii)+1),1,N_z,N_e]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zBind;
        EVfine=reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z,N_e]);
        entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        Vhat_jj=shiftdim(Vtempii,1);
        Vhat(:,:,:,jj)=Vhat_jj;
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind+N_d*N_a*N_z*eind;
        Policy(1,:,:,:,jj)=d_ind;
        Policy(2,:,:,:,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1);
        Policy(3,:,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1);
        L2offset=ceil(maxindexL2/N_d);
        linidx_lower=d_ind                + N_d*n2long*aind + N_d*n2long*N_a*zind + N_d*n2long*N_a*N_z*eind;
        linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zind + N_d*n2long*N_a*N_z*eind;
        isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
        inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
        inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
        Policy(4,:,:,:,jj)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

        linidx=reshape(maxindexL2,[1,N_a*N_z*N_e])+N_d*n2long*(0:N_a*N_z*N_e-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_z,N_e]);
        V(:,:,:,jj)=Vhat_jj+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);

            %% Vhat (beta0*beta) — find QH-optimal Policy
            entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,level1ii,:)=maxindex1;
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,6);
                    aprimez=aprimeindexes+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez(:)),[N_d,(maxgap(ii)+1),1,N_z]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),e_val,ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*zBind;
            EVfine=reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z]);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            Vhat_jj_e=shiftdim(Vtempii,1);
            Vhat(:,:,e_c,jj)=Vhat_jj_e;
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zind;
            Policy(1,:,:,e_c,jj)=d_ind;
            Policy(2,:,:,e_c,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1);
            Policy(3,:,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1);
            L2offset=ceil(maxindexL2/N_d);
            linidx_lower=d_ind                + N_d*n2long*aind + N_d*n2long*N_a*zind;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zind;
            isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
            Policy(4,:,:,e_c,jj)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

            linidx=reshape(maxindexL2,[1,N_a*N_z])+N_d*n2long*(0:N_a*N_z-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_z]);
            V(:,:,e_c,jj)=Vhat_jj_e+(beta-beta0beta)*EV_at_policy;
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);

                %% Vhat (beta0*beta) — find QH-optimal Policy
                entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV_z,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,level1ii)=maxindex1;
                maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,6);
                        entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV_z(aprimeindexes(:)),[N_d,(maxgap(ii)+1),1]);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
                    end
                end
                midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                EVfine_ze=reshape(EVinterp_z(aprimeindexes(:)),[N_d*n2long,N_a]);
                entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine_ze;
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                Vhat_jj_ze=shiftdim(Vtempii,1);
                Vhat(:,z_c,e_c,jj)=Vhat_jj_ze;
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind;
                Policy(1,:,z_c,e_c,jj)=d_ind;
                Policy(2,:,z_c,e_c,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1);
                Policy(3,:,z_c,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1);
                L2offset=ceil(maxindexL2/N_d);
                linidx_lower=d_ind                + N_d*n2long*aind;
                linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*aind;
                isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
                Policy(4,:,z_c,e_c,jj)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

                linidx_ze=reshape(maxindexL2,[1,N_a])+N_d*n2long*(0:N_a-1);
                EV_at_policy_ze=reshape(EVfine_ze(linidx_ze),[N_a,1]);
                V(:,z_c,e_c,jj)=Vhat_jj_ze+(beta-beta0beta)*EV_at_policy_ze;
            end
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

end
