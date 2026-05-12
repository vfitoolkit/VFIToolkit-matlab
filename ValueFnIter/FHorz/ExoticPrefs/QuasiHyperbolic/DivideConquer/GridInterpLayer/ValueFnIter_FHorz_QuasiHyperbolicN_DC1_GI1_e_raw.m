function varargout=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_GI1_e_raw(n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_GI_e_raw.
% Has d variables. Has z variable. Has e variable. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{d,a'} u + beta*E[V_{j+1}]
%         Vtilde_j = max_{d,a'} u + beta_0*beta*E[V_{j+1}]   (agent's choice)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_z,N_e,N_j,'gpuArray'); % [d_ind; midpoint; aprimeL2ind]

if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d,1,N_a,N_z,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(N_d,1,N_a,N_z,'gpuArray');
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    midpoints_jj=zeros(N_d,1,N_a,'gpuArray');
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
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

pi_e_J=shiftdim(pi_e_J,-2);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoints_jj(:,1,level1ii,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,6);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind+N_d*N_a*N_z*eind;
        Policy(1,:,:,:,N_j)=d_ind;
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
        Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoints_jj(:,1,level1ii,:)=maxindex1;
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,6);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zind;
            Policy(1,:,:,e_c,N_j)=d_ind;
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);
                midpoints_jj(:,1,level1ii)=maxindex1;
                maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,6);
                        [~,maxindex]=max(ReturnMatrix_ii,[],2);
                        midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
                    end
                end
                midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind;
                Policy(1,:,z_c,e_c,N_j)=d_ind;
                Policy(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
                Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
            end
        end
    end

    Vtilde=V;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*pi_e_J(1,1,:,N_j),3);
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);   % N_a-by-1-by-N_z
    EVinterp=interp1(a_grid,EV,aprime_grid);

    Vtilde=zeros(N_a,N_z,N_e,N_j,'gpuArray');

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        %% V (beta)
        entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:,:)=maxindex1;
        maxgap_V=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,6);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimez(:)),[N_d*(maxgap_V(ii)+1),1,N_z,N_e]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zBind;
        entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z,N_e]);
        [Vtempii,~]=max(entireRHS_L2,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,6);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez(:)),[N_d*(maxgap(ii)+1),1,N_z,N_e]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zBind;
        entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        Vtilde(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind+N_d*N_a*N_z*eind;
        Policy(1,:,:,:,N_j)=d_ind;
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
        Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            %% V (beta)
            entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,level1ii,:)=maxindex1;
            maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,6);
                    aprimez=aprimeindexes+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimez(:)),[N_d*(maxgap_V(ii)+1),1,N_z]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*zBind;
            entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z]);
            [Vtempii,~]=max(entireRHS_L2,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,level1ii,:)=maxindex1;
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,6);
                    aprimez=aprimeindexes+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez(:)),[N_d*(maxgap(ii)+1),1,N_z]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*zBind;
            entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            Vtilde(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zind;
            Policy(1,:,:,e_c,N_j)=d_ind;
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                %% V (beta)
                entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV_z,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,level1ii)=maxindex1;
                maxgap_V=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap_V(ii));
                        aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,6);
                        entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV_z(aprimeindexes(:)),[N_d*(maxgap_V(ii)+1),1]);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
                    end
                end
                midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_z(aprimeindexes(:)),[N_d*n2long,N_a]);
                [Vtempii,~]=max(entireRHS_L2,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                %% Vtilde (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV_z,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,level1ii)=maxindex1;
                maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,6);
                        entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV_z(aprimeindexes(:)),[N_d*(maxgap(ii)+1),1]);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
                    end
                end
                midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_z(aprimeindexes(:)),[N_d*n2long,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                Vtilde(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind;
                Policy(1,:,z_c,e_c,N_j)=d_ind;
                Policy(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
                Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
            end
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EV=sum(V(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);   % N_a-by-1-by-N_z
    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        %% V (beta)
        entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:,:)=maxindex1;
        maxgap_V=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,6);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimez(:)),[N_d*(maxgap_V(ii)+1),1,N_z,N_e]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zBind;
        entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z,N_e]);
        [Vtempii,~]=max(entireRHS_L2,[],1);
        V(:,:,:,jj)=shiftdim(Vtempii,1);
        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,6);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez(:)),[N_d*(maxgap(ii)+1),1,N_z,N_e]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zBind;
        entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        Vtilde(:,:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind+N_d*N_a*N_z*eind;
        Policy(1,:,:,:,jj)=d_ind;
        Policy(2,:,:,:,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1);
        Policy(3,:,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);
            %% V (beta)
            entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,level1ii,:)=maxindex1;
            maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,6);
                    aprimez=aprimeindexes+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimez(:)),[N_d*(maxgap_V(ii)+1),1,N_z]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),e_val,ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*zBind;
            entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z]);
            [Vtempii,~]=max(entireRHS_L2,[],1);
            V(:,:,e_c,jj)=shiftdim(Vtempii,1);
            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(:,1,level1ii,:)=maxindex1;
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,6);
                    aprimez=aprimeindexes+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez(:)),[N_d*(maxgap(ii)+1),1,N_z]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),e_val,ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*zBind;
            entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            Vtilde(:,:,e_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zind;
            Policy(1,:,:,e_c,jj)=d_ind;
            Policy(2,:,:,e_c,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1);
            Policy(3,:,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                %% V (beta)
                entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV_z,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,level1ii)=maxindex1;
                maxgap_V=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap_V(ii));
                        aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,6);
                        entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV_z(aprimeindexes(:)),[N_d*(maxgap_V(ii)+1),1]);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
                    end
                end
                midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_z(aprimeindexes(:)),[N_d*n2long,N_a]);
                [Vtempii,~]=max(entireRHS_L2,[],1);
                V(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                %% Vtilde (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV_z,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,level1ii)=maxindex1;
                maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,6);
                        entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV_z(aprimeindexes(:)),[N_d*(maxgap(ii)+1),1]);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
                    end
                end
                midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_z(aprimeindexes(:)),[N_d*n2long,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                Vtilde(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind;
                Policy(1,:,z_c,e_c,jj)=d_ind;
                Policy(2,:,z_c,e_c,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1);
                Policy(3,:,z_c,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1);
            end
        end
    end
end

%% Post-process Policy: convert [d_ind, midpoint, aprimeL2ind] to canonical combined index
adjust=(Policy(3,:,:,:,:)<1+n2short+1);
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust;
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1);

Policy=squeeze(Policy(1,:,:,:,:)+N_d*(Policy(2,:,:,:,:)-1)+N_d*N_a*(Policy(3,:,:,:,:)-1));

%%
nOutputs=nargout;
if nOutputs==2
    varargout={Vtilde,Policy};
elseif nOutputs==3
    varargout={Vtilde,Policy,V};
end

end
