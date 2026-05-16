function [Vunderbar,Policy2,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_e_raw(n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_e_raw.
% Has d, z, and e variables. GPU (parallel==2 only).
%
% Sophisticated: Vhat_j = max u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EV_at_optimal_aprime

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray');

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
else
    eind=shiftdim(gpuArray(0:1:N_e-1),-2);
end
if vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
else
    zind=shiftdim(gpuArray(0:1:N_z-1),-1);
end

zBind=shiftdim(gpuArray(0:1:N_z-1),-2);
zindC=gpuArray(0:1:N_z-1);  % 1-by-N_z (for EV_at_Policy indexing)

level1ii=round(linspace(1,n_a,vfoptions.level1n));

pi_e_J=shiftdim(pi_e_J,-2);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_z,N_e]),[],1);
        Vhat(level1ii,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,N_j)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind;
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind;
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);
            Vhat(level1ii,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,N_j)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind;
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind;
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);
                Vhat(level1ii,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(level1ii,z_c,e_c,N_j)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(dind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(dind)-1),1);
                    end
                end
            end
        end
    end
    Vunderbar=Vhat;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*pi_e_J(1,1,:,N_j),3);
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z,N_e]),[],1);
        Vhat(level1ii,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,N_j)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[N_d*(maxgap(ii)+1),1,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind;
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[N_d*1,1,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind;
                Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
        aprime_ind=ceil(Policy(:,:,:,N_j)/N_d);
        EV_at_policy=EV(aprime_ind+N_a*zindC);
        Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            EV_e=EV;
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV_e,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);
            Vhat(level1ii,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,N_j)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_e(aprimez),[N_d*(maxgap(ii)+1),1,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind;
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_e(aprimez),[N_d*1,1,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind;
                    Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                end
            end
            aprime_ind_e=ceil(Policy(:,:,e_c,N_j)/N_d);
            EV_at_policy_e=EV_e(aprime_ind_e+N_a*zindC);
            Vunderbar(:,:,e_c,N_j)=Vhat(:,:,e_c,N_j)+(beta-beta0beta)*EV_at_policy_e;
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV_z,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
                Vhat(level1ii,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(level1ii,z_c,e_c,N_j)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_z(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(dind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(loweredge);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex+N_d*(loweredge(dind)-1),1);
                    end
                end
                aprime_ind_ze=ceil(Policy(:,z_c,e_c,N_j)/N_d);
                EV_at_policy_ze=EV_z(aprime_ind_ze);
                Vunderbar(:,z_c,e_c,N_j)=Vhat(:,z_c,e_c,N_j)+(beta-beta0beta)*EV_at_policy_ze;
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

    EVsource=Vunderbar(:,:,:,jj+1);
    EV=sum(EVsource.*pi_e_J(1,1,:,jj),3);
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z,N_e]),[],1);
        Vhat(level1ii,:,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,:,jj)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[N_d*(maxgap(ii)+1),1,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind;
                Policy(curraindex,:,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV(aprimez),[N_d*1,1,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                allind=dind+N_d*zind+N_d*N_z*eind;
                Policy(curraindex,:,:,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end
        aprime_ind=ceil(Policy(:,:,:,jj)/N_d);
        EV_at_policy=EV(aprime_ind+N_a*zindC);
        Vunderbar(:,:,:,jj)=Vhat(:,:,:,jj)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            EV_e=EV;
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV_e,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);
            Vhat(level1ii,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,:,e_c,jj)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_e(aprimez),[N_d*(maxgap(ii)+1),1,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind;
                    Policy(curraindex,:,e_c,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*zBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_e(aprimez),[N_d*1,1,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=dind+N_d*zind;
                    Policy(curraindex,:,e_c,jj)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                end
            end
            aprime_ind_e=ceil(Policy(:,:,e_c,jj)/N_d);
            EV_at_policy_e=EV_e(aprime_ind_e+N_a*zindC);
            Vunderbar(:,:,e_c,jj)=Vhat(:,:,e_c,jj)+(beta-beta0beta)*EV_at_policy_e;
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV_z,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
                Vhat(level1ii,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(level1ii,z_c,e_c,jj)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_z(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex+N_d*(loweredge(dind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid(loweredge), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z(loweredge);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex+N_d*(loweredge(dind)-1),1);
                    end
                end
                aprime_ind_ze=ceil(Policy(:,z_c,e_c,jj)/N_d);
                EV_at_policy_ze=EV_z(aprime_ind_ze);
                Vunderbar(:,z_c,e_c,jj)=Vhat(:,z_c,e_c,jj)+(beta-beta0beta)*EV_at_policy_ze;
            end
        end
    end
end

%%
Policy2=zeros(2,N_a,N_z,N_e,N_j,'gpuArray');
Policy2(1,:,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
