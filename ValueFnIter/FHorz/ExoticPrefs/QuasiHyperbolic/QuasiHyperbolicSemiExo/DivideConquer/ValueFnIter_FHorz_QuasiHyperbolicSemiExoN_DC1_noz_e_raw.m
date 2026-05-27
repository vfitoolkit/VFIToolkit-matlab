function [Vtilde,Policy3,V]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_noz_e_raw(n_d1,n_d2,n_a,n_semiz, n_e,N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive QH + SemiExo + DC, with d1, no z, with e. Dual-V; cross-d2 max on Vtilde.

n_d=[n_d1,n_d2];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod(n_d);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

eind=shiftdim(gpuArray(0:1:N_e-1),-2);
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-2);

V_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Vtilde_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2);

level1ii=round(linspace(1,n_a,vfoptions.level1n));

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states');
end

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

            [~,maxindex1]=max(ReturnMatrix_d2ii,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_d2ii,[N_d1*N_a,vfoptions.level1n,N_semiz,N_e]),[],1);

            V_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Vtilde_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind+N_d1*N_semiz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind+N_d1*N_semiz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

                [~,maxindex1]=max(ReturnMatrix_d2ii,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_d2ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

                V_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Vtilde_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*semizind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*semizind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    end
                end
            end
        end
    end
    [V1_jj,maxindex]=max(Vtilde_ford2_jj,[],4);
    Vtilde(:,:,:,N_j)=V1_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    NN=N_a*N_semiz*N_e;
    maxindex_lin=reshape(maxindex,[NN,1]);
    V(:,:,:,N_j)=reshape(V_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[N_a,N_semiz,N_e]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]);
    EV=sum(EV.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

            %% V (beta)
            entireRHS_V=ReturnMatrix_d2ii+beta*shiftdim(EV_d2,-1);
            [~,maxindex1_V]=max(entireRHS_V,[],2);
            [Vtempii_V,~]=max(reshape(entireRHS_V,[N_d1*N_a,vfoptions.level1n,N_semiz,N_e]),[],1);
            V_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii_V,1);
            maxgap_V=squeeze(max(max(max(maxindex1_V(:,1,2:end,:,:)-maxindex1_V(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1_V(:,1,ii,:,:),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[N_d1*(maxgap_V(ii)+1),1,N_semiz,N_e]);
                    [Vtempii,~]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                else
                    loweredge=maxindex1_V(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[N_d1,1,N_semiz,N_e]);
                    [Vtempii,~]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                end
            end

            %% Vtilde (beta0*beta)
            entireRHS_Vt=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex1]=max(entireRHS_Vt,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_Vt,[N_d1*N_a,vfoptions.level1n,N_semiz,N_e]),[],1);
            Vtilde_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind+N_d1*N_semiz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind+N_d1*N_semiz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

                %% V (beta)
                entireRHS_V=ReturnMatrix_d2ii+beta*shiftdim(EV_d2,-1);
                [~,maxindex1_V]=max(entireRHS_V,[],2);
                [Vtempii_V,~]=max(reshape(entireRHS_V,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);
                V_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii_V,1);
                maxgap_V=squeeze(max(max(maxindex1_V(:,1,2:end,:)-maxindex1_V(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1_V(:,1,ii,:),n_a-maxgap_V(ii));
                        aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        aprimez=aprimeindexes+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[N_d1*(maxgap_V(ii)+1),1,N_semiz]);
                        [Vtempii,~]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    else
                        loweredge=maxindex1_V(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        aprimez=loweredge+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[N_d1,1,N_semiz]);
                        [Vtempii,~]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    end
                end

                %% Vtilde (beta0*beta)
                entireRHS_Vt=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
                [~,maxindex1]=max(entireRHS_Vt,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_Vt,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);
                Vtilde_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        aprimez=aprimeindexes+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*semizind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        aprimez=loweredge+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*semizind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    end
                end
            end
        end
    end
    [V1_jj,maxindex]=max(Vtilde_ford2_jj,[],4);
    Vtilde(:,:,:,N_j)=V1_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    NN=N_a*N_semiz*N_e;
    maxindex_lin=reshape(maxindex,[NN,1]);
    V(:,:,:,N_j)=reshape(V_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[N_a,N_semiz,N_e]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
end

%% Iterate backwards
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

    EV=V(:,:,:,jj+1);
    EV=sum(EV.*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

            entireRHS_V=ReturnMatrix_d2ii+beta*shiftdim(EV_d2,-1);
            [~,maxindex1_V]=max(entireRHS_V,[],2);
            [Vtempii_V,~]=max(reshape(entireRHS_V,[N_d1*N_a,vfoptions.level1n,N_semiz,N_e]),[],1);
            V_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii_V,1);
            maxgap_V=squeeze(max(max(max(maxindex1_V(:,1,2:end,:,:)-maxindex1_V(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1_V(:,1,ii,:,:),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[N_d1*(maxgap_V(ii)+1),1,N_semiz,N_e]);
                    [Vtempii,~]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                else
                    loweredge=maxindex1_V(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[N_d1,1,N_semiz,N_e]);
                    [Vtempii,~]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                end
            end

            entireRHS_Vt=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex1]=max(entireRHS_Vt,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_Vt,[N_d1*N_a,vfoptions.level1n,N_semiz,N_e]),[],1);
            Vtilde_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind+N_d1*N_semiz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind+N_d1*N_semiz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);

                entireRHS_V=ReturnMatrix_d2ii+beta*shiftdim(EV_d2,-1);
                [~,maxindex1_V]=max(entireRHS_V,[],2);
                [Vtempii_V,~]=max(reshape(entireRHS_V,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);
                V_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii_V,1);
                maxgap_V=squeeze(max(max(maxindex1_V(:,1,2:end,:)-maxindex1_V(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1_V(:,1,ii,:),n_a-maxgap_V(ii));
                        aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                        aprimez=aprimeindexes+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[N_d1*(maxgap_V(ii)+1),1,N_semiz]);
                        [Vtempii,~]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    else
                        loweredge=maxindex1_V(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                        aprimez=loweredge+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[N_d1,1,N_semiz]);
                        [Vtempii,~]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                    end
                end

                entireRHS_Vt=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
                [~,maxindex1]=max(entireRHS_Vt,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_Vt,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);
                Vtilde_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                        aprimez=aprimeindexes+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*semizind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                        aprimez=loweredge+N_a*semizBind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vtilde_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*semizind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    end
                end
            end
        end
    end
    [V1_jj,maxindex]=max(Vtilde_ford2_jj,[],4);
    Vtilde(:,:,:,jj)=V1_jj;
    Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1);
    NN=N_a*N_semiz*N_e;
    maxindex_lin=reshape(maxindex,[NN,1]);
    V(:,:,:,jj)=reshape(V_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[N_a,N_semiz,N_e]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

end


end
