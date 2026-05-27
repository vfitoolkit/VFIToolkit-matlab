function [Vunderbar,Policy3,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_noz_raw(n_d1,n_d2,n_a,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated QH + SemiExo + DC, with d1, no z, no e.

n_d=[n_d1,n_d2];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod(n_d);
N_a=prod(n_a);
N_semiz=prod(n_semiz);

Vhat=zeros(N_a,N_semiz,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz,N_j,'gpuArray');

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-2);

Vhat_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Vunderbar_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        [~,maxindex1]=max(ReturnMatrix_d2ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_d2ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

        Vhat_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Vunderbar_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1);

        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Vunderbar_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*semizind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Vunderbar_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*semizind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            end
        end
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,N_j)=V1_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
    Vunderbar(:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[N_a,N_semiz]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

        EV_d2=EV.*shiftdim(pi_semiz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2);

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
        [~,maxindex1]=max(entireRHS_Vh,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);
        Vhat_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*semizBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*semizind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*semizBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_semiz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*semizind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            end
        end

        aprime_ind_d2=ceil(Policy_ford2_jj(:,:,d2_c)/N_d1);
        EV_at_policy_d2=reshape(EV_d2(aprime_ind_d2+N_a*reshape(0:1:N_semiz-1,1,N_semiz)),[N_a,N_semiz]);
        Vunderbar_ford2_jj(:,:,d2_c)=Vhat_ford2_jj(:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,N_j)=V1_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
    Vunderbar(:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[N_a,N_semiz]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
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

    EV=Vunderbar(:,:,jj+1);

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_semiz=pi_semiz_J(:,:,d2_c,jj);

        EV_d2=EV.*shiftdim(pi_semiz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2);

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
        [~,maxindex1]=max(entireRHS_Vh,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);
        Vhat_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*semizBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*semizind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*semizBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_semiz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*semizind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            end
        end

        aprime_ind_d2=ceil(Policy_ford2_jj(:,:,d2_c)/N_d1);
        EV_at_policy_d2=reshape(EV_d2(aprime_ind_d2+N_a*reshape(0:1:N_semiz-1,1,N_semiz)),[N_a,N_semiz]);
        Vunderbar_ford2_jj(:,:,d2_c)=Vhat_ford2_jj(:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,jj)=V1_jj;
    Policy3(2,:,:,jj)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
    Vunderbar(:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[N_a,N_semiz]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

end


end
