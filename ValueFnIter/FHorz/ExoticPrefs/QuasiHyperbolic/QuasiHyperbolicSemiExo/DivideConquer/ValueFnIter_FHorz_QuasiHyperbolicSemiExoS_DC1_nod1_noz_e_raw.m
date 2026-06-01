function [Vhat,Policy3,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_nod1_noz_e_raw(n_d2,n_a,n_semiz,n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated QH + SemiExo + DC, no d1, no z, with e.

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

Vhat=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy3=zeros(2,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
special_n_d2=ones(1,length(n_d2));

loweredgesize=[1,1,N_semiz,N_e];

semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
eind=shiftdim(gpuArray(0:1:N_e-1),-2); %#ok<NASGU>

Vhat_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Vunderbar_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2);

level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

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
            d2_val=d2_gridvals(d2_c,:);
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);

            [Vtempii,maxindex1]=max(ReturnMatrix_d2ii,[],1);

            Vhat_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Vunderbar_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex1,1);

            maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Vunderbar_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+(loweredge-1));
                else
                    loweredge=maxindex1(1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(ReturnMatrix_ii,1);
                    Vunderbar_ford2_jj(curraindex,:,:,d2_c)=shiftdim(ReturnMatrix_ii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,4);

                [Vtempii,maxindex1]=max(ReturnMatrix_d2ii,[],1);

                Vhat_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Vunderbar_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex1,1);

                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Vunderbar_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+(loweredge-1));
                    else
                        loweredge=maxindex1(1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, reshape(a_grid(loweredge),[1,1,N_semiz]), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(ReturnMatrix_ii,1);
                        Vunderbar_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(ReturnMatrix_ii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
                    end
                end
            end
        end
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],4);
    Vhat(:,:,:,N_j)=V1_jj;
    Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1);
    NN=N_a*N_semiz*N_e;
    maxindex_lin=reshape(maxindex,[NN,1]);
    Vunderbar(:,:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[N_a,N_semiz,N_e]);
    aprime_ind=reshape(Policy_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);

            entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*EV_d2;
            [Vtempii,maxindex1]=max(entireRHS_Vh,[],1);
            Vhat_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex1,1);
            maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    aprimez=aprimeindexes+N_a*semizind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+(loweredge-1));
                else
                    loweredge=maxindex1(1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    aprimez=loweredge+N_a*semizind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[1,1,N_semiz,N_e]);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(entireRHS_ii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
                end
            end

            aprime_ind_d2=Policy_ford2_jj(:,:,:,d2_c);
            EVd2_2d=reshape(EV_d2,[N_a,N_semiz]);
            EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_semiz-1,1,N_semiz)),[N_a,N_semiz,N_e]);
            Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,4);

                entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*EV_d2;
                [Vtempii,maxindex1]=max(entireRHS_Vh,[],1);
                Vhat_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex1,1);
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        aprimez=aprimeindexes+N_a*semizind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+(loweredge-1));
                    else
                        loweredge=maxindex1(1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, reshape(a_grid(loweredge),[1,1,N_semiz]), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        aprimez=loweredge+N_a*semizind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[1,1,N_semiz]);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(entireRHS_ii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
                    end
                end
            end

            aprime_ind_d2=Policy_ford2_jj(:,:,:,d2_c);
            EVd2_2d=reshape(EV_d2,[N_a,N_semiz]);
            EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_semiz-1,1,N_semiz)),[N_a,N_semiz,N_e]);
            Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],4);
    Vhat(:,:,:,N_j)=V1_jj;
    Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1);
    NN=N_a*N_semiz*N_e;
    maxindex_lin=reshape(maxindex,[NN,1]);
    Vunderbar(:,:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[N_a,N_semiz,N_e]);
    aprime_ind=reshape(Policy_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);
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

    EV=sum(Vunderbar(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,4);

            entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*EV_d2;
            [Vtempii,maxindex1]=max(entireRHS_Vh,[],1);
            Vhat_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex1,1);
            maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                    aprimez=aprimeindexes+N_a*semizind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+(loweredge-1));
                else
                    loweredge=maxindex1(1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                    aprimez=loweredge+N_a*semizind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[1,1,N_semiz,N_e]);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(entireRHS_ii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
                end
            end

            aprime_ind_d2=Policy_ford2_jj(:,:,:,d2_c);
            EVd2_2d=reshape(EV_d2,[N_a,N_semiz]);
            EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_semiz-1,1,N_semiz)),[N_a,N_semiz,N_e]);
            Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,4);

                entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*EV_d2;
                [Vtempii,maxindex1]=max(entireRHS_Vh,[],1);
                Vhat_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex1,1);
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5);
                        aprimez=aprimeindexes+N_a*semizind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+(loweredge-1));
                    else
                        loweredge=maxindex1(1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, reshape(a_grid(loweredge),[1,1,N_semiz]), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5);
                        aprimez=loweredge+N_a*semizind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[1,1,N_semiz]);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(entireRHS_ii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1,1);
                    end
                end
            end

            aprime_ind_d2=Policy_ford2_jj(:,:,:,d2_c);
            EVd2_2d=reshape(EV_d2,[N_a,N_semiz]);
            EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_semiz-1,1,N_semiz)),[N_a,N_semiz,N_e]);
            Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],4);
    Vhat(:,:,:,jj)=V1_jj;
    Policy3(1,:,:,:,jj)=shiftdim(maxindex,-1);
    NN=N_a*N_semiz*N_e;
    maxindex_lin=reshape(maxindex,[NN,1]);
    Vunderbar(:,:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[N_a,N_semiz,N_e]);
    aprime_ind=reshape(Policy_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    Policy3(2,:,:,:,jj)=shiftdim(aprime_ind,-1);

end


end
