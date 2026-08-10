function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, n_e,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated QH + SemiExo + DC, with d1, z, e.

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod(n_d);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

Vhat=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

eind=shiftdim(gpuArray(0:1:N_e-1),-2);
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-2);
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-2);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

Vhat_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Vunderbar_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2);

level1ii=round(linspace(1,n_a,vfoptions.level1n));

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==3 % joint bothz, inner e
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_e=ones(1,length(n_e));
end

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz,n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

            [~,maxindex1]=max(ReturnMatrix_d2ii,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_d2ii,[N_d1*N_a,vfoptions.level1n,N_bothz,N_e]),[],1);

            Vhat_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Vunderbar_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Vunderbar_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*bothzind+N_d1*N_bothz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Vunderbar_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*bothzind+N_d1*N_bothz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

                [~,maxindex1]=max(ReturnMatrix_d2ii,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_d2ii,[N_d1*N_a,vfoptions.level1n,N_bothz]),[],1);

                Vhat_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Vunderbar_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Vunderbar_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*bothzind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Vunderbar_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*bothzind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), z_valblock, e_val, ReturnFnParamsVec,1);

                    [~,maxindex1]=max(ReturnMatrix_d2iize,[],2);
                    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_d2iize,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

                    Vhat_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                    Vunderbar_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                            aprimeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,2);
                            [Vtempii,maxindex]=max(ReturnMatrix_iize,[],1);
                            Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            Vunderbar_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            allind=dind+N_d1*semizind;
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,2);
                            [Vtempii,maxindex]=max(ReturnMatrix_iize,[],1);
                            Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            Vunderbar_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            allind=dind+N_d1*semizind;
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                        end
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);

                    [~,maxindex1]=max(ReturnMatrix_d2iize,[],2);
                    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_d2iize,[N_d1*N_a,vfoptions.level1n]),[],1);

                    Vhat_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                    Vunderbar_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                            aprimeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                            [Vtempii,maxindex]=max(ReturnMatrix_iize,[],1);
                            Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            Vunderbar_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(dind),size(maxindex))-1));
                        else
                            loweredge=maxindex1(:,1,ii);
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                            [Vtempii,maxindex]=max(ReturnMatrix_iize,[],1);
                            Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            Vunderbar_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(dind),size(maxindex))-1));
                        end
                    end
                end
            end
        end
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],4);
    Vhat(:,:,:,N_j)=V1_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    NN=N_a*N_semiz*N_z*N_e;
    maxindex_lin=reshape(maxindex,[NN,1]);
    Vunderbar(:,:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[N_a,N_semiz*N_z,N_e]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);
    EV=sum(EV.*pi_e_J(1,1,:,N_j+1),3); % N_a x N_bothz

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2); % N_a x 1 x N_bothz

            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz,n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

            entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex1]=max(entireRHS_Vh,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n,N_bothz,N_e]),[],1);
            Vhat_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*bothzBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*bothzind+N_d1*N_bothz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*bothzBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*bothzind+N_d1*N_bothz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                end
            end

            % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
            aprime_ind_d2=ceil(Policy_ford2_jj(:,:,:,d2_c)/N_d1); % N_a x N_bothz x N_e
            EVd2_2d=reshape(EV_d2,[N_a,N_bothz]); % N_a x N_bothz
            EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz,N_e]);
            Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2); % N_a x 1 x N_bothz

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

                entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
                [~,maxindex1]=max(entireRHS_Vh,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n,N_bothz]),[],1);
                Vhat_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        aprimez=aprimeindexes+N_a*bothzBind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*bothzind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                        aprimez=loweredge+N_a*bothzBind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*bothzind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    end
                end
            end

            % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy (after e_c loop, all e at once)
            aprime_ind_d2=ceil(Policy_ford2_jj(:,:,:,d2_c)/N_d1); % N_a x N_bothz x N_e
            EVd2_2d=reshape(EV_d2,[N_a,N_bothz]); % N_a x N_bothz
            EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz,N_e]);
            Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end
    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), z_valblock, e_val, ReturnFnParamsVec,1);

                    entireRHS_Vh=ReturnMatrix_d2iize+beta0beta*shiftdim(EV_d2z,-1);
                    [~,maxindex1]=max(entireRHS_Vh,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);
                    Vhat_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                            aprimeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,2);
                            aprimez=aprimeindexes+N_a*semizBind;
                            entireRHS_ii=ReturnMatrix_iize+beta0beta*reshape(EV_d2z(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            allind=dind+N_d1*semizind;
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,2);
                            aprimez=loweredge+N_a*semizBind;
                            entireRHS_ii=ReturnMatrix_iize+beta0beta*reshape(EV_d2z(aprimez),[N_d1,1,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            allind=dind+N_d1*semizind;
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                        end
                    end
                end

                % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
                aprime_ind_d2=ceil(Policy_ford2_jj(:,semizblock,:,d2_c)/N_d1); % N_a x N_semiz x N_e
                EVd2z_2d=reshape(EV_d2z,[N_a,N_semiz]);
                EV_at_policy_d2=reshape(EVd2z_2d(aprime_ind_d2+N_a*reshape(0:1:N_semiz-1,1,N_semiz)),[N_a,N_semiz,N_e]);
                Vunderbar_ford2_jj(:,semizblock,:,d2_c)=Vhat_ford2_jj(:,semizblock,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);

                    entireRHS_Vh=ReturnMatrix_d2iize+beta0beta*shiftdim(EV_d2z,-1);
                    [~,maxindex1]=max(entireRHS_Vh,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n]),[],1);
                    Vhat_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(maxindex2,1);
                    maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                            aprimeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                            aprimez=aprimeindexes;
                            entireRHS_ii=ReturnMatrix_iize+beta0beta*reshape(EV_d2z(aprimez),[N_d1*(maxgap(ii)+1),1]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(dind),size(maxindex))-1));
                        else
                            loweredge=maxindex1(:,1,ii);
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                            aprimez=loweredge;
                            entireRHS_ii=ReturnMatrix_iize+beta0beta*reshape(EV_d2z(aprimez),[N_d1,1]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(dind),size(maxindex))-1));
                        end
                    end
                end

                aprime_ind_d2=ceil(Policy_ford2_jj(:,z_c,:,d2_c)/N_d1);
                EV_at_policy_d2=reshape(EV_d2z(aprime_ind_d2),[N_a,1,N_e]);
                Vunderbar_ford2_jj(:,z_c,:,d2_c)=Vhat_ford2_jj(:,z_c,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
            end
        end
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],4);
    Vhat(:,:,:,N_j)=V1_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    NN=N_a*N_semiz*N_z*N_e;
    maxindex_lin=reshape(maxindex,[NN,1]);
    Vunderbar(:,:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[N_a,N_semiz*N_z,N_e]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
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

    EV=Vunderbar(:,:,:,jj+1);
    EV=sum(EV.*pi_e_J(1,1,:,jj+1),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz,n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

            entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex1]=max(entireRHS_Vh,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n,N_bothz,N_e]),[],1);
            Vhat_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    aprimez=aprimeindexes+N_a*bothzBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*bothzind+N_d1*N_bothz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    aprimez=loweredge+N_a*bothzBind;
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vhat_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*bothzind+N_d1*N_bothz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                end
            end

            aprime_ind_d2=ceil(Policy_ford2_jj(:,:,:,d2_c)/N_d1);
            EVd2_2d=reshape(EV_d2,[N_a,N_bothz]);
            EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz,N_e]);
            Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);

                entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
                [~,maxindex1]=max(entireRHS_Vh,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n,N_bothz]),[],1);
                Vhat_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex2,1);
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                        aprimez=aprimeindexes+N_a*bothzBind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*bothzind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                        aprimez=loweredge+N_a*bothzBind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        Vhat_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind+N_d1*bothzind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                    end
                end
            end

            aprime_ind_d2=ceil(Policy_ford2_jj(:,:,:,d2_c)/N_d1);
            EVd2_2d=reshape(EV_d2,[N_a,N_bothz]);
            EV_at_policy_d2=reshape(EVd2_2d(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz,N_e]);
            Vunderbar_ford2_jj(:,:,:,d2_c)=Vhat_ford2_jj(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
        end
    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), z_valblock, e_val, ReturnFnParamsVec,1);

                    entireRHS_Vh=ReturnMatrix_d2iize+beta0beta*shiftdim(EV_d2z,-1);
                    [~,maxindex1]=max(entireRHS_Vh,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);
                    Vhat_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(maxindex2,1);
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                            aprimeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,2);
                            aprimez=aprimeindexes+N_a*semizBind;
                            entireRHS_ii=ReturnMatrix_iize+beta0beta*reshape(EV_d2z(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            allind=dind+N_d1*semizind;
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,2);
                            aprimez=loweredge+N_a*semizBind;
                            entireRHS_ii=ReturnMatrix_iize+beta0beta*reshape(EV_d2z(aprimez),[N_d1,1,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            Vhat_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            allind=dind+N_d1*semizind;
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
                        end
                    end
                end

                % Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy
                aprime_ind_d2=ceil(Policy_ford2_jj(:,semizblock,:,d2_c)/N_d1); % N_a x N_semiz x N_e
                EVd2z_2d=reshape(EV_d2z,[N_a,N_semiz]);
                EV_at_policy_d2=reshape(EVd2z_2d(aprime_ind_d2+N_a*reshape(0:1:N_semiz-1,1,N_semiz)),[N_a,N_semiz,N_e]);
                Vunderbar_ford2_jj(:,semizblock,:,d2_c)=Vhat_ford2_jj(:,semizblock,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);

                    entireRHS_Vh=ReturnMatrix_d2iize+beta0beta*shiftdim(EV_d2z,-1);
                    [~,maxindex1]=max(entireRHS_Vh,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n]),[],1);
                    Vhat_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(maxindex2,1);
                    maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
                            aprimeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                            aprimez=aprimeindexes;
                            entireRHS_ii=ReturnMatrix_iize+beta0beta*reshape(EV_d2z(aprimez),[N_d1*(maxgap(ii)+1),1]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(dind),size(maxindex))-1));
                        else
                            loweredge=maxindex1(:,1,ii);
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,2);
                            aprimez=loweredge;
                            entireRHS_ii=ReturnMatrix_iize+beta0beta*reshape(EV_d2z(aprimez),[N_d1,1]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            Vhat_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1)+1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(dind),size(maxindex))-1));
                        end
                    end
                end

                aprime_ind_d2=ceil(Policy_ford2_jj(:,z_c,:,d2_c)/N_d1);
                EV_at_policy_d2=reshape(EV_d2z(aprime_ind_d2),[N_a,1,N_e]);
                Vunderbar_ford2_jj(:,z_c,:,d2_c)=Vhat_ford2_jj(:,z_c,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
            end
        end
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],4);
    Vhat(:,:,:,jj)=V1_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1);
    NN=N_a*N_semiz*N_z*N_e;
    maxindex_lin=reshape(maxindex,[NN,1]);
    Vunderbar(:,:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[N_a,N_semiz*N_z,N_e]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:NN)'+NN*(maxindex_lin-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(1,:,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy(3,:,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

end


end
