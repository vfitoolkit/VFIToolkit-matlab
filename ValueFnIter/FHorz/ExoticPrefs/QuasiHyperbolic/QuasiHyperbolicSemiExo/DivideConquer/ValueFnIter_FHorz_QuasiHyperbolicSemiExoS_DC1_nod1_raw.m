function [Vunderbar,Policy3,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_nod1_raw(n_d2,n_a,n_z,n_semiz,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated QH + SemiExo + DC, no d1, with z, no e.

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

Vhat=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Policy3=zeros(2,N_a,N_semiz*N_z,N_j,'gpuArray');

%%
special_n_d2=ones(1,length(n_d2));

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1);

loweredgesize=[1,1,N_semiz*N_z];

Vhat_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Vunderbar_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);

        [Vtempii,maxindex1]=max(ReturnMatrix_d2ii,[],1);

        Vhat_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Vunderbar_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex1,1);

        maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Vunderbar_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+(loweredge-1));
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(ReturnMatrix_ii,1);
                Vunderbar_ford2_jj(curraindex,:,d2_c)=shiftdim(ReturnMatrix_ii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,N_j)=V1_jj;
    Policy3(1,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    Vunderbar(:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[N_a,N_semiz*N_z]);
    Policy3(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2);

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);

        entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*EV_d2;
        [Vtempii,maxindex1]=max(entireRHS_Vh,[],1);
        Vhat_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex1,1);
        maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                aprimez=aprimeindexes+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+(loweredge-1));
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                aprimez=loweredge+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[1,1,N_bothz]);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(entireRHS_ii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end

        aprime_ind_d2=Policy_ford2_jj(:,:,d2_c); % aprime directly (no d1)
        EV_at_policy_d2=reshape(EV_d2(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz]);
        Vunderbar_ford2_jj(:,:,d2_c)=Vhat_ford2_jj(:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,N_j)=V1_jj;
    Policy3(1,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    Vunderbar(:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[N_a,N_semiz*N_z]);
    Policy3(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
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
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2);

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,4);

        entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*EV_d2;
        [Vtempii,maxindex1]=max(entireRHS_Vh,[],1);
        Vhat_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex1,1);
        maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                aprimez=aprimeindexes+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+(loweredge-1));
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                aprimez=loweredge+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[1,1,N_bothz]);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(entireRHS_ii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end

        aprime_ind_d2=Policy_ford2_jj(:,:,d2_c);
        EV_at_policy_d2=reshape(EV_d2(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz]);
        Vunderbar_ford2_jj(:,:,d2_c)=Vhat_ford2_jj(:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,jj)=V1_jj;
    Policy3(1,:,:,jj)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    Vunderbar(:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[N_a,N_semiz*N_z]);
    Policy3(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);

end


end
