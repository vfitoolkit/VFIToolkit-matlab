function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_nod1_raw(n_d2,n_a,n_z,n_semiz,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive QH + SemiExo + DC, no d1, with z, no e. Dual-Valt; cross-d2 max on Vtilde.

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

Valt=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
% No d1, so Policy here stores (d2,aprime); shape (2, ...)
Policy=zeros(2,N_a,N_semiz*N_z,N_j,'gpuArray');
Policyalt=zeros(2,N_a,N_semiz*N_z,N_j,'gpuArray'); % exponential discounter optimal (d2, aprime)

%%
special_n_d2=ones(1,length(n_d2));

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1);

loweredgesize=[1,1,N_semiz*N_z];

V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Vtilde_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);

        [Vtempii,maxindex1]=max(ReturnMatrix_d2ii,[],1);

        V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Vtilde_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex1,1);

        maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Vtilde_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+(loweredge-1));
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(ReturnMatrix_ii,1);
                Vtilde_ford2_jj(curraindex,:,d2_c)=shiftdim(ReturnMatrix_ii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
    end
    [V1_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
    Vtilde(:,:,N_j)=V1_jj;
    Policy(1,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    Valt(:,:,N_j)=reshape(V_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[N_a,N_semiz*N_z]);
    Policy(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
    % terminal: QH and exponential discounter coincide
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j);

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

        %% Valt slab (beta)
        entireRHS_V=ReturnMatrix_d2ii+beta*EV_d2;
        [Vtempii_V,maxindex1_V]=max(entireRHS_V,[],1);
        V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii_V,1);
        Policy_V_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex1_V,1);
        maxgap_V=squeeze(max(maxindex1_V(1,2:end,:)-maxindex1_V(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1_V(1,ii,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                aprimez=aprimeindexes+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[(maxgap_V(ii)+1),1,N_bothz]);
                [Vtempii,maxindex2alt]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Policy_V_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex2alt+(loweredge-1));
            else
                loweredge=maxindex1_V(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                aprimez=loweredge+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[1,1,N_bothz]);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(entireRHS_ii,1);
                Policy_V_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end

        %% Vtilde slab (beta0*beta) -> Policy
        entireRHS_Vt=ReturnMatrix_d2ii+beta0beta*EV_d2;
        [Vtempii,maxindex1]=max(entireRHS_Vt,[],1);
        Vtilde_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
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
                Vtilde_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+(loweredge-1));
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                aprimez=loweredge+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[1,1,N_bothz]);
                Vtilde_ford2_jj(curraindex,:,d2_c)=shiftdim(entireRHS_ii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
    end
    [V1_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
    Vtilde(:,:,N_j)=V1_jj;
    Policy(1,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    Policy(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
    % Valt at exponential discounter optimum (full max over d2 and aprime)
    [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],3);
    Valt(:,:,N_j)=V_jj;
    Policyalt(1,:,:,N_j)=shiftdim(maxindexalt_d2,-1);
    maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z,1]);
    Policyalt(2,:,:,N_j)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);
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

    EV=Valt(:,:,jj+1);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2);

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,4);

        %% Valt slab (beta)
        entireRHS_V=ReturnMatrix_d2ii+beta*EV_d2;
        [Vtempii_V,maxindex1_V]=max(entireRHS_V,[],1);
        V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii_V,1);
        Policy_V_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex1_V,1);
        maxgap_V=squeeze(max(maxindex1_V(1,2:end,:)-maxindex1_V(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1_V(1,ii,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                aprimez=aprimeindexes+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[(maxgap_V(ii)+1),1,N_bothz]);
                [Vtempii,maxindex2alt]=max(entireRHS_ii,[],1);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Policy_V_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex2alt+(loweredge-1));
            else
                loweredge=maxindex1_V(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                aprimez=loweredge+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[1,1,N_bothz]);
                V_ford2_jj(curraindex,:,d2_c)=shiftdim(entireRHS_ii,1);
                Policy_V_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end

        %% Vtilde slab (beta0*beta) -> Policy
        entireRHS_Vt=ReturnMatrix_d2ii+beta0beta*EV_d2;
        [Vtempii,maxindex1]=max(entireRHS_Vt,[],1);
        Vtilde_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
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
                Vtilde_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+(loweredge-1));
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                aprimez=loweredge+N_a*bothzind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[1,1,N_bothz]);
                Vtilde_ford2_jj(curraindex,:,d2_c)=shiftdim(entireRHS_ii,1);
                Policy_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
            end
        end
    end
    [V1_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
    Vtilde(:,:,jj)=V1_jj;
    Policy(1,:,:,jj)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    Policy(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
    % Valt at exponential discounter optimum (full max over d2 and aprime)
    [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],3);
    Valt(:,:,jj)=V_jj;
    Policyalt(1,:,:,jj)=shiftdim(maxindexalt_d2,-1);
    maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z,1]);
    Policyalt(2,:,:,jj)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);

end


end
