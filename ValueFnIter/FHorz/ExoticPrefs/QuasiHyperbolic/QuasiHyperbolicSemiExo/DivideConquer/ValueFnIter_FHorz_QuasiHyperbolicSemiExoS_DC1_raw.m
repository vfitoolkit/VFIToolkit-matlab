function [Vhat,Policy3,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_raw(n_d1,n_d2,n_a,n_z,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated QH + SemiExo + DC (with d1, z, no e).
% Agent maximises Vhat = u + beta0*beta*E[Vunderbar_{j+1}].
% Vunderbar = Vhat + (beta - beta0beta)*EV_at_policy   (long-run value)
% Output: [Vhat, Policy3, Vunderbar]  (Vhat is V1; cross-d2 max on Vhat per the agent's choice)

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod(n_d);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

Vhat=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz*N_z,N_j,'gpuArray');

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-2);

% Per-d2 slabs: Vhat (agent's value, used for d2 choice), Vunderbar (long-run, post-correction)
Vhat_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Vunderbar_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % Terminal: Vhat=Vunderbar=u, no discounting
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        [~,maxindex1]=max(ReturnMatrix_d2ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_d2ii,[N_d1*N_a,vfoptions.level1n,N_bothz]),[],1);
        Vhat_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Vunderbar_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1);

        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Vunderbar_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*bothzind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                Vunderbar_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*bothzind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            end
        end
    end
    % Cross-d2 max on Vhat
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,N_j)=V1_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    Vunderbar(:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[N_a,N_semiz*N_z]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
    Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

else
    % Using V_Jplus1 (= Vunderbar for sophisticated)
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2); % N_a-by-1-by-N_bothz

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        %% Vhat slab (beta0*beta) -> Policy
        entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
        [~,maxindex1]=max(entireRHS_Vh,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n,N_bothz]),[],1);
        Vhat_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*bothzBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*bothzind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*bothzBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*bothzind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            end
        end

        %% Vunderbar slab: Vhat + (beta-beta0beta)*EV_at_policy
        % EV_d2 is N_a-by-1-by-N_bothz -> use aprime ind chosen for this d2
        aprime_ind_d2=ceil(Policy_ford2_jj(:,:,d2_c)/N_d1);
        EV_at_policy_d2=reshape(EV_d2(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz]);
        Vunderbar_ford2_jj(:,:,d2_c)=Vhat_ford2_jj(:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
    end
    % Cross-d2 max on Vhat (V1)
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,N_j)=V1_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    Vunderbar(:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[N_a,N_semiz*N_z]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
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

    EV=Vunderbar(:,:,jj+1); % S: continuation is Vunderbar

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2);

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_Vh=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
        [~,maxindex1]=max(entireRHS_Vh,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_Vh,[N_d1*N_a,vfoptions.level1n,N_bothz]),[],1);
        Vhat_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
        Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1);
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*bothzBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1*(maxgap(ii)+1),1,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*bothzind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=loweredge+N_a*bothzBind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,1,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d1)+1);
                allind=dind+N_d1*bothzind;
                Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1));
            end
        end

        aprime_ind_d2=ceil(Policy_ford2_jj(:,:,d2_c)/N_d1);
        EV_at_policy_d2=reshape(EV_d2(aprime_ind_d2+N_a*reshape(0:1:N_bothz-1,1,N_bothz)),[N_a,N_bothz]);
        Vunderbar_ford2_jj(:,:,d2_c)=Vhat_ford2_jj(:,:,d2_c)+(beta-beta0beta)*EV_at_policy_d2;
    end
    [V1_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,jj)=V1_jj;
    Policy3(2,:,:,jj)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    Vunderbar(:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[N_a,N_semiz*N_z]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
    Policy3(1,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

end


end
