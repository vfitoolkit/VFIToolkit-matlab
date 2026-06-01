function [Vtilde, Policy3, Valt, Policy3alt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_noz_raw(n_d1, n_d2, n_a, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic with semi-exogenous shock, no z.

n_d=[n_d1,n_d2];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod([n_d1,n_d2]);
N_a=prod(n_a);
N_semiz=prod(n_semiz);

Valt=zeros(N_a,N_semiz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz,N_j,'gpuArray');
Policy3alt=zeros(3,N_a,N_semiz,N_j,'gpuArray'); % exponential discounter optimal (d1, d2, aprime)

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

V_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Vtilde_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_V_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');

%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_semiz, d_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,N_j)=Vtemp;
        d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        Policy3(1,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy3(2,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_semiz, d_gridvals, a_grid, z_val, ReturnFnParamsVec,0);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Valt(:,z_c,N_j)=Vtemp;
            d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
            Policy3(1,:,z_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy3(2,:,z_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy3(3,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        end
    end
    Vtilde=Valt;
    % terminal: QH and exponential discounter coincide
    Policy3alt(:,:,:,N_j)=Policy3(:,:,:,N_j);
else
    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, n_semiz, d12c_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireEV=repelem(EV_d2,N_d1,1,1);

            entireRHS_V=ReturnMatrix_d2+beta*entireEV;
            [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_V_ford2_jj(:,:,d2_c)=shiftdim(maxindexalt,1);

            entireRHS=ReturnMatrix_d2+beta0beta*entireEV;
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);
        end
        [Vtilde_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
        Vtilde(:,:,N_j)=Vtilde_jj;
        Policy3(2,:,:,N_j)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
        Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
        % Valt at exponential discounter optimum (full max over d2, d1, aprime)
        [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],3);
        Valt(:,:,N_j)=V_jj;
        Policy3alt(2,:,:,N_j)=shiftdim(maxindexalt_d2,-1);
        maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz,1]);
        d1aprime_ind_alt=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt_lin-1)),[1,N_a,N_semiz]);
        Policy3alt(1,:,:,N_j)=shiftdim(rem(d1aprime_ind_alt-1,N_d1)+1,-1);
        Policy3alt(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind_alt/N_d1),-1);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, special_n_semiz, d12c_gridvals, a_grid, z_val, ReturnFnParamsVec,0);

                EV_d2z=EV.*pi_semiz(z_c,:);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                entireEV_z=repelem(EV_d2z,N_d1,1,1);

                entireRHS_V=ReturnMatrix_d2z+beta*entireEV_z;
                [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
                V_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_V_ford2_jj(:,z_c,d2_c)=maxindexalt;

                entireRHS_z=ReturnMatrix_d2z+beta0beta*entireEV_z;
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                Vtilde_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;
            end
        end
        [Vtilde_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
        Vtilde(:,:,N_j)=Vtilde_jj;
        Policy3(2,:,:,N_j)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
        Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
        % Valt at exponential discounter optimum (full max over d2, d1, aprime)
        [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],3);
        Valt(:,:,N_j)=V_jj;
        Policy3alt(2,:,:,N_j)=shiftdim(maxindexalt_d2,-1);
        maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz,1]);
        d1aprime_ind_alt=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt_lin-1)),[1,N_a,N_semiz]);
        Policy3alt(1,:,:,N_j)=shiftdim(rem(d1aprime_ind_alt-1,N_d1)+1,-1);
        Policy3alt(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind_alt/N_d1),-1);
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

    EV=Valt(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, n_semiz, d12c_gridvals, a_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireEV=repelem(EV_d2,N_d1,1,1);

            entireRHS_V=ReturnMatrix_d2+beta*entireEV;
            [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_V_ford2_jj(:,:,d2_c)=shiftdim(maxindexalt,1);

            entireRHS=ReturnMatrix_d2+beta0beta*entireEV;
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);
        end
        [Vtilde_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
        Vtilde(:,:,jj)=Vtilde_jj;
        Policy3(2,:,:,jj)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
        Policy3(1,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        Policy3(3,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
        % Valt at exponential discounter optimum (full max over d2, d1, aprime)
        [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],3);
        Valt(:,:,jj)=V_jj;
        Policy3alt(2,:,:,jj)=shiftdim(maxindexalt_d2,-1);
        maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz,1]);
        d1aprime_ind_alt=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt_lin-1)),[1,N_a,N_semiz]);
        Policy3alt(1,:,:,jj)=shiftdim(rem(d1aprime_ind_alt-1,N_d1)+1,-1);
        Policy3alt(3,:,:,jj)=shiftdim(ceil(d1aprime_ind_alt/N_d1),-1);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, special_n_semiz, d12c_gridvals, a_grid, z_val, ReturnFnParamsVec,0);

                EV_d2z=EV.*pi_semiz(z_c,:);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                entireEV_z=repelem(EV_d2z,N_d1,1,1);

                entireRHS_V=ReturnMatrix_d2z+beta*entireEV_z;
                [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
                V_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_V_ford2_jj(:,z_c,d2_c)=maxindexalt;

                entireRHS_z=ReturnMatrix_d2z+beta0beta*entireEV_z;
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                Vtilde_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;
            end
        end
        [Vtilde_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
        Vtilde(:,:,jj)=Vtilde_jj;
        Policy3(2,:,:,jj)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
        Policy3(1,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        Policy3(3,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
        % Valt at exponential discounter optimum (full max over d2, d1, aprime)
        [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],3);
        Valt(:,:,jj)=V_jj;
        Policy3alt(2,:,:,jj)=shiftdim(maxindexalt_d2,-1);
        maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz,1]);
        d1aprime_ind_alt=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt_lin-1)),[1,N_a,N_semiz]);
        Policy3alt(1,:,:,jj)=shiftdim(rem(d1aprime_ind_alt-1,N_d1)+1,-1);
        Policy3alt(3,:,:,jj)=shiftdim(ceil(d1aprime_ind_alt/N_d1),-1);
    end
end

end
