function [Vtilde, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_nod1_noz_e_raw(n_d2, n_a, n_semiz, n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic with semi-exogenous shock and iid e shock, no d1, no z.

N_d2=prod(n_d2);
N_d=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

Valt=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy=zeros(2,N_a,N_semiz,N_e,N_j,'gpuArray'); % d2, aprime
Policyalt=zeros(2,N_a,N_semiz,N_e,N_j,'gpuArray'); % exponential discounter optimal (d2, aprime)

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states');
end

V_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Vtilde_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_V_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2);

%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d2, n_a, n_semiz, n_e, d2_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,:,N_j)=Vtemp;
        d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        Policy(1,:,:,:,N_j)=shiftdim(d_ind,-1);
        Policy(2,:,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d2, n_a, n_semiz, special_n_e, d2_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Valt(:,:,e_c,N_j)=Vtemp;
            d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
            Policy(1,:,:,e_c,N_j)=shiftdim(d_ind,-1);
            Policy(2,:,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        end
    end
    Vtilde=Valt;
    % terminal: QH and exponential discounter coincide
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j);
else
    EV_pre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*pi_e_J(1,1,:,N_j),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, n_semiz, n_e, d2_val, a_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

            EV_d2=EV_pre.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireRHS_V=ReturnMatrix_d2+beta*EV_d2;
            [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_V_ford2_jj(:,:,:,d2_c)=shiftdim(maxindexalt,1);

            entireRHS=ReturnMatrix_d2+beta0beta*EV_d2;
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            EV_d2=EV_pre.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, n_semiz, special_n_e, d2_val, a_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);

                entireRHS_V=ReturnMatrix_d2e+beta*EV_d2;
                [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_V_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindexalt,1);

                entireRHS_e=ReturnMatrix_d2e+beta0beta*EV_d2;
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                Vtilde_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);
            end
        end
    end
    [Vtilde_jj,maxindex]=max(Vtilde_ford2_jj,[],4);
    Vtilde(:,:,:,N_j)=Vtilde_jj;
    Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    Policy(2,:,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    % Valt at exponential discounter optimum (full max over d2 and aprime)
    [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policyalt(1,:,:,:,N_j)=shiftdim(maxindexalt_d2,-1);
    maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_e,1]);
    Policyalt(2,:,:,:,N_j)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindexalt_lin-1)),[1,N_a,N_semiz,N_e]);
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

    EV_pre=sum(Valt(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, n_semiz, n_e, d2_val, a_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

            EV_d2=EV_pre.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireRHS_V=ReturnMatrix_d2+beta*EV_d2;
            [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_V_ford2_jj(:,:,:,d2_c)=shiftdim(maxindexalt,1);

            entireRHS=ReturnMatrix_d2+beta0beta*EV_d2;
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            EV_d2=EV_pre.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d2e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, n_semiz, special_n_e, d2_val, a_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0);

                entireRHS_V=ReturnMatrix_d2e+beta*EV_d2;
                [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_V_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindexalt,1);

                entireRHS_e=ReturnMatrix_d2e+beta0beta*EV_d2;
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                Vtilde_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);
            end
        end
    end
    [Vtilde_jj,maxindex]=max(Vtilde_ford2_jj,[],4);
    Vtilde(:,:,:,jj)=Vtilde_jj;
    Policy(1,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    Policy(2,:,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    % Valt at exponential discounter optimum (full max over d2 and aprime)
    [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],4);
    Valt(:,:,:,jj)=V_jj;
    Policyalt(1,:,:,:,jj)=shiftdim(maxindexalt_d2,-1);
    maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_e,1]);
    Policyalt(2,:,:,:,jj)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindexalt_lin-1)),[1,N_a,N_semiz,N_e]);
end

end
