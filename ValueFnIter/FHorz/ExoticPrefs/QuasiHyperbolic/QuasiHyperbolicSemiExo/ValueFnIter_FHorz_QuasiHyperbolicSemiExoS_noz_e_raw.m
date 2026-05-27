function [Vunderbar, Policy3, Vhat]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_noz_e_raw(n_d1, n_d2, n_a, n_semiz, n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic with semi-exogenous shock and iid e shock, no z.

n_d=[n_d1,n_d2];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod([n_d1,n_d2]);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

Vhat=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

special_n_d=[n_d1,ones(1,length(n_d2))];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states');
end

Vhat_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Vunderbar_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2);

%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_semiz, n_e, d_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,:,N_j)=Vtemp;
        d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        Policy3(1,:,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy3(2,:,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy3(3,:,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_semiz, special_n_e, d_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Vhat(:,:,e_c,N_j)=Vtemp;
            d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
            Policy3(1,:,:,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy3(2,:,:,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy3(3,:,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        end
    end
    Vunderbar=Vhat;
else
    EV_pre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*pi_e_J(1,1,:,N_j),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_semiz, n_e, d12c_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

            EV_d2=EV_pre.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireEV=repelem(EV_d2,N_d1,1,1);

            entireRHS=ReturnMatrix_d2+beta0beta*entireEV;
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vhat_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            entireRHS_alt=ReturnMatrix_d2+beta*entireEV;
            maxindexfull=maxindex+N_d1*N_a*(0:1:N_a-1)+shiftdim(N_d1*N_a*N_a*(0:1:N_semiz-1),-1)+shiftdim(N_d1*N_a*N_a*N_semiz*(0:1:N_e-1),-2);
            Vunderbar_ford2_jj(:,:,:,d2_c)=shiftdim(entireRHS_alt(maxindexfull),1);
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            EV_d2=EV_pre.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireEV=repelem(EV_d2,N_d1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_semiz, special_n_e, d12c_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);

                entireRHS_e=ReturnMatrix_d2e+beta0beta*entireEV;
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                Vhat_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                entireRHS_alt=ReturnMatrix_d2e+beta*entireEV;
                maxindexfull=maxindex+N_d1*N_a*(0:1:N_a-1)+shiftdim(N_d1*N_a*N_a*(0:1:N_semiz-1),-1);
                Vunderbar_ford2_jj(:,:,e_c,d2_c)=shiftdim(entireRHS_alt(maxindexfull),1);
            end
        end
    end
    [Vhat_jj,maxindex]=max(Vhat_ford2_jj,[],4);
    Vhat(:,:,:,N_j)=Vhat_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_lin-1)),[N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
    Vunderbar(:,:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_lin-1)),[N_a,N_semiz,N_e]);
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

    EV_pre=sum(Vunderbar(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_semiz, n_e, d12c_gridvals, a_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

            EV_d2=EV_pre.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireEV=repelem(EV_d2,N_d1,1,1);

            entireRHS=ReturnMatrix_d2+beta0beta*entireEV;
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vhat_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            entireRHS_alt=ReturnMatrix_d2+beta*entireEV;
            maxindexfull=maxindex+N_d1*N_a*(0:1:N_a-1)+shiftdim(N_d1*N_a*N_a*(0:1:N_semiz-1),-1)+shiftdim(N_d1*N_a*N_a*N_semiz*(0:1:N_e-1),-2);
            Vunderbar_ford2_jj(:,:,:,d2_c)=shiftdim(entireRHS_alt(maxindexfull),1);
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            EV_d2=EV_pre.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireEV=repelem(EV_d2,N_d1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d2e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_semiz, special_n_e, d12c_gridvals, a_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0);

                entireRHS_e=ReturnMatrix_d2e+beta0beta*entireEV;
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                Vhat_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                entireRHS_alt=ReturnMatrix_d2e+beta*entireEV;
                maxindexfull=maxindex+N_d1*N_a*(0:1:N_a-1)+shiftdim(N_d1*N_a*N_a*(0:1:N_semiz-1),-1);
                Vunderbar_ford2_jj(:,:,e_c,d2_c)=shiftdim(entireRHS_alt(maxindexfull),1);
            end
        end
    end
    [Vhat_jj,maxindex]=max(Vhat_ford2_jj,[],4);
    Vhat(:,:,:,jj)=Vhat_jj;
    Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_lin-1)),[N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
    Vunderbar(:,:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_lin-1)),[N_a,N_semiz,N_e]);
end

end
