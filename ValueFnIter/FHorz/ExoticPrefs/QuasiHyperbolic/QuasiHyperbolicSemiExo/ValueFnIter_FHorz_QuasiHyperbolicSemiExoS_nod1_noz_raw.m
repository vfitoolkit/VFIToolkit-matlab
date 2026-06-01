function [Vhat, Policy, Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_nod1_noz_raw(n_d2, n_a, n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic with semi-exogenous shock, no d1, no z.

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);

Vhat=zeros(N_a,N_semiz,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy=zeros(2,N_a,N_semiz,N_j,'gpuArray'); % d2, aprime

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

Vhat_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Vunderbar_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');

%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d2, n_a, n_semiz, d2_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,N_j)=Vtemp;
        Policy(1,:,:,N_j)=shiftdim(rem(maxindex-1,N_d2)+1,-1);
        Policy(2,:,:,N_j)=shiftdim(ceil(maxindex/N_d2),-1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d2, n_a, special_n_semiz, d2_gridvals, a_grid, z_val, ReturnFnParamsVec,0);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Vhat(:,z_c,N_j)=Vtemp;
            Policy(1,:,z_c,N_j)=shiftdim(rem(maxindex-1,N_d2)+1,-1);
            Policy(2,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d2),-1);
        end
    end
    Vunderbar=Vhat;
else
    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, n_semiz, d2_val, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireRHS=ReturnMatrix_d2+beta0beta*EV_d2;
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vhat_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            entireRHS_alt=ReturnMatrix_d2+beta*EV_d2;
            maxindexfull=maxindex+N_a*(0:1:N_a-1)+shiftdim(N_a*N_a*(0:1:N_semiz-1),-1);
            Vunderbar_ford2_jj(:,:,d2_c)=shiftdim(entireRHS_alt(maxindexfull),1);
        end
        [Vhat_jj,maxindex]=max(Vhat_ford2_jj,[],3);
        Vhat(:,:,N_j)=Vhat_jj;
        Policy(1,:,:,N_j)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
        Policy(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
        Vunderbar(:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[N_a,N_semiz]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, special_n_semiz, d2_val, a_grid, z_val, ReturnFnParamsVec,0);

                EV_d2z=EV.*pi_semiz(z_c,:);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                entireRHS_z=ReturnMatrix_d2z+beta0beta*EV_d2z;
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                Vhat_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;

                entireRHS_alt=ReturnMatrix_d2z+beta*EV_d2z;
                maxindexfull=maxindex+N_a*(0:1:N_a-1);
                Vunderbar_ford2_jj(:,z_c,d2_c)=entireRHS_alt(maxindexfull);
            end
        end
        [Vhat_jj,maxindex]=max(Vhat_ford2_jj,[],3);
        Vhat(:,:,N_j)=Vhat_jj;
        Policy(1,:,:,N_j)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
        Policy(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
        Vunderbar(:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[N_a,N_semiz]);
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

    EV=Vunderbar(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, n_semiz, d2_val, a_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireRHS=ReturnMatrix_d2+beta0beta*EV_d2;
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vhat_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            entireRHS_alt=ReturnMatrix_d2+beta*EV_d2;
            maxindexfull=maxindex+N_a*(0:1:N_a-1)+shiftdim(N_a*N_a*(0:1:N_semiz-1),-1);
            Vunderbar_ford2_jj(:,:,d2_c)=shiftdim(entireRHS_alt(maxindexfull),1);
        end
        [Vhat_jj,maxindex]=max(Vhat_ford2_jj,[],3);
        Vhat(:,:,jj)=Vhat_jj;
        Policy(1,:,:,jj)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
        Policy(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
        Vunderbar(:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[N_a,N_semiz]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, special_n_semiz, d2_val, a_grid, z_val, ReturnFnParamsVec,0);

                EV_d2z=EV.*pi_semiz(z_c,:);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                entireRHS_z=ReturnMatrix_d2z+beta0beta*EV_d2z;
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                Vhat_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;

                entireRHS_alt=ReturnMatrix_d2z+beta*EV_d2z;
                maxindexfull=maxindex+N_a*(0:1:N_a-1);
                Vunderbar_ford2_jj(:,z_c,d2_c)=entireRHS_alt(maxindexfull);
            end
        end
        [Vhat_jj,maxindex]=max(Vhat_ford2_jj,[],3);
        Vhat(:,:,jj)=Vhat_jj;
        Policy(1,:,:,jj)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
        Policy(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
        Vunderbar(:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[N_a,N_semiz]);
    end
end

end
