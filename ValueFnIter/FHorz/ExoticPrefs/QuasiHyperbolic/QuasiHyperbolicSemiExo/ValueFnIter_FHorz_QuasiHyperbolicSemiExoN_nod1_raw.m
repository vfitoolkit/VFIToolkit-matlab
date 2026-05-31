function [Vtilde, Policy3, V, Policy3alt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_nod1_raw(n_d2, n_a, n_z, n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic with semi-exogenous shock, no d1.

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Policy3=zeros(2,N_a,N_semiz*N_z,N_j,'gpuArray'); % d2, aprime
Policy3alt=zeros(2,N_a,N_semiz*N_z,N_j,'gpuArray'); % exponential discounter optimal (d2, aprime)

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Vtilde_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d2, n_a, n_bothz, d2_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        Policy3(1,:,:,N_j)=shiftdim(d_ind,-1);
        Policy3(2,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d2, n_a, special_n_bothz, d2_gridvals, a_grid, z_val, ReturnFnParamsVec,0);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
            Policy3(1,:,z_c,N_j)=shiftdim(d_ind,-1);
            Policy3(2,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        end
    end
    Vtilde=V;
    % terminal: QH and exponential discounter coincide
    Policy3alt(:,:,:,N_j)=Policy3(:,:,:,N_j);
else
    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, n_bothz, d2_val, a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireRHS_V=ReturnMatrix_d2+beta*EV_d2;
            [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_V_ford2_jj(:,:,d2_c)=shiftdim(maxindexalt,1);

            entireRHS=ReturnMatrix_d2+beta0beta*EV_d2;
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);
        end
        [Vtilde_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
        Vtilde(:,:,N_j)=Vtilde_jj;
        Policy3(1,:,:,N_j)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
        Policy3(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
        % V at exponential discounter optimum (full max over d2 and aprime)
        [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],3);
        V(:,:,N_j)=V_jj;
        Policy3alt(1,:,:,N_j)=shiftdim(maxindexalt_d2,-1);
        maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z,1]);
        Policy3alt(2,:,:,N_j)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, special_n_bothz, d2_val, a_grid, z_val, ReturnFnParamsVec,0);

                EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                entireRHS_V=ReturnMatrix_d2z+beta*EV_d2z;
                [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
                V_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_V_ford2_jj(:,z_c,d2_c)=maxindexalt;

                entireRHS_z=ReturnMatrix_d2z+beta0beta*EV_d2z;
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                Vtilde_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;
            end
        end
        [Vtilde_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
        Vtilde(:,:,N_j)=Vtilde_jj;
        Policy3(1,:,:,N_j)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
        Policy3(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
        % V at exponential discounter optimum (full max over d2 and aprime)
        [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],3);
        V(:,:,N_j)=V_jj;
        Policy3alt(1,:,:,N_j)=shiftdim(maxindexalt_d2,-1);
        maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z,1]);
        Policy3alt(2,:,:,N_j)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);
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

    EV=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, n_bothz, d2_val, a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            entireRHS_V=ReturnMatrix_d2+beta*EV_d2;
            [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_V_ford2_jj(:,:,d2_c)=shiftdim(maxindexalt,1);

            entireRHS=ReturnMatrix_d2+beta0beta*EV_d2;
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);
        end
        [Vtilde_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
        Vtilde(:,:,jj)=Vtilde_jj;
        Policy3(1,:,:,jj)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
        Policy3(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
        % V at exponential discounter optimum (full max over d2 and aprime)
        [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],3);
        V(:,:,jj)=V_jj;
        Policy3alt(1,:,:,jj)=shiftdim(maxindexalt_d2,-1);
        maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z,1]);
        Policy3alt(2,:,:,jj)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, special_n_bothz, d2_val, a_grid, z_val, ReturnFnParamsVec,0);

                EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                entireRHS_V=ReturnMatrix_d2z+beta*EV_d2z;
                [Vtemp,maxindexalt]=max(entireRHS_V,[],1);
                V_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_V_ford2_jj(:,z_c,d2_c)=maxindexalt;

                entireRHS_z=ReturnMatrix_d2z+beta0beta*EV_d2z;
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                Vtilde_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;
            end
        end
        [Vtilde_jj,maxindex]=max(Vtilde_ford2_jj,[],3);
        Vtilde(:,:,jj)=Vtilde_jj;
        Policy3(1,:,:,jj)=shiftdim(maxindex,-1);
        maxindex_lin=reshape(maxindex,[N_a*N_semiz*N_z,1]);
        Policy3(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex_lin-1)),[1,N_a,N_semiz*N_z]);
        % V at exponential discounter optimum (full max over d2 and aprime)
        [V_jj,maxindexalt_d2]=max(V_ford2_jj,[],3);
        V(:,:,jj)=V_jj;
        Policy3alt(1,:,:,jj)=shiftdim(maxindexalt_d2,-1);
        maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z,1]);
        Policy3alt(2,:,:,jj)=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);
    end
end

end
