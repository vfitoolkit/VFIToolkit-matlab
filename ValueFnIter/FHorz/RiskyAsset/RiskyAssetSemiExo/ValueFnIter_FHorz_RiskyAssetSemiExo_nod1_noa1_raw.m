function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_nod1_noa1_raw(n_d2,n_d3,n_d4,n_a,n_semiz,n_z,n_u,N_j, d2_grid, d3_grid, d4_grid, a_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% noa1: only a2

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_u=prod(n_u);

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Policy=zeros(3,N_a,N_semiz*N_z,N_j,'gpuArray'); % d2, d3, d4

%%
d23_grid=gpuArray(d23_grid);
a_grid=gpuArray(a_grid);
u_grid=gpuArray(u_grid);

d3d4_gridvals=gpuArray(CreateGridvals([n_d3,n_d4],[d3_grid;d4_grid],1));
a_gridvals=gpuArray(CreateGridvals(n_a,a_grid,1));

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

bothzind=shiftdim(0:1:N_bothz-1,-1);

V_ford4_jj=zeros(N_a,N_semiz*N_z,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_semiz*N_z,N_d4,'gpuArray');
d2index_ford4_jj=zeros(N_d3,N_semiz*N_z,N_d4,'gpuArray');

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_d4], n_a, n_bothz, d3d4_gridvals, a_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(1,:,:,N_j)=1;
        Policy(2,:,:,N_j)=rem(maxindex-1,N_d3)+1;
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),-1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_d4], n_a, special_n_bothz, d3d4_gridvals, a_gridvals, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(1,:,z_c,N_j)=1;
            Policy(2,:,z_c,N_j)=rem(maxindex-1,N_d3)+1;
            Policy(3,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
        end
    end
else
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [aprimeIndex,aprimeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a, n_u, d23_grid, a_grid, u_grid, aprimeFnParamsVec,1);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz(:,:,d4_c));
            d3_special_d4_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4], [d3_grid; d4_gridvals(d4_c,:)'], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,special_n_d4], n_a, n_bothz, d3_special_d4_gridvals, a_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=EV(aprimeIndex+N_a*((1:1:N_bothz)-1));
            EV2=EV((aprimeIndex+1)+N_a*((1:1:N_bothz)-1));

            EV1=reshape(EV1,[N_d23,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23,N_u,N_bothz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3,1,N_bothz]),[],1);
            entireRHS=ReturnMatrix_d4+shiftdim(DiscountFactorParamsVec*EV_onlyd3,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,d4_c)=shiftdim(maxindex,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],3);
        V(:,:,N_j)=V_jj;
        Policy(3,:,:,N_j)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_bothz,1]);
        d3_ind=reshape(Policy_ford4_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex_d4-1)),[1,N_a,N_bothz]);
        Policy(2,:,:,N_j)=d3_ind;
        Policy(1,:,:,N_j)=shiftdim(d2index_ford4_jj(d3_ind+N_d3*bothzind+N_d3*N_bothz*shiftdim(maxindex-1,-1)),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz(:,:,d4_c));
            d3_special_d4_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4], [d3_grid; d4_gridvals(d4_c,:)'], 1));
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d4z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,special_n_d4], n_a, special_n_bothz, d3_special_d4_gridvals, a_gridvals, z_val, ReturnFnParamsVec);

                EV_z=V_Jplus1.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23,N_u]);
                EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeIndex+1),[N_d23,N_u]);

                EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2);

                [EV_onlyd3,d2index]=max(reshape(EV_z,[N_d2,N_d3,1]),[],1);
                entireRHS_d4z=ReturnMatrix_d4z+shiftdim(DiscountFactorParamsVec*EV_onlyd3,1);

                [Vtemp,maxindex]=max(entireRHS_d4z,[],1);

                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(maxindex,1);
                d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2index,1);
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],3);
        V(:,:,N_j)=V_jj;
        Policy(3,:,:,N_j)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_bothz,1]);
        d3_ind=reshape(Policy_ford4_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex_d4-1)),[1,N_a,N_bothz]);
        Policy(2,:,:,N_j)=d3_ind;
        Policy(1,:,:,N_j)=shiftdim(d2index_ford4_jj(d3_ind+N_d3*bothzind+N_d3*N_bothz*shiftdim(maxindex-1,-1)),-1);
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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndex,aprimeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a, n_u, d23_grid, a_grid, u_grid, aprimeFnParamsVec,1);

    EV=V(:,:,jj+1);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz(:,:,d4_c));
            d3_special_d4_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4], [d3_grid; d4_gridvals(d4_c,:)'], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,special_n_d4], n_a, n_bothz, d3_special_d4_gridvals, a_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec);

            EV=EV.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=EV(aprimeIndex+N_a*((1:1:N_bothz)-1));
            EV2=EV((aprimeIndex+1)+N_a*((1:1:N_bothz)-1));

            EV1=reshape(EV1,[N_d23,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23,N_u,N_bothz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3,1,N_bothz]),[],1);
            entireRHS=ReturnMatrix_d4+shiftdim(DiscountFactorParamsVec*EV_onlyd3,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,d4_c)=shiftdim(maxindex,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],3);
        V(:,:,jj)=V_jj;
        Policy(3,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_bothz,1]);
        d3_ind=reshape(Policy_ford4_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex_d4-1)),[1,N_a,N_bothz]);
        Policy(2,:,:,jj)=d3_ind;
        Policy(1,:,:,jj)=shiftdim(d2index_ford4_jj(d3_ind+N_d3*bothzind+N_d3*N_bothz*shiftdim(maxindex-1,-1)),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz(:,:,d4_c));
            d3_special_d4_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4], [d3_grid; d4_gridvals(d4_c,:)'], 1));
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d4z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,special_n_d4], n_a, special_n_bothz, d3_special_d4_gridvals, a_gridvals, z_val, ReturnFnParamsVec);

                EV_z=EV.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23,N_u]);
                EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeIndex+1),[N_d23,N_u]);

                EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2);

                [EV_onlyd3,d2index]=max(reshape(EV_z,[N_d2,N_d3,1]),[],1);
                entireRHS_d4z=ReturnMatrix_d4z+shiftdim(DiscountFactorParamsVec*EV_onlyd3,1);

                [Vtemp,maxindex]=max(entireRHS_d4z,[],1);

                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(maxindex,1);
                d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2index,1);
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],3);
        V(:,:,jj)=V_jj;
        Policy(3,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_bothz,1]);
        d3_ind=reshape(Policy_ford4_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex_d4-1)),[1,N_a,N_bothz]);
        Policy(2,:,:,jj)=d3_ind;
        Policy(1,:,:,jj)=shiftdim(d2index_ford4_jj(d3_ind+N_d3*bothzind+N_d3*N_bothz*shiftdim(maxindex-1,-1)),-1);
    end
end



end
