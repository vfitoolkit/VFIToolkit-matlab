function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% no z (only semiz)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_semiz=prod(n_semiz);
N_u=prod(n_u);

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

N_a=N_a1*N_a2;

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy=zeros(5,N_a,N_semiz,N_j,'gpuArray'); % d1, d2, d3, d4, a1prime

%%
d23_grid=gpuArray(d23_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
u_grid=gpuArray(u_grid);

d1d3d4a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,n_d4,n_a1],[d1_grid;d3_grid;d4_grid;a1_grid],1));
a1a2_gridvals=gpuArray(CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1));

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

aind=gpuArray(0:1:N_a-1);
semizind=shiftdim(0:1:N_semiz-1,-1);

% Preallocate
V_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
d1index_ford4_jj=zeros(N_d3*N_a1,N_a,N_semiz,N_d4,'gpuArray');
d2index_ford4_jj=zeros(N_d3*N_a1,N_semiz,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, d1d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
        d1d3_ind=rem(dindex-1,N_d1*N_d3)+1;
        Policy(1,:,:,N_j)=shiftdim(rem(d1d3_ind-1,N_d1)+1,-1);
        Policy(2,:,:,N_j)=1;
        Policy(3,:,:,N_j)=shiftdim(ceil(d1d3_ind/N_d1),-1);
        Policy(4,:,:,N_j)=shiftdim(ceil(dindex/(N_d1*N_d3)),-1);
        Policy(5,:,:,N_j)=shiftdim(ceil(maxindex/(N_d1*N_d3*N_d4)),-1);

    elseif vfoptions.lowmemory==1

        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], special_n_semiz, d1d3d4a1_gridvals, a1a2_gridvals, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
            d1d3_ind=rem(dindex-1,N_d1*N_d3)+1;
            Policy(1,:,z_c,N_j)=shiftdim(rem(d1d3_ind-1,N_d1)+1,-1);
            Policy(2,:,z_c,N_j)=1;
            Policy(3,:,z_c,N_j)=shiftdim(ceil(d1d3_ind/N_d1),-1);
            Policy(4,:,z_c,N_j)=shiftdim(ceil(dindex/(N_d1*N_d3)),-1);
            Policy(5,:,z_c,N_j)=shiftdim(ceil(maxindex/(N_d1*N_d3*N_d4)),-1);
        end
    end
else
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, [n_d23,n_a1], n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d1_d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,special_n_d4,n_a1], [d1_grid; d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_semiz, d1_d3_special_d4_a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            EV=V_Jplus1.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            skipinterp=logical(EV(aprimeIndex+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,1);
            aprimeProbs(skipinterp)=0;

            EV1=EV(aprimeIndex+N_a*((1:1:N_semiz)-1));
            EV2=EV((aprimeplus1Index)+N_a*((1:1:N_semiz)-1));

            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_d4,[N_d1,N_d3*N_a1,N_a,N_semiz]),[],1);
            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);

            entireRHS=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,d4_c)=shiftdim(maxindex,1);
            d1index_ford4_jj(:,:,:,d4_c)=shiftdim(d1index,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],3);
        V(:,:,N_j)=V_jj;
        Policy(4,:,:,N_j)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_d4-1)),[1,N_a,N_semiz]);
        Policy(3,:,:,N_j)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(5,:,:,N_j)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
        Policy(2,:,:,N_j)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        d1_lookup=d3a1prime_ind+N_d3*N_a1*shiftdim(aind,-1)+N_d3*N_a1*N_a*semizind+N_d3*N_a1*N_a*N_semiz*shiftdim(maxindex-1,-1);
        Policy(1,:,:,N_j)=shiftdim(d1index_ford4_jj(d1_lookup),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d1_d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,special_n_d4,n_a1], [d1_grid; d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d4z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,special_n_d4,n_a1], [n_a1,n_a2], special_n_semiz, d1_d3_special_d4_a1_gridvals, a1a2_gridvals, z_val, ReturnFnParamsVec);

                EV_z=V_Jplus1.*pi_semizd4(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                skipinterp=logical(EV_z(aprimeIndex)==EV_z(aprimeplus1Index));
                aprimeProbs=repmat(a2primeProbs,N_a1,1);
                aprimeProbs(skipinterp)=0;

                EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23*N_a1,N_u]);
                EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeplus1Index),[N_d23*N_a1,N_u]);

                EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2);

                [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_d4z,[N_d1,N_d3*N_a1,N_a]),[],1);
                [EV_onlyd3,d2index]=max(reshape(EV_z,[N_d2,N_d3*N_a1,1]),[],1);

                entireRHS_d4z=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);

                [Vtemp,maxindex]=max(entireRHS_d4z,[],1);

                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(maxindex,1);
                d1index_ford4_jj(:,:,z_c,d4_c)=shiftdim(d1index,1);
                d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2index,1);
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],3);
        V(:,:,N_j)=V_jj;
        Policy(4,:,:,N_j)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_d4-1)),[1,N_a,N_semiz]);
        Policy(3,:,:,N_j)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(5,:,:,N_j)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
        Policy(2,:,:,N_j)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        d1_lookup=d3a1prime_ind+N_d3*N_a1*shiftdim(aind,-1)+N_d3*N_a1*N_a*semizind+N_d3*N_a1*N_a*N_semiz*shiftdim(maxindex-1,-1);
        Policy(1,:,:,N_j)=shiftdim(d1index_ford4_jj(d1_lookup),-1);
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
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    EV=V(:,:,jj+1);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d1_d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,special_n_d4,n_a1], [d1_grid; d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_semiz, d1_d3_special_d4_a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec);

            EV=EV.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1));
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1));

            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_d4,[N_d1,N_d3*N_a1,N_a,N_semiz]),[],1);
            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);

            entireRHS=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,d4_c)=shiftdim(maxindex,1);
            d1index_ford4_jj(:,:,:,d4_c)=shiftdim(d1index,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],3);
        V(:,:,jj)=V_jj;
        Policy(4,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_d4-1)),[1,N_a,N_semiz]);
        Policy(3,:,:,jj)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(5,:,:,jj)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
        Policy(2,:,:,jj)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        d1_lookup=d3a1prime_ind+N_d3*N_a1*shiftdim(aind,-1)+N_d3*N_a1*N_a*semizind+N_d3*N_a1*N_a*N_semiz*shiftdim(maxindex-1,-1);
        Policy(1,:,:,jj)=shiftdim(d1index_ford4_jj(d1_lookup),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d1_d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,special_n_d4,n_a1], [d1_grid; d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d4z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,special_n_d4,n_a1], [n_a1,n_a2], special_n_semiz, d1_d3_special_d4_a1_gridvals, a1a2_gridvals, z_val, ReturnFnParamsVec);

                EV_z=EV.*pi_semizd4(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                skipinterp=logical(EV_z(aprimeIndex)==EV_z(aprimeplus1Index));
                aprimeProbs=repmat(a2primeProbs,N_a1,1);
                aprimeProbs(skipinterp)=0;

                EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23*N_a1,N_u]);
                EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeplus1Index),[N_d23*N_a1,N_u]);

                EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2);

                [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_d4z,[N_d1,N_d3*N_a1,N_a]),[],1);
                [EV_onlyd3,d2index]=max(reshape(EV_z,[N_d2,N_d3*N_a1,1]),[],1);

                entireRHS_d4z=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);

                [Vtemp,maxindex]=max(entireRHS_d4z,[],1);

                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(maxindex,1);
                d1index_ford4_jj(:,:,z_c,d4_c)=shiftdim(d1index,1);
                d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2index,1);
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],3);
        V(:,:,jj)=V_jj;
        Policy(4,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_d4-1)),[1,N_a,N_semiz]);
        Policy(3,:,:,jj)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(5,:,:,jj)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
        Policy(2,:,:,jj)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        d1_lookup=d3a1prime_ind+N_d3*N_a1*shiftdim(aind,-1)+N_d3*N_a1*N_a*semizind+N_d3*N_a1*N_a*N_semiz*shiftdim(maxindex-1,-1);
        Policy(1,:,:,jj)=shiftdim(d1index_ford4_jj(d1_lookup),-1);
    end
end



end
