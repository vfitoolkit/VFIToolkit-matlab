function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_nod1_noz_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_e,n_u,N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% no z; e iid

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_semiz=prod(n_semiz);
N_e=prod(n_e);
N_u=prod(n_u);

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

N_a=N_a1*N_a2;

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray'); % (d2,d3,d4,a1prime)

%%
d23_grid=gpuArray(d23_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
u_grid=gpuArray(u_grid);

d3d4a1_gridvals=gpuArray(CreateGridvals([n_d3,n_d4,n_a1],[d3_grid;d4_grid;a1_grid],1));
a1a2_gridvals=gpuArray(CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1));

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
end

semizind=shiftdim(0:1:N_semiz-1,-1);

V_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
d2index_ford4_jj=zeros(N_d3*N_a1,N_semiz,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, n_e, d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        dindex=rem(maxindex-1,N_d3*N_d4)+1;
        Policy(1,:,:,:,N_j)=1;
        Policy(2,:,:,:,N_j)=rem(dindex-1,N_d3)+1;
        Policy(3,:,:,:,N_j)=shiftdim(ceil(dindex/N_d3),-1);
        Policy(4,:,:,:,N_j)=shiftdim(ceil(maxindex/(N_d3*N_d4)),-1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, special_n_e, d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            dindex=rem(maxindex-1,N_d3*N_d4)+1;
            Policy(1,:,:,e_c,N_j)=1;
            Policy(2,:,:,e_c,N_j)=rem(dindex-1,N_d3)+1;
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(dindex/N_d3),-1);
            Policy(4,:,:,e_c,N_j)=shiftdim(ceil(maxindex/(N_d3*N_d4)),-1);
        end

    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,n_d4,n_a1], [n_a1,n_a2], special_n_semiz, special_n_e, d3d4a1_gridvals, a1a2_gridvals, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                dindex=rem(maxindex-1,N_d3*N_d4)+1;
                Policy(1,:,z_c,e_c,N_j)=1;
                Policy(2,:,z_c,e_c,N_j)=rem(dindex-1,N_d3)+1;
                Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(dindex/N_d3),-1);
                Policy(4,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/(N_d3*N_d4)),-1);
            end
        end
    end
else
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, [n_d23,n_a1], n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    EVpre=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j),-2),3);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4,n_a1], [d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_semiz, n_e, d3_special_d4_a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            EV=EVpre.*shiftdim(pi_semizd4',-1);
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

            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);
            entireRHS=ReturnMatrix_d4+shiftdim(DiscountFactorParamsVec*EV_onlyd3,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(maxindex,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,N_j)=V_jj;
        Policy(3,:,:,:,N_j)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_d4-1)),[1,N_a,N_semiz,N_e]);
        Policy(1,:,:,:,N_j)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        Policy(2,:,:,:,N_j)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(4,:,:,:,N_j)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4,n_a1], [d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));

            EV=EVpre.*shiftdim(pi_semizd4',-1);
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

            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);
            DiscountedEV_onlyd3=shiftdim(DiscountFactorParamsVec*EV_onlyd3,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d4e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_semiz, special_n_e, d3_special_d4_a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
                entireRHS_e=ReturnMatrix_d4e+DiscountedEV_onlyd3;
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(maxindex,1);
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,N_j)=V_jj;
        Policy(3,:,:,:,N_j)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_d4-1)),[1,N_a,N_semiz,N_e]);
        Policy(1,:,:,:,N_j)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        Policy(2,:,:,:,N_j)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(4,:,:,:,N_j)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
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

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4,n_a1], [d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_semiz, n_e, d3_special_d4_a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            EV=EVpre.*shiftdim(pi_semizd4',-1);
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

            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);
            entireRHS=ReturnMatrix_d4+shiftdim(DiscountFactorParamsVec*EV_onlyd3,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(maxindex,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,jj)=V_jj;
        Policy(3,:,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_d4-1)),[1,N_a,N_semiz,N_e]);
        Policy(1,:,:,:,jj)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        Policy(2,:,:,:,jj)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(4,:,:,:,jj)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d3,special_n_d4,n_a1], [d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));

            EV=EVpre.*shiftdim(pi_semizd4',-1);
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

            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_semiz]),[],1);
            DiscountedEV_onlyd3=shiftdim(DiscountFactorParamsVec*EV_onlyd3,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d4e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_semiz, special_n_e, d3_special_d4_a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
                entireRHS_e=ReturnMatrix_d4e+DiscountedEV_onlyd3;
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(maxindex,1);
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,jj)=V_jj;
        Policy(3,:,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_semiz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex_d4-1)),[1,N_a,N_semiz,N_e]);
        Policy(1,:,:,:,jj)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*semizind+N_d3*N_a1*N_semiz*shiftdim(maxindex-1,-1)),-1);
        Policy(2,:,:,:,jj)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy(4,:,:,:,jj)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
    end
end



end
