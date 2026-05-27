function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% e is iid

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);
N_u=prod(n_u);

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

N_a=N_a1*N_a2;

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy5=zeros(5,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
d23_grid=gpuArray(d23_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
u_grid=gpuArray(u_grid);

d1d3d4a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,n_d4,n_a1],[d1_grid;d3_grid;d4_grid;a1_grid],1));
a1a2_gridvals=gpuArray(CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1));

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

aind=gpuArray(0:1:N_a-1);
bothzind=shiftdim(0:1:N_bothz-1,-1);
eind=shiftdim(0:1:N_e-1,-2);

% Preallocate
V_ford4_jj=zeros(N_a,N_semiz*N_z,N_e,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_semiz*N_z,N_e,N_d4,'gpuArray');
d1index_ford4_jj=zeros(N_d3*N_a1,N_a,N_semiz*N_z,N_e,N_d4,'gpuArray');
d2index_ford4_jj=zeros(N_d3*N_a1,N_semiz*N_z,N_d4,'gpuArray');

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_bothz, n_e, d1d3d4a1_gridvals, a1a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
        d1d3_ind=rem(dindex-1,N_d1*N_d3)+1;
        Policy5(1,:,:,:,N_j)=shiftdim(rem(d1d3_ind-1,N_d1)+1,-1);
        Policy5(2,:,:,:,N_j)=1;
        Policy5(3,:,:,:,N_j)=shiftdim(ceil(d1d3_ind/N_d1),-1);
        Policy5(4,:,:,:,N_j)=shiftdim(ceil(dindex/(N_d1*N_d3)),-1);
        Policy5(5,:,:,:,N_j)=shiftdim(ceil(maxindex/(N_d1*N_d3*N_d4)),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_bothz, special_n_e, d1d3d4a1_gridvals, a1a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
            d1d3_ind=rem(dindex-1,N_d1*N_d3)+1;
            Policy5(1,:,:,e_c,N_j)=shiftdim(rem(d1d3_ind-1,N_d1)+1,-1);
            Policy5(2,:,:,e_c,N_j)=1;
            Policy5(3,:,:,e_c,N_j)=shiftdim(ceil(d1d3_ind/N_d1),-1);
            Policy5(4,:,:,e_c,N_j)=shiftdim(ceil(dindex/(N_d1*N_d3)),-1);
            Policy5(5,:,:,e_c,N_j)=shiftdim(ceil(maxindex/(N_d1*N_d3*N_d4)),-1);
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], special_n_bothz, special_n_e, d1d3d4a1_gridvals, a1a2_gridvals, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
                d1d3_ind=rem(dindex-1,N_d1*N_d3)+1;
                Policy5(1,:,z_c,e_c,N_j)=shiftdim(rem(d1d3_ind-1,N_d1)+1,-1);
                Policy5(2,:,z_c,e_c,N_j)=1;
                Policy5(3,:,z_c,e_c,N_j)=shiftdim(ceil(d1d3_ind/N_d1),-1);
                Policy5(4,:,z_c,e_c,N_j)=shiftdim(ceil(dindex/(N_d1*N_d3)),-1);
                Policy5(5,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/(N_d1*N_d3*N_d4)),-1);
            end
        end
    end
else
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, [n_d23,n_a1], n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Integrate over e' first (e is iid)
    EVpre=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j),-2),3); % [N_a,N_bothz]

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz(:,:,d4_c));
            d1_d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,special_n_d4,n_a1], [d1_grid; d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_bothz, n_e, d1_d3_special_d4_a1_gridvals, a1a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            skipinterp=logical(EV(aprimeIndex+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,1);
            aprimeProbs(skipinterp)=0;

            EV1=EV(aprimeIndex+N_a*((1:1:N_bothz)-1));
            EV2=EV((aprimeplus1Index)+N_a*((1:1:N_bothz)-1));

            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_bothz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_d4,[N_d1,N_d3*N_a1,N_a,N_bothz,N_e]),[],1);
            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_bothz,1]),[],1);

            entireRHS=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(maxindex,1);
            d1index_ford4_jj(:,:,:,:,d4_c)=shiftdim(d1index,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,N_j)=V_jj;
        Policy5(4,:,:,:,N_j)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_bothz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex_d4-1)),[1,N_a,N_bothz,N_e]);
        Policy5(3,:,:,:,N_j)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy5(5,:,:,:,N_j)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
        % d2 only depends on (d3a1prime, bothz, d4) (no e)
        Policy5(2,:,:,:,N_j)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*bothzind+N_d3*N_a1*N_bothz*shiftdim(maxindex-1,-1)),-1);
        % d1 depends on (d3a1prime, a, bothz, e, d4)
        d1_lookup=d3a1prime_ind+N_d3*N_a1*shiftdim(aind,-1)+N_d3*N_a1*N_a*bothzind+N_d3*N_a1*N_a*N_bothz*eind+N_d3*N_a1*N_a*N_bothz*N_e*shiftdim(maxindex-1,-1);
        Policy5(1,:,:,:,N_j)=shiftdim(d1index_ford4_jj(d1_lookup),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz(:,:,d4_c));
            d1_d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,special_n_d4,n_a1], [d1_grid; d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            skipinterp=logical(EV(aprimeIndex+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,1);
            aprimeProbs(skipinterp)=0;

            EV1=EV(aprimeIndex+N_a*((1:1:N_bothz)-1));
            EV2=EV((aprimeplus1Index)+N_a*((1:1:N_bothz)-1));

            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_bothz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_bothz]),[],1);
            DiscountedEV_onlyd3=DiscountFactorParamsVec*shiftdim(EV_onlyd3,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d4e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_bothz, special_n_e, d1_d3_special_d4_a1_gridvals, a1a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_d4e,[N_d1,N_d3*N_a1,N_a,N_bothz]),[],1);

                entireRHS_e=shiftdim(ReturnMatrix_onlyd3,1)+DiscountedEV_onlyd3;

                [Vtemp,maxindex]=max(entireRHS_e,[],1);

                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(maxindex,1);
                d1index_ford4_jj(:,:,:,e_c,d4_c)=shiftdim(d1index,1);
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,N_j)=V_jj;
        Policy5(4,:,:,:,N_j)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_bothz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex_d4-1)),[1,N_a,N_bothz,N_e]);
        Policy5(3,:,:,:,N_j)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy5(5,:,:,:,N_j)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
        Policy5(2,:,:,:,N_j)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*bothzind+N_d3*N_a1*N_bothz*shiftdim(maxindex-1,-1)),-1);
        d1_lookup=d3a1prime_ind+N_d3*N_a1*shiftdim(aind,-1)+N_d3*N_a1*N_a*bothzind+N_d3*N_a1*N_a*N_bothz*eind+N_d3*N_a1*N_a*N_bothz*N_e*shiftdim(maxindex-1,-1);
        Policy5(1,:,:,:,N_j)=shiftdim(d1index_ford4_jj(d1_lookup),-1);
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
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz(:,:,d4_c));
            d1_d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,special_n_d4,n_a1], [d1_grid; d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));
            ReturnMatrix_d4=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_bothz, n_e, d1_d3_special_d4_a1_gridvals, a1a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_bothz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_bothz]);

            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1));
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1));

            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_bothz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_d4,[N_d1,N_d3*N_a1,N_a,N_bothz,N_e]),[],1);
            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_bothz,1]),[],1);

            entireRHS=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtemp,1);
            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(maxindex,1);
            d1index_ford4_jj(:,:,:,:,d4_c)=shiftdim(d1index,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,jj)=V_jj;
        Policy5(4,:,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_bothz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex_d4-1)),[1,N_a,N_bothz,N_e]);
        Policy5(3,:,:,:,jj)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy5(5,:,:,:,jj)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
        Policy5(2,:,:,:,jj)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*bothzind+N_d3*N_a1*N_bothz*shiftdim(maxindex-1,-1)),-1);
        d1_lookup=d3a1prime_ind+N_d3*N_a1*shiftdim(aind,-1)+N_d3*N_a1*N_a*bothzind+N_d3*N_a1*N_a*N_bothz*eind+N_d3*N_a1*N_a*N_bothz*N_e*shiftdim(maxindex-1,-1);
        Policy5(1,:,:,:,jj)=shiftdim(d1index_ford4_jj(d1_lookup),-1);

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz(:,:,d4_c));
            d1_d3_special_d4_a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,special_n_d4,n_a1], [d1_grid; d3_grid; d4_gridvals(d4_c,:)'; a1_grid], 1));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            skipinterp=logical(EV(aprimeIndex+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,1);
            aprimeProbs(skipinterp)=0;

            EV1=EV(aprimeIndex+N_a*((1:1:N_bothz)-1));
            EV2=EV((aprimeplus1Index)+N_a*((1:1:N_bothz)-1));

            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_bothz]).*(1-aprimeProbs);

            EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2);

            [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_bothz]),[],1);
            DiscountedEV_onlyd3=DiscountFactorParamsVec*shiftdim(EV_onlyd3,1);
            d2index_ford4_jj(:,:,d4_c)=squeeze(d2index);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d4e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,special_n_d4,n_a1], [n_a1,n_a2], n_bothz, special_n_e, d1_d3_special_d4_a1_gridvals, a1a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_d4e,[N_d1,N_d3*N_a1,N_a,N_bothz]),[],1);

                entireRHS_e=shiftdim(ReturnMatrix_onlyd3,1)+DiscountedEV_onlyd3;

                [Vtemp,maxindex]=max(entireRHS_e,[],1);

                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtemp,1);
                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(maxindex,1);
                d1index_ford4_jj(:,:,:,e_c,d4_c)=shiftdim(d1index,1);
            end
        end

        [V_jj,maxindex]=max(V_ford4_jj,[],4);
        V(:,:,:,jj)=V_jj;
        Policy5(4,:,:,:,jj)=maxindex;
        maxindex_d4=reshape(maxindex,[N_a*N_bothz*N_e,1]);
        d3a1prime_ind=reshape(Policy_ford4_jj((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex_d4-1)),[1,N_a,N_bothz,N_e]);
        Policy5(3,:,:,:,jj)=shiftdim(rem(d3a1prime_ind-1,N_d3)+1,-1);
        Policy5(5,:,:,:,jj)=shiftdim(ceil(d3a1prime_ind/N_d3),-1);
        Policy5(2,:,:,:,jj)=shiftdim(d2index_ford4_jj(d3a1prime_ind+N_d3*N_a1*bothzind+N_d3*N_a1*N_bothz*shiftdim(maxindex-1,-1)),-1);
        d1_lookup=d3a1prime_ind+N_d3*N_a1*shiftdim(aind,-1)+N_d3*N_a1*N_a*bothzind+N_d3*N_a1*N_a*N_bothz*eind+N_d3*N_a1*N_a*N_bothz*N_e*shiftdim(maxindex-1,-1);
        Policy5(1,:,:,:,jj)=shiftdim(d1index_ford4_jj(d1_lookup),-1);
    end
end

Policy=Policy5(1,:,:,:,:)+N_d1*(Policy5(2,:,:,:,:)-1)+N_d1*N_d2*(Policy5(3,:,:,:,:)-1)+N_d1*N_d2*N_d3*(Policy5(4,:,:,:,:)-1)+N_d1*N_d2*N_d3*N_d4*(Policy5(5,:,:,:,:)-1);


end
