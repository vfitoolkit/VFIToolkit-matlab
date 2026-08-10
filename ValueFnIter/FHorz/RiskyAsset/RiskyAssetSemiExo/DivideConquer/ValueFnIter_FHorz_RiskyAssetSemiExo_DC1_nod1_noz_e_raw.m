function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_nod1_noz_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_e,n_u,N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% No d1, no z. e is iid
% Policy output has the choices on the first dimension: (d2,d3,d4,a1prime).

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);
N_u=prod(n_u);

N_d2d3=N_d2*N_d3;

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray'); % d2, d3, d4, a1prime

%%
u_grid=gpuArray(u_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
d23_grid=gpuArray(d23_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d3_gridvals=gpuArray(CreateGridvals(n_d3,d3_grid,1));
d3d4a1_gridvals=gpuArray(CreateGridvals([n_d3,n_d4,n_a1],[d3_grid;d4_grid;a1_grid],1));
a1a2_gridvals=gpuArray(CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1));

pi_u_col=pi_u(:);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2Bind=gpuArray(0:1:N_a2-1);
zBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
d3ind=(1:1:N_d3)';

% Accumulators for the choice of d4 (max is taken across d4 at the end of each period)
V_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
d2_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
d3_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
a1prime_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, n_e, d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        dindex=rem(maxindex-1,N_d3*N_d4)+1;
        Policy(1,:,:,:,N_j)=1; % d2 is meaningless in the terminal period
        Policy(2,:,:,:,N_j)=rem(dindex-1,N_d3)+1; % d3
        Policy(3,:,:,:,N_j)=ceil(dindex/N_d3); % d4
        Policy(4,:,:,:,N_j)=ceil(maxindex/(N_d3*N_d4)); % a1prime
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, special_n_e, d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            dindex=rem(maxindex-1,N_d3*N_d4)+1;
            Policy(1,:,:,e_c,N_j)=1; % d2 is meaningless in the terminal period
            Policy(2,:,:,e_c,N_j)=rem(dindex-1,N_d3)+1; % d3
            Policy(3,:,:,e_c,N_j)=ceil(dindex/N_d3); % d4
            Policy(4,:,:,e_c,N_j)=ceil(maxindex/(N_d3*N_d4)); % a1prime
        end
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,n_d4,n_a1], [n_a1,n_a2], special_n_semiz, special_n_e, d3d4a1_gridvals, a1a2_gridvals, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtemp,1);
                dindex=rem(maxindex-1,N_d3*N_d4)+1;
                Policy(1,:,z_c,e_c,N_j)=1; % d2 is meaningless in the terminal period
                Policy(2,:,z_c,e_c,N_j)=rem(dindex-1,N_d3)+1; % d3
                Policy(3,:,z_c,e_c,N_j)=ceil(dindex/N_d3); % d4
                Policy(4,:,z_c,e_c,N_j)=ceil(maxindex/(N_d3*N_d4)); % a1prime
            end
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]);
    EVnext=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j+1),-2),3); % [N_a,N_semiz]
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c); % no kron in noz case
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

            EV=EVnext.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a1,N_semiz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a1,N_semiz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_semiz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_semiz]);

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d3_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii,[N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);
            DEV=reshape(DiscountedEV,[N_d3,N_a1,1,1,N_semiz,1]);
            entireRHS_ii=RM+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);

            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d3*N_a1,vfoptions.level1n*N_a2,N_semiz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford4_jj(curraindex,:,:,d4_c)=shiftdim(Vtempii,1);
            pol_d3_a1=shiftdim(maxindex2,1);
            d3part=rem(pol_d3_a1-1,N_d3)+1;
            a1primepart=ceil(pol_d3_a1/N_d3);
            [npts,nz,ne]=size(pol_d3_a1);
            zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
            d3_ford4_jj(curraindex,:,:,d4_c)=d3part;
            a1prime_ford4_jj(curraindex,:,:,d4_c)=a1primepart;
            d2_ford4_jj(curraindex,:,:,d4_c)=d2index_resh(lin);

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d3_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind+N_d3*N_a2*N_semiz*eBind;
                    pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                    pol_d3_a1=shiftdim(pol_d3_a1,1);
                    d3part=rem(pol_d3_a1-1,N_d3)+1;
                    a1primepart=ceil(pol_d3_a1/N_d3);
                    [npts,nz,ne]=size(pol_d3_a1);
                    zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d3_ford4_jj(curraindex,:,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,:,d4_c)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,n_semiz,n_e, d3_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3,level1iidiff(ii)*N_a2,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind+N_d3*N_a2*N_semiz*eBind;
                    pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                    pol_d3_a1=shiftdim(pol_d3_a1,1);
                    d3part=rem(pol_d3_a1-1,N_d3)+1;
                    a1primepart=ceil(pol_d3_a1/N_d3);
                    [npts,nz,ne]=size(pol_d3_a1);
                    zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d3_ford4_jj(curraindex,:,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,:,d4_c)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory>=1
        % Loop over e inside d4
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

            EV=EVnext.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a1,N_semiz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a1,N_semiz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_semiz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_semiz]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,ones(1,length(n_e)), d3_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii_e,[N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                DEV=reshape(DiscountedEV,[N_d3,N_a1,1,1,N_semiz]);
                entireRHS_ii_e=RM+DEV;
                entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);

                [~,maxindex1]=max(entireRHS_ii_e,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d3*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford4_jj(curraindex,:,e_c,d4_c)=shiftdim(Vtempii,1);
                pol_d3_a1=shiftdim(maxindex2,1);
                d3part=rem(pol_d3_a1-1,N_d3)+1;
                a1primepart=ceil(pol_d3_a1/N_d3);
                [npts,nz]=size(pol_d3_a1);
                zidx=repmat(gpuArray(1:nz),npts,1);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                d3_ford4_jj(curraindex,:,e_c,d4_c)=d3part;
                a1prime_ford4_jj(curraindex,:,e_c,d4_c)=a1primepart;
                d2_ford4_jj(curraindex,:,e_c,d4_c)=d2index_resh(lin);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,ones(1,length(n_e)), d3_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                        entireRHS_ii_e=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                        V_ford4_jj(curraindex,:,e_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d3)+1);
                        allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                        pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                        pol_d3_a1=shiftdim(pol_d3_a1,1);
                        d3part=rem(pol_d3_a1-1,N_d3)+1;
                        a1primepart=ceil(pol_d3_a1/N_d3);
                        [npts,nz]=size(pol_d3_a1);
                        zidx=repmat(gpuArray(1:nz),npts,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                        d3_ford4_jj(curraindex,:,e_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,:,e_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,:,e_c,d4_c)=d2index_resh(lin);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,n_semiz,ones(1,length(n_e)), d3_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                        entireRHS_ii_e=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3,level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                        V_ford4_jj(curraindex,:,e_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d3)+1);
                        allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                        pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                        pol_d3_a1=shiftdim(pol_d3_a1,1);
                        d3part=rem(pol_d3_a1-1,N_d3)+1;
                        a1primepart=ceil(pol_d3_a1/N_d3);
                        [npts,nz]=size(pol_d3_a1);
                        zidx=repmat(gpuArray(1:nz),npts,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                        d3_ford4_jj(curraindex,:,e_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,:,e_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,:,e_c,d4_c)=d2index_resh(lin);
                    end
                end
            end
        end
    end

    % Cross-d4 max (max over dim 4 since shape is [N_a,N_semiz,N_e,N_d4])
    [Vbest,d4winner]=max(V_ford4_jj,[],4);
    V(:,:,:,N_j)=Vbest;
    N=N_a*N_semiz*N_e;
    linidx_d4=(1:1:N)'+N*(reshape(d4winner,[N,1])-1);
    Policy(1,:,:,:,N_j)=reshape(d2_ford4_jj(linidx_d4),[1,N_a,N_semiz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(d3_ford4_jj(linidx_d4),[1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(d4winner,[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(a1prime_ford4_jj(linidx_d4),[1,N_a,N_semiz,N_e]);
end


%% Iterate backwards
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

    EVnext=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3); % [N_a,N_semiz]

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c); % no kron in noz case
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

            EV=EVnext.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a1,N_semiz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a1,N_semiz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_semiz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_semiz]);

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d3_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii,[N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);
            DEV=reshape(DiscountedEV,[N_d3,N_a1,1,1,N_semiz,1]);
            entireRHS_ii=RM+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);

            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d3*N_a1,vfoptions.level1n*N_a2,N_semiz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford4_jj(curraindex,:,:,d4_c)=shiftdim(Vtempii,1);
            pol_d3_a1=shiftdim(maxindex2,1);
            d3part=rem(pol_d3_a1-1,N_d3)+1;
            a1primepart=ceil(pol_d3_a1/N_d3);
            [npts,nz,ne]=size(pol_d3_a1);
            zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
            d3_ford4_jj(curraindex,:,:,d4_c)=d3part;
            a1prime_ford4_jj(curraindex,:,:,d4_c)=a1primepart;
            d2_ford4_jj(curraindex,:,:,d4_c)=d2index_resh(lin);

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d3_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind+N_d3*N_a2*N_semiz*eBind;
                    pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                    pol_d3_a1=shiftdim(pol_d3_a1,1);
                    d3part=rem(pol_d3_a1-1,N_d3)+1;
                    a1primepart=ceil(pol_d3_a1/N_d3);
                    [npts,nz,ne]=size(pol_d3_a1);
                    zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d3_ford4_jj(curraindex,:,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,:,d4_c)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,n_semiz,n_e, d3_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3,level1iidiff(ii)*N_a2,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind+N_d3*N_a2*N_semiz*eBind;
                    pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                    pol_d3_a1=shiftdim(pol_d3_a1,1);
                    d3part=rem(pol_d3_a1-1,N_d3)+1;
                    a1primepart=ceil(pol_d3_a1/N_d3);
                    [npts,nz,ne]=size(pol_d3_a1);
                    zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d3_ford4_jj(curraindex,:,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,:,d4_c)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory>=1
        % Loop over e inside d4
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

            EV=EVnext.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a1,N_semiz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a1,N_semiz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_semiz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_semiz]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,ones(1,length(n_e)), d3_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii_e,[N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                DEV=reshape(DiscountedEV,[N_d3,N_a1,1,1,N_semiz]);
                entireRHS_ii_e=RM+DEV;
                entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);

                [~,maxindex1]=max(entireRHS_ii_e,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d3*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford4_jj(curraindex,:,e_c,d4_c)=shiftdim(Vtempii,1);
                pol_d3_a1=shiftdim(maxindex2,1);
                d3part=rem(pol_d3_a1-1,N_d3)+1;
                a1primepart=ceil(pol_d3_a1/N_d3);
                [npts,nz]=size(pol_d3_a1);
                zidx=repmat(gpuArray(1:nz),npts,1);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                d3_ford4_jj(curraindex,:,e_c,d4_c)=d3part;
                a1prime_ford4_jj(curraindex,:,e_c,d4_c)=a1primepart;
                d2_ford4_jj(curraindex,:,e_c,d4_c)=d2index_resh(lin);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,ones(1,length(n_e)), d3_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                        entireRHS_ii_e=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                        V_ford4_jj(curraindex,:,e_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d3)+1);
                        allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                        pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                        pol_d3_a1=shiftdim(pol_d3_a1,1);
                        d3part=rem(pol_d3_a1-1,N_d3)+1;
                        a1primepart=ceil(pol_d3_a1/N_d3);
                        [npts,nz]=size(pol_d3_a1);
                        zidx=repmat(gpuArray(1:nz),npts,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                        d3_ford4_jj(curraindex,:,e_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,:,e_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,:,e_c,d4_c)=d2index_resh(lin);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,n_semiz,ones(1,length(n_e)), d3_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                        entireRHS_ii_e=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3,level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                        V_ford4_jj(curraindex,:,e_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d3)+1);
                        allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                        pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                        pol_d3_a1=shiftdim(pol_d3_a1,1);
                        d3part=rem(pol_d3_a1-1,N_d3)+1;
                        a1primepart=ceil(pol_d3_a1/N_d3);
                        [npts,nz]=size(pol_d3_a1);
                        zidx=repmat(gpuArray(1:nz),npts,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                        d3_ford4_jj(curraindex,:,e_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,:,e_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,:,e_c,d4_c)=d2index_resh(lin);
                    end
                end
            end
        end
    end

    % Cross-d4 max (max over dim 4 since shape is [N_a,N_semiz,N_e,N_d4])
    [Vbest,d4winner]=max(V_ford4_jj,[],4);
    V(:,:,:,jj)=Vbest;
    N=N_a*N_semiz*N_e;
    linidx_d4=(1:1:N)'+N*(reshape(d4winner,[N,1])-1);
    Policy(1,:,:,:,jj)=reshape(d2_ford4_jj(linidx_d4),[1,N_a,N_semiz,N_e]);
    Policy(2,:,:,:,jj)=reshape(d3_ford4_jj(linidx_d4),[1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,jj)=reshape(d4winner,[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,jj)=reshape(a1prime_ford4_jj(linidx_d4),[1,N_a,N_semiz,N_e]);
end


end
