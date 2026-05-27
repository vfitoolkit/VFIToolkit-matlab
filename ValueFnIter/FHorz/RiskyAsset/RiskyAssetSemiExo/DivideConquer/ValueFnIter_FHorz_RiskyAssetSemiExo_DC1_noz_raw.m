function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% No z (only semiz)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_u=prod(n_u);

N_d13=N_d1*N_d3;
N_d1d2d3=N_d1*N_d2*N_d3;

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy=zeros(N_a,N_semiz,N_j,'gpuArray');

%%
u_grid=gpuArray(u_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
d23_grid=gpuArray(d23_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d13_gridvals=gpuArray(CreateGridvals([n_d1,n_d3],[d1_grid;d3_grid],1));
d1d3d4a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,n_d4,n_a1],[d1_grid;d3_grid;d4_grid;a1_grid],1));
a1a2_gridvals=gpuArray(CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1));

pi_u_col=pi_u(:);

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2Bind=gpuArray(0:1:N_a2-1);
zBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
d3ind=repelem((1:1:N_d3)',N_d1,1);

V_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, d1d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=shiftdim(Vtemp,1);
        dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
        d1d3_ind=rem(dindex-1,N_d13)+1;
        d1part=rem(d1d3_ind-1,N_d1)+1;
        d3part=ceil(d1d3_ind/N_d1);
        d4part=ceil(dindex/N_d13);
        a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
        Policy(:,:,N_j)=shiftdim(d1part+N_d1*(1-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(d4part-1)+N_d1*N_d2*N_d3*N_d4*(a1primepart-1),1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], special_n_semiz, d1d3d4a1_gridvals, a1a2_gridvals, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtemp,1);
            dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
            d1d3_ind=rem(dindex-1,N_d13)+1;
            d1part=rem(d1d3_ind-1,N_d1)+1;
            d3part=ceil(d1d3_ind/N_d1);
            d4part=ceil(dindex/N_d13);
            a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
            Policy(:,z_c,N_j)=shiftdim(d1part+N_d1*(1-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(d4part-1)+N_d1*N_d2*N_d3*N_d4*(a1primepart-1),1);
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    [V(:,:,N_j),Policy(:,:,N_j)]=internal_per_j(EVpre,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_d4,n_a1,n_a2,n_semiz,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d1d2d3,N_a1,N_a2,N_a,N_semiz,N_u,...
        d13_gridvals,d4_grid,d4_gridvals,special_n_d4,a1_gridvals,a2_gridvals,semiz_gridvals_J(:,:,N_j),pi_semiz,pi_u_col,...
        level1ii,level1iidiff,a2Bind,zBind,d3ind,V_ford4_jj,Policy_ford4_jj,vfoptions);
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

    EVnext=V(:,:,jj+1);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    [V(:,:,jj),Policy(:,:,jj)]=internal_per_j(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_d4,n_a1,n_a2,n_semiz,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d1d2d3,N_a1,N_a2,N_a,N_semiz,N_u,...
        d13_gridvals,d4_grid,d4_gridvals,special_n_d4,a1_gridvals,a2_gridvals,semiz_gridvals_J(:,:,jj),pi_semiz,pi_u_col,...
        level1ii,level1iidiff,a2Bind,zBind,d3ind,V_ford4_jj,Policy_ford4_jj,vfoptions);
end


end


%% Per-period inner (no z)
function [V_jj,Policy_jj]=internal_per_j(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d1,n_d3,n_d4,n_a1,n_a2,n_semiz,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d1d2d3,N_a1,N_a2,N_a,N_semiz_count,N_u,...
    d13_gridvals,d4_grid,d4_gridvals,special_n_d4,a1_gridvals,a2_gridvals,semiz_gridvals,pi_semiz,pi_u_col,...
    level1ii,level1iidiff,a2Bind,zBind,d3ind,V_ford4_jj,Policy_ford4_jj,vfoptions)

aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

if vfoptions.lowmemory==0
    for d4_c=1:N_d4
        pi_semizd4=pi_semiz(:,:,d4_c); % no kron in noz case
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        EV=EVnext.*shiftdim(pi_semizd4',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a,N_semiz_count]);

        skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz_count)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz_count)-1)));
        aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz_count);
        aprimeProbs(skipinterp)=0;
        aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz_count]);

        EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz_count)-1)),[N_d23*N_a1,N_u,N_semiz_count]).*aprimeProbs;
        EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz_count)-1)),[N_d23*N_a1,N_u,N_semiz_count]).*(1-aprimeProbs);
        EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
        EV=reshape(EV,[N_d23*N_a1,N_semiz_count]);

        EVres=reshape(EV,[N_d2,N_d3*N_a1,N_semiz_count]);
        [EV_onlyd3,d2index]=max(EVres,[],1);
        EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_semiz_count]);
        d2index_resh=reshape(d2index,[N_d3,N_a1,N_semiz_count]);

        DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_semiz_count]);

        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_semiz, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals, ReturnFnParamsVec,1,0);
        RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2,N_semiz_count]);
        DEV=reshape(DiscountedEV,[1,N_d3,1,N_a1,1,N_semiz_count]);
        entireRHS_ii=RM+DEV;
        entireRHS_ii=reshape(entireRHS_ii,[N_d13,vfoptions.level1n,N_a1,N_a2,N_semiz_count]);

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_semiz_count]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
        Policy_ford4_jj(curraindex,:,d4_c)=encodePolicy_with_d2lookup(shiftdim(maxindex2,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_semiz_count);

        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals, ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*zBind;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz_count]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                Policy_ford4_jj(curraindex,:,d4_c)=encodePolicy_with_d2lookup(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_semiz_count);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_semiz, d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals, ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*zBind;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_semiz_count]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                Policy_ford4_jj(curraindex,:,d4_c)=encodePolicy_with_d2lookup(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_semiz_count);
            end
        end
    end

elseif vfoptions.lowmemory==1
    for d4_c=1:N_d4
        pi_semizd4=pi_semiz(:,:,d4_c);
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        for z_c=1:N_semiz_count
            z_val=semiz_gridvals(z_c,:);

            EV_z=EVnext.*pi_semizd4(z_c,:);
            EV_z(isnan(EV_z))=0;
            EV_z=sum(EV_z,2);
            EV_z=reshape(EV_z,[N_a,1]);

            skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
            aprimeProbs=repmat(a2primeProbs,N_a1,1);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);

            EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
            EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
            EV_z=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);

            EVres=reshape(EV_z,[N_d2,N_d3*N_a1]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,1]);
            d2index_z=reshape(d2index,[N_d3,N_a1]);

            DiscountedEV_z=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1]);

            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,ones(1,length(n_semiz)), d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii_z,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2]);
            DEV=reshape(DiscountedEV_z,[1,N_d3,1,N_a1,1]);
            entireRHS_ii_z=RM+DEV;
            entireRHS_ii_z=reshape(entireRHS_ii_z,[N_d13,vfoptions.level1n,N_a1,N_a2]);

            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
            Policy_ford4_jj(curraindex,z_c,d4_c)=encodePolicy_with_d2lookup_z(shiftdim(maxindex2,1),N_d1,N_d2,N_d3,N_d13,d2index_z,N_a1);

            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,ones(1,length(n_semiz)), d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                    d3aprime=d3ind+N_d3*(a1primeindexes-1);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                    Policy_ford4_jj(curraindex,z_c,d4_c)=encodePolicy_with_d2lookup_z(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_z,N_a1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,ones(1,length(n_semiz)), d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                    d3aprime=d3ind+N_d3*(loweredge-1);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                    Policy_ford4_jj(curraindex,z_c,d4_c)=encodePolicy_with_d2lookup_z(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_z,N_a1);
                end
            end
        end
    end
end

% Cross-d4 max
[V_jj,d4winner]=max(V_ford4_jj,[],3);
linidx_d4=(1:1:N_a*N_semiz_count)'+(N_a*N_semiz_count)*(reshape(d4winner,[N_a*N_semiz_count,1])-1);
pol_no_d4=reshape(Policy_ford4_jj(linidx_d4),[N_a,N_semiz_count]);
d1d2d3_part=rem(pol_no_d4-1,N_d1d2d3)+1;
a1primepart=ceil(pol_no_d4/N_d1d2d3);
Policy_jj=d1d2d3_part+N_d1d2d3*(d4winner-1)+N_d1d2d3*N_d4*(a1primepart-1);

end


%% Helpers
function pol=encodePolicy_with_d2lookup(pol_d13_a1,N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_bothz_count)
d1part=rem(pol_d13_a1-1,N_d1)+1;
d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
a1primepart=ceil(pol_d13_a1/N_d13);
[npts,nz]=size(pol_d13_a1);
zidx=repmat(gpuArray(1:nz),npts,1);
lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
d2part=d2index_resh(lin);
pol=d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(a1primepart-1);
end

function pol=encodePolicy_with_d2lookup_z(pol_d13_a1,N_d1,N_d2,N_d3,N_d13,d2index_z,N_a1)
d1part=rem(pol_d13_a1-1,N_d1)+1;
d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
a1primepart=ceil(pol_d13_a1/N_d13);
lin=d3part+N_d3*(a1primepart-1);
d2part=d2index_z(lin);
pol=d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(a1primepart-1);
end
