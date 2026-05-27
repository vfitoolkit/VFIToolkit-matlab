function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
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
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);
N_u=prod(n_u);

N_d13=N_d1*N_d3;
N_d1d2d3=N_d1*N_d2*N_d3;

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');

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
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2Bind=gpuArray(0:1:N_a2-1);
zBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
d3ind=repelem((1:1:N_d3)',N_d1,1);

V_ford4_jj=zeros(N_a,N_bothz,N_e,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_bothz,N_e,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_bothz, n_e, d1d3d4a1_gridvals, a1a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
        d1d3_ind=rem(dindex-1,N_d13)+1;
        d1part=rem(d1d3_ind-1,N_d1)+1;
        d3part=ceil(d1d3_ind/N_d1);
        d4part=ceil(dindex/N_d13);
        a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
        Policy(:,:,:,N_j)=shiftdim(d1part+N_d1*(1-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(d4part-1)+N_d1*N_d2*N_d3*N_d4*(a1primepart-1),1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_bothz, special_n_e, d1d3d4a1_gridvals, a1a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
            d1d3_ind=rem(dindex-1,N_d13)+1;
            d1part=rem(d1d3_ind-1,N_d1)+1;
            d3part=ceil(d1d3_ind/N_d1);
            d4part=ceil(dindex/N_d13);
            a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
            Policy(:,:,e_c,N_j)=shiftdim(d1part+N_d1*(1-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(d4part-1)+N_d1*N_d2*N_d3*N_d4*(a1primepart-1),1);
        end
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], special_n_bothz, special_n_e, d1d3d4a1_gridvals, a1a2_gridvals, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtemp,1);
                dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
                d1d3_ind=rem(dindex-1,N_d13)+1;
                d1part=rem(d1d3_ind-1,N_d1)+1;
                d3part=ceil(d1d3_ind/N_d1);
                d4part=ceil(dindex/N_d13);
                a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
                Policy(:,z_c,e_c,N_j)=shiftdim(d1part+N_d1*(1-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(d4part-1)+N_d1*N_d2*N_d3*N_d4*(a1primepart-1),1);
            end
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]);
    EVpre=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j),-2),3); % [N_a,N_bothz]
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    [V(:,:,:,N_j),Policy(:,:,:,N_j)]=internal_per_j_e(EVpre,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_d4,n_a1,n_a2,n_bothz,n_e,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d1d2d3,N_a1,N_a2,N_a,N_bothz,N_e,N_u,...
        d13_gridvals,d4_grid,d4_gridvals,special_n_d4,a1_gridvals,a2_gridvals,bothz_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),pi_z_J(:,:,N_j),pi_semiz,pi_u_col,...
        level1ii,level1iidiff,a2Bind,zBind,eBind,d3ind,V_ford4_jj,Policy_ford4_jj,vfoptions);
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

    EVnext=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3); % [N_a,N_bothz]

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    [V(:,:,:,jj),Policy(:,:,:,jj)]=internal_per_j_e(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_d4,n_a1,n_a2,n_bothz,n_e,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d1d2d3,N_a1,N_a2,N_a,N_bothz,N_e,N_u,...
        d13_gridvals,d4_grid,d4_gridvals,special_n_d4,a1_gridvals,a2_gridvals,bothz_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),pi_z_J(:,:,jj),pi_semiz,pi_u_col,...
        level1ii,level1iidiff,a2Bind,zBind,eBind,d3ind,V_ford4_jj,Policy_ford4_jj,vfoptions);
end


end


%% Per-period inner (with e). EVnext has shape [N_a,N_bothz] (e already integrated out)
function [V_jj,Policy_jj]=internal_per_j_e(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d1,n_d3,n_d4,n_a1,n_a2,n_bothz,n_e,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d1d2d3,N_a1,N_a2,N_a,N_bothz_count,N_e_count,N_u,...
    d13_gridvals,d4_grid,d4_gridvals,special_n_d4,a1_gridvals,a2_gridvals,bothz_gridvals,e_gridvals,pi_z,pi_semiz,pi_u_col,...
    level1ii,level1iidiff,a2Bind,zBind,eBind,d3ind,V_ford4_jj,Policy_ford4_jj,vfoptions)

aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

if vfoptions.lowmemory==0
    for d4_c=1:N_d4
        pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        EV=EVnext.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a,N_bothz_count]);

        skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz_count)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz_count)-1)));
        aprimeProbs=repmat(a2primeProbs,N_a1,N_bothz_count);
        aprimeProbs(skipinterp)=0;
        aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_bothz_count]);

        EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_bothz_count)-1)),[N_d23*N_a1,N_u,N_bothz_count]).*aprimeProbs;
        EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz_count)-1)),[N_d23*N_a1,N_u,N_bothz_count]).*(1-aprimeProbs);
        EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
        EV=reshape(EV,[N_d23*N_a1,N_bothz_count]);

        EVres=reshape(EV,[N_d2,N_d3*N_a1,N_bothz_count]);
        [EV_onlyd3,d2index]=max(EVres,[],1);
        EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_bothz_count]);
        d2index_resh=reshape(d2index,[N_d3,N_a1,N_bothz_count]);

        DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_bothz_count]);

        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec,1,0);
        % [N_d13, level1n, N_a1, N_a2, N_bothz, N_e]
        RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2,N_bothz_count,N_e_count]);
        DEV=reshape(DiscountedEV,[1,N_d3,1,N_a1,1,N_bothz_count,1]);
        entireRHS_ii=RM+DEV;
        entireRHS_ii=reshape(entireRHS_ii,[N_d13,vfoptions.level1n,N_a1,N_a2,N_bothz_count,N_e_count]);

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_bothz_count,N_e_count]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V_ford4_jj(curraindex,:,:,d4_c)=shiftdim(Vtempii,1);
        Policy_ford4_jj(curraindex,:,:,d4_c)=encodePolicy_with_d2lookup_e(shiftdim(maxindex2,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_bothz_count,N_e_count);

        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*zBind;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz_count,N_e_count]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford4_jj(curraindex,:,:,d4_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind+N_d13*N_a2*N_bothz_count*eBind;
                pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                Policy_ford4_jj(curraindex,:,:,d4_c)=encodePolicy_with_d2lookup_e(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_bothz_count,N_e_count);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_bothz,n_e, d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*zBind;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_bothz_count,N_e_count]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford4_jj(curraindex,:,:,d4_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind+N_d13*N_a2*N_bothz_count*eBind;
                pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                Policy_ford4_jj(curraindex,:,:,d4_c)=encodePolicy_with_d2lookup_e(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_bothz_count,N_e_count);
            end
        end
    end

elseif vfoptions.lowmemory>=1
    % Loop over e inside d4
    for d4_c=1:N_d4
        pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        % Compute d2-refined EV once for this d4 (independent of e)
        EV=EVnext.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a,N_bothz_count]);

        skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz_count)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz_count)-1)));
        aprimeProbs=repmat(a2primeProbs,N_a1,N_bothz_count);
        aprimeProbs(skipinterp)=0;
        aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_bothz_count]);

        EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_bothz_count)-1)),[N_d23*N_a1,N_u,N_bothz_count]).*aprimeProbs;
        EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz_count)-1)),[N_d23*N_a1,N_u,N_bothz_count]).*(1-aprimeProbs);
        EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
        EV=reshape(EV,[N_d23*N_a1,N_bothz_count]);

        EVres=reshape(EV,[N_d2,N_d3*N_a1,N_bothz_count]);
        [EV_onlyd3,d2index]=max(EVres,[],1);
        EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_bothz_count]);
        d2index_resh=reshape(d2index,[N_d3,N_a1,N_bothz_count]);

        DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_bothz_count]);

        for e_c=1:N_e_count
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_bothz,ones(1,length(n_e)), d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals, e_val, ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii_e,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2,N_bothz_count]);
            DEV=reshape(DiscountedEV,[1,N_d3,1,N_a1,1,N_bothz_count]);
            entireRHS_ii_e=RM+DEV;
            entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,vfoptions.level1n,N_a1,N_a2,N_bothz_count]);

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d13*N_a1,vfoptions.level1n*N_a2,N_bothz_count]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford4_jj(curraindex,:,e_c,d4_c)=shiftdim(Vtempii,1);
            Policy_ford4_jj(curraindex,:,e_c,d4_c)=encodePolicy_with_d2lookup(shiftdim(maxindex2,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_bothz_count);

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,ones(1,length(n_e)), d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals, e_val, ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*zBind;
                    entireRHS_ii_e=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz_count]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V_ford4_jj(curraindex,:,e_c,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                    Policy_ford4_jj(curraindex,:,e_c,d4_c)=encodePolicy_with_d2lookup(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_bothz_count);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_bothz,ones(1,length(n_e)), d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals, e_val, ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*zBind;
                    entireRHS_ii_e=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_bothz_count]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V_ford4_jj(curraindex,:,e_c,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                    Policy_ford4_jj(curraindex,:,e_c,d4_c)=encodePolicy_with_d2lookup(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_bothz_count);
                end
            end
        end
    end
end

% Cross-d4 max (max over dim 4 since shape is [N_a,N_bothz,N_e,N_d4])
[V_jj,d4winner]=max(V_ford4_jj,[],4);
N=N_a*N_bothz_count*N_e_count;
linidx_d4=(1:1:N)'+N*(reshape(d4winner,[N,1])-1);
pol_no_d4=reshape(Policy_ford4_jj(linidx_d4),[N_a,N_bothz_count,N_e_count]);
d1d2d3_part=rem(pol_no_d4-1,N_d1d2d3)+1;
a1primepart=ceil(pol_no_d4/N_d1d2d3);
Policy_jj=d1d2d3_part+N_d1d2d3*(d4winner-1)+N_d1d2d3*N_d4*(a1primepart-1);

end


%% Helpers
function pol=encodePolicy_with_d2lookup(pol_d13_a1,N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_bothz_count)
% pol_d13_a1: [npts,N_bothz]
d1part=rem(pol_d13_a1-1,N_d1)+1;
d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
a1primepart=ceil(pol_d13_a1/N_d13);
[npts,nz]=size(pol_d13_a1);
zidx=repmat(gpuArray(1:nz),npts,1);
lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
d2part=d2index_resh(lin);
pol=d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(a1primepart-1);
end

function pol=encodePolicy_with_d2lookup_e(pol_d13_a1,N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_bothz_count,N_e_count)
% pol_d13_a1: [npts,N_bothz,N_e]; d2index_resh: [N_d3,N_a1,N_bothz]  (no e dep)
d1part=rem(pol_d13_a1-1,N_d1)+1;
d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
a1primepart=ceil(pol_d13_a1/N_d13);
[npts,nz,ne]=size(pol_d13_a1);
zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
d2part=d2index_resh(lin);
pol=d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(a1primepart-1);
end
