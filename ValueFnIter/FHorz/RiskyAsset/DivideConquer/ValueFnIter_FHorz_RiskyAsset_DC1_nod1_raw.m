function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC1_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_u,N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
%
% No d1 variant: ReturnFn depends on (d3,...) only.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_u=prod(n_u);

% For aprimeFn (d2 and d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); % single index encoding (d2,d3,a1prime)

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d3_gridvals=CreateGridvals(n_d3,d3_grid,1);

pi_u_col=pi_u(:);

if vfoptions.lowmemory==0
    zind=shiftdim(gpuArray(0:1:N_z-1),-3);
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1);
else
    special_n_z=ones(1,length(n_z));
end

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2Bind=gpuArray(0:1:N_a2-1);
d3ind=(1:1:N_d3)';

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_z, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d3*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,N_j)=encodePolicy_no_d2_nod1(shiftdim(maxindex2,1),N_d2,N_d3);

        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d3)+1);
                allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                Policy(curraindex,:,N_j)=encodePolicy_no_d2_nod1(shiftdim(pol_d3_a1,1),N_d2,N_d3);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,n_z, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d3)+1);
                allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                Policy(curraindex,:,N_j)=encodePolicy_no_d2_nod1(shiftdim(pol_d3_a1,1),N_d2,N_d3);
            end
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d3*N_a1,vfoptions.level1n*N_a2]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,z_c,N_j)=encodePolicy_no_d2_nod1(shiftdim(maxindex2,1),N_d2,N_d3);

            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                    Policy(curraindex,z_c,N_j)=encodePolicy_no_d2_nod1(shiftdim(pol_d3_a1,1),N_d2,N_d3);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,special_n_z, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                    Policy(curraindex,z_c,N_j)=encodePolicy_no_d2_nod1(shiftdim(pol_d3_a1,1),N_d2,N_d3);
                end
            end
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]);
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    [V(:,:,N_j),Policy(:,:,N_j)]=internal_per_j_nod1(EVpre,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d3,n_a1,n_a2,n_z,N_d2,N_d3,N_d23,N_a1,N_a2,N_a,N_z,N_u,...
        d3_gridvals,a1_gridvals,a2_gridvals,z_gridvals_J(:,:,N_j),pi_z_J(:,:,N_j),pi_u_col,...
        level1ii,level1iidiff,a2Bind,zBind,d3ind,vfoptions);
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
    [V(:,:,jj),Policy(:,:,jj)]=internal_per_j_nod1(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d3,n_a1,n_a2,n_z,N_d2,N_d3,N_d23,N_a1,N_a2,N_a,N_z,N_u,...
        d3_gridvals,a1_gridvals,a2_gridvals,z_gridvals_J(:,:,jj),pi_z_J(:,:,jj),pi_u_col,...
        level1ii,level1iidiff,a2Bind,zBind,d3ind,vfoptions);
end


end


%% Per-period inner (no d1)
function [V_jj,Policy_jj]=internal_per_j_nod1(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d3,n_a1,n_a2,n_z,N_d2,N_d3,N_d23,N_a1,N_a2,N_a,N_z,N_u,...
    d3_gridvals,a1_gridvals,a2_gridvals,z_gridvals,pi_z,pi_u_col,...
    level1ii,level1iidiff,a2Bind,zBind,d3ind,vfoptions)

V_jj=zeros(N_a,N_z,'gpuArray');
Policy_jj=zeros(N_a,N_z,'gpuArray');

aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

EV=EVnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0;
EV=sum(EV,2);
EV=reshape(EV,[N_a,N_z]);

skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
aprimeProbs=repmat(a2primeProbs,N_a1,N_z);
aprimeProbs(skipinterp)=0;
aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_z]);

EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbs;
EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbs);
EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
EV=reshape(EV,[N_d23*N_a1,N_z]);

EVres=reshape(EV,[N_d2,N_d3*N_a1,N_z]);
[EV_onlyd3,d2index]=max(EVres,[],1);
EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_z]);
d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_z]);

if vfoptions.lowmemory==0
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_z, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals, ReturnFnParamsVec,1,0);
    % ReturnMatrix_ii: [N_d3, level1n, N_a1, N_a2, N_z]
    RM=reshape(ReturnMatrix_ii,[N_d3,vfoptions.level1n,N_a1,N_a2,N_z]);
    DEV=reshape(DiscountedEV,[N_d3,1,N_a1,1,N_z]);
    entireRHS_ii=RM+DEV; % [N_d3, level1n, N_a1, N_a2, N_z]
    entireRHS_ii=reshape(entireRHS_ii,[N_d3,vfoptions.level1n,N_a1,N_a2,N_z]);

    [~,maxindex1]=max(entireRHS_ii,[],2);
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d3*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
    V_jj(curraindex,:)=shiftdim(Vtempii,1);
    Policy_jj(curraindex,:)=encodePolicy_with_d2lookup_nod1(shiftdim(maxindex2,1),N_d2,N_d3,d2index_resh,N_a1);

    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals, ReturnFnParamsVec,3,0);
            d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*zBind;
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V_jj(curraindex,:)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d3)+1);
            allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
            pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
            Policy_jj(curraindex,:)=encodePolicy_with_d2lookup_nod1(shiftdim(pol_d3_a1,1),N_d2,N_d3,d2index_resh,N_a1);
        else
            loweredge=maxindex1(:,1,ii,:,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,n_z, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals, ReturnFnParamsVec,3,0);
            d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*zBind;
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3,level1iidiff(ii)*N_a2,N_z]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V_jj(curraindex,:)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d3)+1);
            allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
            pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
            Policy_jj(curraindex,:)=encodePolicy_with_d2lookup_nod1(shiftdim(pol_d3_a1,1),N_d2,N_d3,d2index_resh,N_a1);
        end
    end

elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        d2index_z=d2index_resh(:,:,z_c);
        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,ones(1,length(n_z)), d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
        RM=reshape(ReturnMatrix_ii_z,[N_d3,vfoptions.level1n,N_a1,N_a2]);
        DEV=reshape(DiscountedEV_z,[N_d3,1,N_a1,1]);
        entireRHS_ii_z=RM+DEV;

        [~,maxindex1]=max(entireRHS_ii_z,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d3*N_a1,vfoptions.level1n*N_a2]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V_jj(curraindex,z_c)=shiftdim(Vtempii,1);
        Policy_jj(curraindex,z_c)=encodePolicy_with_d2lookup_nod1_z(shiftdim(maxindex2,1),N_d2,N_d3,d2index_z,N_a1);

        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,ones(1,length(n_z)), d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                d3aprime=d3ind+N_d3*(a1primeindexes-1);
                entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                V_jj(curraindex,z_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d3)+1);
                allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii));
                pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                Policy_jj(curraindex,z_c)=encodePolicy_with_d2lookup_nod1_z(shiftdim(pol_d3_a1,1),N_d2,N_d3,d2index_z,N_a1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,ones(1,length(n_z)), d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                d3aprime=d3ind+N_d3*(loweredge-1);
                entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d3,level1iidiff(ii)*N_a2]);
                [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                V_jj(curraindex,z_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d3)+1);
                allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii));
                pol_d3_a1=maxindex+N_d3*(loweredge(allind)-1);
                Policy_jj(curraindex,z_c)=encodePolicy_with_d2lookup_nod1_z(shiftdim(pol_d3_a1,1),N_d2,N_d3,d2index_z,N_a1);
            end
        end
    end
end

end


%% Helpers (no d1)
function pol=encodePolicy_no_d2_nod1(pol_d3_a1,N_d2,N_d3)
% pol_d3_a1 encodes (d3, a1prime) with d3 fastest
% Combined index (d2,d3,a1prime) with d2 fastest, then d3, then a1prime. d2 set to 1.
d3part=rem(pol_d3_a1-1,N_d3)+1;
a1primepart=ceil(pol_d3_a1/N_d3);
pol=1+N_d2*(d3part-1)+N_d2*N_d3*(a1primepart-1);
end

function pol=encodePolicy_with_d2lookup_nod1(pol_d3_a1,N_d2,N_d3,d2index_resh,N_a1)
d3part=rem(pol_d3_a1-1,N_d3)+1;
a1primepart=ceil(pol_d3_a1/N_d3);
[npts,nz]=size(pol_d3_a1);
zidx=repmat(gpuArray(1:nz),npts,1);
lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
d2part=d2index_resh(lin);
pol=d2part+N_d2*(d3part-1)+N_d2*N_d3*(a1primepart-1);
end

function pol=encodePolicy_with_d2lookup_nod1_z(pol_d3_a1,N_d2,N_d3,d2index_z,N_a1)
d3part=rem(pol_d3_a1-1,N_d3)+1;
a1primepart=ceil(pol_d3_a1/N_d3);
lin=d3part+N_d3*(a1primepart-1);
d2part=d2index_z(lin);
pol=d2part+N_d2*(d3part-1)+N_d2*N_d3*(a1primepart-1);
end
