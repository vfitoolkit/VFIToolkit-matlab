function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC1_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No z variant.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_u=prod(n_u);

n_d13=[n_d1,n_d3];
N_d13=N_d1*N_d3;
d13_grid=[d1_grid;d3_grid];
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray');

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

pi_u_col=pi_u(:);

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2Bind=gpuArray(0:1:N_a2-1);
d3ind=repelem((1:1:N_d3)',N_d1,1); % [N_d13,1]

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, ReturnFnParamsVec,1,0);
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
    V(curraindex,N_j)=shiftdim(Vtempii,1);
    Policy(curraindex,N_j)=encodePolicy_no_d2(shiftdim(maxindex2,1),N_d1,N_d2,N_d3,N_d13);

    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, ReturnFnParamsVec,2,0);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,N_j)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d13)+1);
            allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
            pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
            Policy(curraindex,N_j)=encodePolicy_no_d2(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13);
        else
            loweredge=maxindex1(:,1,ii,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, ReturnFnParamsVec,2,0);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,N_j)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d13)+1);
            allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
            pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
            Policy(curraindex,N_j)=encodePolicy_no_d2(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13);
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    [V(:,N_j),Policy(:,N_j)]=internal_per_j_noz(EVpre,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_a1,n_a2,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_u,...
        d13_gridvals,a1_gridvals,a2_gridvals,pi_u_col,...
        level1ii,level1iidiff,a2Bind,d3ind,vfoptions);
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

    EVnext=V(:,jj+1);
    [V(:,jj),Policy(:,jj)]=internal_per_j_noz(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_a1,n_a2,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_u,...
        d13_gridvals,a1_gridvals,a2_gridvals,pi_u_col,...
        level1ii,level1iidiff,a2Bind,d3ind,vfoptions);
end


end


%% Per-period inner (noz)
function [V_jj,Policy_jj]=internal_per_j_noz(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d1,n_d3,n_a1,n_a2,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_u,...
    d13_gridvals,a1_gridvals,a2_gridvals,pi_u_col,...
    level1ii,level1iidiff,a2Bind,d3ind,vfoptions)

V_jj=zeros(N_a,1,'gpuArray');
Policy_jj=zeros(N_a,1,'gpuArray');

aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

EV=EVnext(:); % [N_a,1]
skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
aprimeProbs=repmat(a2primeProbs,N_a1,1);
aprimeProbs(skipinterp)=0;
aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);

EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2); % [N_d23*N_a1,1]

EVres=reshape(EV,[N_d2,N_d3*N_a1]);
[EV_onlyd3,d2index]=max(EVres,[],1); % [1,N_d3*N_a1]
EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,1]);
d2index_resh=reshape(d2index,[N_d3,N_a1]);

DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1]); % [N_d3,N_a1,1,1]

ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, ReturnFnParamsVec,1,0);
% [N_d13, level1n, N_a1, N_a2]
RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2]);
DEV=reshape(DiscountedEV,[1,N_d3,1,N_a1,1]);
entireRHS_ii=RM+DEV;
entireRHS_ii=reshape(entireRHS_ii,[N_d13,vfoptions.level1n,N_a1,N_a2]);

[~,maxindex1]=max(entireRHS_ii,[],2);
[Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
V_jj(curraindex)=shiftdim(Vtempii,1);
Policy_jj(curraindex)=encodePolicy_with_d2lookup_z(shiftdim(maxindex2,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_a2);

maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
for ii=1:(vfoptions.level1n-1)
    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
        a1primeindexes=loweredge+(0:1:maxgap(ii));
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, ReturnFnParamsVec,3,0);
        d3aprime=d3ind+N_d3*(a1primeindexes-1);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V_jj(curraindex)=shiftdim(Vtempii,1);
        dind=(rem(maxindex-1,N_d13)+1);
        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
        pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
        Policy_jj(curraindex)=encodePolicy_with_d2lookup_z(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_a2);
    else
        loweredge=maxindex1(:,1,ii,:);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, ReturnFnParamsVec,3,0);
        d3aprime=d3ind+N_d3*(loweredge-1);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V_jj(curraindex)=shiftdim(Vtempii,1);
        dind=(rem(maxindex-1,N_d13)+1);
        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
        pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
        Policy_jj(curraindex)=encodePolicy_with_d2lookup_z(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_a2);
    end
end

end


%% Helpers (shared name with main DC1_raw but defined here for self-contained file)
function pol=encodePolicy_no_d2(pol_d13_a1,N_d1,N_d2,N_d3,N_d13)
d1part=rem(pol_d13_a1-1,N_d1)+1;
d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
a1primepart=ceil(pol_d13_a1/N_d13);
pol=d1part+N_d1*(1-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(a1primepart-1);
end

function pol=encodePolicy_with_d2lookup_z(pol_d13_a1,N_d1,N_d2,N_d3,N_d13,d2index_z,N_a1,N_a2)
d1part=rem(pol_d13_a1-1,N_d1)+1;
d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
a1primepart=ceil(pol_d13_a1/N_d13);
lin=d3part+N_d3*(a1primepart-1);
d2part=d2index_z(lin);
pol=d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(a1primepart-1);
end
