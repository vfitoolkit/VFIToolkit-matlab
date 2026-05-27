function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
%
% Strategy: pre-refine d2 out of EV (max over d2 for each d3,a1prime,a2,z),
% then apply the ExpAssetu-style DC level1 pattern with d as d3 and broadcast d1 over Return.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_u=prod(n_u);

% For ReturnFn (d1 and d3 only)
n_d13=[n_d1,n_d3];
N_d13=N_d1*N_d3;
d13_grid=[d1_grid;d3_grid];
% For aprimeFn (d2 and d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); % single index encoding (d1,d2,d3,a1prime)

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid; % already a column vector
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

pi_u_col=pi_u(:); % column

if vfoptions.lowmemory==0
    zind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
else
    special_n_z=ones(1,length(n_z));
end

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1
a2Bind=gpuArray(0:1:N_a2-1); % already includes -1
d3ind=repelem((1:1:N_d3)',N_d1,1); % [N_d13,1]; maps full d13-index to d3-component

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,N_j)=encodePolicy_no_d2(shiftdim(maxindex2,1),N_d1,N_d2,N_d3,N_d13);

        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % Level=2, Refine=0
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                Policy(curraindex,:,N_j)=encodePolicy_no_d2(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % Level=2, Refine=0
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                Policy(curraindex,:,N_j)=encodePolicy_no_d2(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13);
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,z_c,N_j)=encodePolicy_no_d2(shiftdim(maxindex2,1),N_d1,N_d2,N_d3,N_d13);

            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                    Policy(curraindex,z_c,N_j)=encodePolicy_no_d2(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,special_n_z, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                    Policy(curraindex,z_c,N_j)=encodePolicy_no_d2(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13);
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

    [V(:,:,N_j),Policy(:,:,N_j)]=internal_per_j(EVpre,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_a1,n_a2,n_z,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_z,N_u,...
        d13_gridvals,a1_gridvals,a2_gridvals,z_gridvals_J(:,:,N_j),pi_z_J(:,:,N_j),pi_u_col,...
        level1ii,level1iidiff,a2Bind,zBind,d3ind,vfoptions);
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

    EVnext=V(:,:,jj+1);
    [V(:,:,jj),Policy(:,:,jj)]=internal_per_j(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_a1,n_a2,n_z,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_z,N_u,...
        d13_gridvals,a1_gridvals,a2_gridvals,z_gridvals_J(:,:,jj),pi_z_J(:,:,jj),pi_u_col,...
        level1ii,level1iidiff,a2Bind,zBind,d3ind,vfoptions);
end


end


%% Per-period inner: compute V(:,:) and Policy(:,:) at age jj using EVnext = V(:,:,jj+1) (or V_Jplus1)
function [V_jj,Policy_jj]=internal_per_j(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d1,n_d3,n_a1,n_a2,n_z,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_z,N_u,...
    d13_gridvals,a1_gridvals,a2_gridvals,z_gridvals,pi_z,pi_u_col,...
    level1ii,level1iidiff,a2Bind,zBind,d3ind,vfoptions)

V_jj=zeros(N_a,N_z,'gpuArray');
Policy_jj=zeros(N_a,N_z,'gpuArray');

% (a1prime,a2prime) interpolation indexes for full (d23,a1prime)
aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d23*N_a1,N_u]
aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d23*N_a1,N_u]

% Compute EV integrated over u and zprime, then refine out d2
EV=EVnext.*shiftdim(pi_z',-1); % [N_a,N_z,N_z(prime)]
EV(isnan(EV))=0;
EV=sum(EV,2); % [N_a,1,N_z]
EV=reshape(EV,[N_a,N_z]); % [N_a,N_z]

skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
aprimeProbs=repmat(a2primeProbs,N_a1,N_z);  % [N_d23*N_a1, N_u*N_z]? no: repmat ofN_a1 rows, N_z cols of [N_d23,N_u] -> [N_d23*N_a1, N_u*N_z]
aprimeProbs(skipinterp)=0;
aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_z]);

EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbs;
EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbs);
EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2); % [N_d23*N_a1,1,N_z]
EV=reshape(EV,[N_d23*N_a1,N_z]); % [N_d23*N_a1,N_z]

% Refine d2 out: maximize EV over d2 to get EV_onlyd3 [N_d3*N_a1, N_z] and d2index recording argmax d2
EVres=reshape(EV,[N_d2,N_d3*N_a1,N_z]);
[EV_onlyd3,d2index]=max(EVres,[],1); % [1, N_d3*N_a1, N_z]
EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_z]);
% d2index has shape [1,N_d3*N_a1,N_z]
d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]); % for later lookup

DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_z]); % [N_d3,N_a1,1,1,N_z]
% d1 broadcasts (Return only has d1,d3); a2 broadcasts (EV doesn't depend on current a2)

if vfoptions.lowmemory==0
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals, ReturnFnParamsVec,1,0); % Level=1, Refine=0
    % [N_d13, level1n, N_a1, N_a2, N_z]
    % Broadcast DiscountedEV [N_d3,N_a1,1,1,N_z] across d1 and a2.
    % Reshape ReturnMatrix_ii into [N_d1,N_d3,level1n,N_a1,N_a2,N_z] for broadcast
    RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2,N_z]);
    DEV=reshape(DiscountedEV,[1,N_d3,1,N_a1,1,N_z]);
    entireRHS_ii=RM+DEV; % [N_d1,N_d3,level1n,N_a1,N_a2,N_z]
    entireRHS_ii=reshape(entireRHS_ii,[N_d13,vfoptions.level1n,N_a1,N_a2,N_z]);

    [~,maxindex1]=max(entireRHS_ii,[],2);
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
    V_jj(curraindex,:)=shiftdim(Vtempii,1);
    Policy_jj(curraindex,:)=encodePolicy_with_d2lookup(shiftdim(maxindex2,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_a2,N_z);

    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals, ReturnFnParamsVec,3,0); % Level=2, Refine=0
            % ReturnMatrix_ii: [N_d13*(maxgap+1), level1iidiff, N_a2, N_z]
            % Build linear index into DiscountedEV [N_d3,N_a1,1,1,N_z]:
            % For each (d13,maxgap+1,1,N_a2,N_z), index = d3(d13) + N_d3*(a1primeindexes-1) + N_d3*N_a1*0 + N_d3*N_a1*N_z*(z-1)
            d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*zBind; % [N_d13,maxgap+1,1,N_a2,N_z]
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V_jj(curraindex,:)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d13)+1);
            allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
            pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
            Policy_jj(curraindex,:)=encodePolicy_with_d2lookup(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_a2,N_z);
        else
            loweredge=maxindex1(:,1,ii,:,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals, ReturnFnParamsVec,3,0);
            d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*zBind;
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_z]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V_jj(curraindex,:)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d13)+1);
            allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
            pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
            Policy_jj(curraindex,:)=encodePolicy_with_d2lookup(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_a2,N_z);
        end
    end

elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c); % [N_d3,N_a1,1,1]
        d2index_z=d2index_resh(:,:,z_c); % [N_d3,N_a1]
        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,ones(1,length(n_z)), d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
        RM=reshape(ReturnMatrix_ii_z,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2]);
        DEV=reshape(DiscountedEV_z,[1,N_d3,1,N_a1,1]);
        entireRHS_ii_z=RM+DEV;
        entireRHS_ii_z=reshape(entireRHS_ii_z,[N_d13,vfoptions.level1n,N_a1,N_a2]);

        [~,maxindex1]=max(entireRHS_ii_z,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V_jj(curraindex,z_c)=shiftdim(Vtempii,1);
        Policy_jj(curraindex,z_c)=encodePolicy_with_d2lookup_z(shiftdim(maxindex2,1),N_d1,N_d2,N_d3,N_d13,d2index_z,N_a1,N_a2);

        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,ones(1,length(n_z)), d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                d3aprime=d3ind+N_d3*(a1primeindexes-1); % [N_d13,maxgap+1,1,N_a2]
                entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                V_jj(curraindex,z_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                Policy_jj(curraindex,z_c)=encodePolicy_with_d2lookup_z(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_z,N_a1,N_a2);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,ones(1,length(n_z)), d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                d3aprime=d3ind+N_d3*(loweredge-1);
                entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                V_jj(curraindex,z_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                pol_d13_a1=maxindex+N_d13*(loweredge(allind)-1);
                Policy_jj(curraindex,z_c)=encodePolicy_with_d2lookup_z(shiftdim(pol_d13_a1,1),N_d1,N_d2,N_d3,N_d13,d2index_z,N_a1,N_a2);
            end
        end
    end
end

end


%% Helpers for combining (d13,a1prime) policy with d2 (no V_Jplus1: d2 meaningless, set to 1)
function pol=encodePolicy_no_d2(pol_d13_a1,N_d1,N_d2,N_d3,N_d13)
% pol_d13_a1 is the encoded index into [d1,d3,a1prime] (with d1 fastest, then d3, then a1prime)
% Encode into combined [d1,d2,d3,a1prime] index with d1 fastest, then d2, then d3, then a1prime
d1part=rem(pol_d13_a1-1,N_d1)+1;
d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
a1primepart=ceil(pol_d13_a1/N_d13);
pol=d1part+N_d1*(1-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(a1primepart-1);
end

function pol=encodePolicy_with_d2lookup(pol_d13_a1,N_d1,N_d2,N_d3,N_d13,d2index_resh,N_a1,N_a2,N_z)
% pol_d13_a1: [npts,N_z] (or [npts*N_a2,N_z]) where each entry encodes (d1,d3,a1prime) with d1 fastest
% d2index_resh: [N_d3,N_a1,N_z]
d1part=rem(pol_d13_a1-1,N_d1)+1;
d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
a1primepart=ceil(pol_d13_a1/N_d13);
% Look up d2 = d2index_resh(d3part, a1primepart, z)
[npts,nz]=size(pol_d13_a1);
zidx=repmat(gpuArray(1:nz),npts,1); % match cols
lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
d2part=d2index_resh(lin);
pol=d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(a1primepart-1);
end

function pol=encodePolicy_with_d2lookup_z(pol_d13_a1,N_d1,N_d2,N_d3,N_d13,d2index_z,N_a1,N_a2)
d1part=rem(pol_d13_a1-1,N_d1)+1;
d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
a1primepart=ceil(pol_d13_a1/N_d13);
% d2index_z: [N_d3,N_a1]
lin=d3part+N_d3*(a1primepart-1);
d2part=d2index_z(lin);
pol=d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(a1primepart-1);
end
