function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

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
Policy=zeros(4,N_a,N_z,N_j,'gpuArray'); % (1)=d1, (2)=d2, (3)=d3, (4)=a1prime
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid; % already a column vector
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

if vfoptions.lowmemory==0
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1);
else
    special_n_z=ones(1,length(n_z));
end

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Precompute
a2Bind=gpuArray(0:1:N_a2-1);
d3ind=repelem((1:1:N_d3)',N_d1,1); % [N_d13,1]; maps full d13-index to d3-component

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,N_j)=shiftdim(Vtempii,1);
        pol_d13_a1=shiftdim(maxindex2,1); % [npts,N_z]
        d_ind=rem(pol_d13_a1-1,N_d13)+1;
        Policy(1,curraindex,:,N_j)=rem(d_ind-1,N_d1)+1;        % d1
        Policy(3,curraindex,:,N_j)=ceil(d_ind/N_d1);            % d3
        Policy(4,curraindex,:,N_j)=ceil(pol_d13_a1/N_d13);      % a1prime

        % Divide-and-conquer layer 2
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1); % [npts,N_z]
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                Policy(1,curraindex,:,N_j)=rem(d_ind-1,N_d1)+1;
                Policy(3,curraindex,:,N_j)=ceil(d_ind/N_d1);
                Policy(4,curraindex,:,N_j)=ceil(pol_d13_a1/N_d13);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                Policy(1,curraindex,:,N_j)=rem(d_ind-1,N_d1)+1;
                Policy(3,curraindex,:,N_j)=ceil(d_ind/N_d1);
                Policy(4,curraindex,:,N_j)=ceil(pol_d13_a1/N_d13);
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
            pol_d13_a1=shiftdim(maxindex2,1);
            d_ind=rem(pol_d13_a1-1,N_d13)+1;
            Policy(1,curraindex,z_c,N_j)=rem(d_ind-1,N_d1)+1;
            Policy(3,curraindex,z_c,N_j)=ceil(d_ind/N_d1);
            Policy(4,curraindex,z_c,N_j)=ceil(pol_d13_a1/N_d13);

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
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    Policy(1,curraindex,z_c,N_j)=rem(d_ind-1,N_d1)+1;
                    Policy(3,curraindex,z_c,N_j)=ceil(d_ind/N_d1);
                    Policy(4,curraindex,z_c,N_j)=ceil(pol_d13_a1/N_d13);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,special_n_z, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    Policy(1,curraindex,z_c,N_j)=rem(d_ind-1,N_d1)+1;
                    Policy(3,curraindex,z_c,N_j)=ceil(d_ind/N_d1);
                    Policy(4,curraindex,z_c,N_j)=ceil(pol_d13_a1/N_d13);
                end
            end
        end
    end

    % d2, which was not in ReturnFn
    Policy(2,:,:,N_j)=ones(1,N_a,N_z,'gpuArray'); % d2 (terminal: d2 doesn't matter since it's only in the expectations term)

else % V_Jplus1

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    % Build a2primeIndex and a2primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d23*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d23*N_a1,N_u]

    % Get EV integrated over zprime
    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]);
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1); % [N_a,N_z,N_z(prime)]
    EV(isnan(EV))=0;
    EV=sum(EV,2);
    EV=reshape(EV,[N_a,N_z]);

    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
    aprimeProbs=repmat(a2primeProbs,N_a1,N_z);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_z]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    EV=reshape(EV,[N_d23*N_a1,N_z]);

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,N_z]),[],1);
    EV=reshape(EV,[N_d3*N_a1,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,1,1,N_z]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2,N_z]);
        DEV=reshape(DiscountedEV,[1,N_d3,1,N_a1,1,N_z]);
        entireRHS_ii=RM+DEV;
        entireRHS_ii=reshape(entireRHS_ii,[N_d13,vfoptions.level1n,N_a1,N_a2,N_z]);

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,N_j)=shiftdim(Vtempii,1);
        pol_d13_a1=shiftdim(maxindex2,1); % [npts,N_z]
        d_ind=rem(pol_d13_a1-1,N_d13)+1;
        d1part=rem(d_ind-1,N_d1)+1;
        d3part=ceil(d_ind/N_d1);
        a1primepart=ceil(pol_d13_a1/N_d13);
        Policy(1,curraindex,:,N_j)=d1part;
        Policy(3,curraindex,:,N_j)=d3part;
        Policy(4,curraindex,:,N_j)=a1primepart;
        % Get the d2Policy via lookup on (d3,a1prime,z)
        [npts,nz]=size(pol_d13_a1);
        zidx=repmat(gpuArray(1:nz),npts,1);
        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
        Policy(2,curraindex,:,N_j)=d2index_resh(lin);

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*zBind;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,:,N_j)=d1part;
                Policy(3,curraindex,:,N_j)=d3part;
                Policy(4,curraindex,:,N_j)=a1primepart;
                [npts,nz]=size(pol_d13_a1);
                zidx=repmat(gpuArray(1:nz),npts,1);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(2,curraindex,:,N_j)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*zBind;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,:,N_j)=d1part;
                Policy(3,curraindex,:,N_j)=d3part;
                Policy(4,curraindex,:,N_j)=a1primepart;
                [npts,nz]=size(pol_d13_a1);
                zidx=repmat(gpuArray(1:nz),npts,1);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(2,curraindex,:,N_j)=d2index_resh(lin);
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c); % [N_d3,N_a1,1,1]
            d2index_z=d2index_resh(:,:,z_c); % [N_d3,N_a1]
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii_z,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2]);
            DEV=reshape(DiscountedEV_z,[1,N_d3,1,N_a1,1]);
            entireRHS_ii_z=RM+DEV;
            entireRHS_ii_z=reshape(entireRHS_ii_z,[N_d13,vfoptions.level1n,N_a1,N_a2]);

            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
            pol_d13_a1=shiftdim(maxindex2,1);
            d_ind=rem(pol_d13_a1-1,N_d13)+1;
            d1part=rem(d_ind-1,N_d1)+1;
            d3part=ceil(d_ind/N_d1);
            a1primepart=ceil(pol_d13_a1/N_d13);
            Policy(1,curraindex,z_c,N_j)=d1part;
            Policy(3,curraindex,z_c,N_j)=d3part;
            Policy(4,curraindex,z_c,N_j)=a1primepart;
            lin=d3part+N_d3*(a1primepart-1);
            Policy(2,curraindex,z_c,N_j)=d2index_z(lin);

            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                    d3aprime=d3ind+N_d3*(a1primeindexes-1);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,z_c,N_j)=d1part;
                    Policy(3,curraindex,z_c,N_j)=d3part;
                    Policy(4,curraindex,z_c,N_j)=a1primepart;
                    lin=d3part+N_d3*(a1primepart-1);
                    Policy(2,curraindex,z_c,N_j)=d2index_z(lin);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,special_n_z, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                    d3aprime=d3ind+N_d3*(loweredge-1);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,z_c,N_j)=d1part;
                    Policy(3,curraindex,z_c,N_j)=d3part;
                    Policy(4,curraindex,z_c,N_j)=a1primepart;
                    lin=d3part+N_d3*(a1primepart-1);
                    Policy(2,curraindex,z_c,N_j)=d2index_z(lin);
                end
            end
        end
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

    % Build a2primeIndex and a2primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Get EV integrated over zprime
    EV=V(:,:,jj+1);
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);
    EV=reshape(EV,[N_a,N_z]);

    skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
    aprimeProbs=repmat(a2primeProbs,N_a1,N_z);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_z]);

    EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    EV=reshape(EV,[N_d23*N_a1,N_z]);

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,N_z]),[],1);
    EV=reshape(EV,[N_d3*N_a1,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,1,1,N_z]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
        RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2,N_z]);
        DEV=reshape(DiscountedEV,[1,N_d3,1,N_a1,1,N_z]);
        entireRHS_ii=RM+DEV;
        entireRHS_ii=reshape(entireRHS_ii,[N_d13,vfoptions.level1n,N_a1,N_a2,N_z]);

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,jj)=shiftdim(Vtempii,1);
        pol_d13_a1=shiftdim(maxindex2,1);
        d_ind=rem(pol_d13_a1-1,N_d13)+1;
        d1part=rem(d_ind-1,N_d1)+1;
        d3part=ceil(d_ind/N_d1);
        a1primepart=ceil(pol_d13_a1/N_d13);
        Policy(1,curraindex,:,jj)=d1part;
        Policy(3,curraindex,:,jj)=d3part;
        Policy(4,curraindex,:,jj)=a1primepart;
        [npts,nz]=size(pol_d13_a1);
        zidx=repmat(gpuArray(1:nz),npts,1);
        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
        Policy(2,curraindex,:,jj)=d2index_resh(lin);

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*zBind;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,:,jj)=d1part;
                Policy(3,curraindex,:,jj)=d3part;
                Policy(4,curraindex,:,jj)=a1primepart;
                [npts,nz]=size(pol_d13_a1);
                zidx=repmat(gpuArray(1:nz),npts,1);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(2,curraindex,:,jj)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*zBind;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,:,jj)=d1part;
                Policy(3,curraindex,:,jj)=d3part;
                Policy(4,curraindex,:,jj)=a1primepart;
                [npts,nz]=size(pol_d13_a1);
                zidx=repmat(gpuArray(1:nz),npts,1);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(2,curraindex,:,jj)=d2index_resh(lin);
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
            d2index_z=d2index_resh(:,:,z_c);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii_z,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2]);
            DEV=reshape(DiscountedEV_z,[1,N_d3,1,N_a1,1]);
            entireRHS_ii_z=RM+DEV;
            entireRHS_ii_z=reshape(entireRHS_ii_z,[N_d13,vfoptions.level1n,N_a1,N_a2]);

            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
            pol_d13_a1=shiftdim(maxindex2,1);
            d_ind=rem(pol_d13_a1-1,N_d13)+1;
            d1part=rem(d_ind-1,N_d1)+1;
            d3part=ceil(d_ind/N_d1);
            a1primepart=ceil(pol_d13_a1/N_d13);
            Policy(1,curraindex,z_c,jj)=d1part;
            Policy(3,curraindex,z_c,jj)=d3part;
            Policy(4,curraindex,z_c,jj)=a1primepart;
            lin=d3part+N_d3*(a1primepart-1);
            Policy(2,curraindex,z_c,jj)=d2index_z(lin);

            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                    d3aprime=d3ind+N_d3*(a1primeindexes-1);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,z_c,jj)=d1part;
                    Policy(3,curraindex,z_c,jj)=d3part;
                    Policy(4,curraindex,z_c,jj)=a1primepart;
                    lin=d3part+N_d3*(a1primepart-1);
                    Policy(2,curraindex,z_c,jj)=d2index_z(lin);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,special_n_z, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                    d3aprime=d3ind+N_d3*(loweredge-1);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,z_c,jj)=d1part;
                    Policy(3,curraindex,z_c,jj)=d3part;
                    Policy(4,curraindex,z_c,jj)=a1primepart;
                    lin=d3part+N_d3*(a1primepart-1);
                    Policy(2,curraindex,z_c,jj)=d2index_z(lin);
                end
            end
        end
    end
end


end
