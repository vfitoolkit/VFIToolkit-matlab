function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
%
% DC + d4 outer loop. Inside each d4: refine d2 out of EV, then run level1n DC over a1 with d1+d3+a1prime.
% After d4 loop: max over d4 and look up the corresponding (d1,d2,d3,a1prime).
% Policy output has the choices on the first dimension: (d1,d2,d3,d4,a1prime).

n_bothz=[n_semiz,n_z]; % return-fn shock arguments

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
N_u=prod(n_u);

N_d13=N_d1*N_d3;
N_d1d2d3=N_d1*N_d2*N_d3; %#ok<NASGU>

% Variant of d for the semiz transition
special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

% For ReturnFn (d1 and d3 inside the level1 helper)
n_d13_local=[n_d1,n_d3]; %#ok<NASGU>
d13_grid=[d1_grid;d3_grid]; %#ok<NASGU>
% For aprimeFn (d2 and d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_bothz,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_j,'gpuArray'); % d1, d2, d3, d4 and a1prime

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
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2Bind=gpuArray(0:1:N_a2-1);
zBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
d3ind=repelem((1:1:N_d3)',N_d1,1); % [N_d13,1]

% Accumulators for the choice of d4 (max is taken across d4 at the end of each period)
V_ford4_jj=zeros(N_a,N_bothz,N_d4,'gpuArray');
d1_ford4_jj=zeros(N_a,N_bothz,N_d4,'gpuArray');
d2_ford4_jj=zeros(N_a,N_bothz,N_d4,'gpuArray');
d3_ford4_jj=zeros(N_a,N_bothz,N_d4,'gpuArray');
a1prime_ford4_jj=zeros(N_a,N_bothz,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % Terminal: only ReturnFn matters; d2 is meaningless (set to 1).
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_bothz, d1d3d4a1_gridvals, a1a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=shiftdim(Vtemp,1);
        dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
        d1d3_ind=rem(dindex-1,N_d13)+1;
        Policy(1,:,:,N_j)=rem(d1d3_ind-1,N_d1)+1; % d1
        Policy(2,:,:,N_j)=1; % d2 is meaningless in the terminal period
        Policy(3,:,:,N_j)=ceil(d1d3_ind/N_d1); % d3
        Policy(4,:,:,N_j)=ceil(dindex/N_d13); % d4
        Policy(5,:,:,N_j)=ceil(maxindex/(N_d1*N_d3*N_d4)); % a1prime
    else
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], special_n_bothz, d1d3d4a1_gridvals, a1a2_gridvals, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtemp,1);
            dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
            d1d3_ind=rem(dindex-1,N_d13)+1;
            Policy(1,:,z_c,N_j)=rem(d1d3_ind-1,N_d1)+1; % d1
            Policy(2,:,z_c,N_j)=1; % d2 is meaningless in the terminal period
            Policy(3,:,z_c,N_j)=ceil(d1d3_ind/N_d1); % d3
            Policy(4,:,z_c,N_j)=ceil(dindex/N_d13); % d4
            Policy(5,:,z_c,N_j)=ceil(maxindex/(N_d1*N_d3*N_d4)); % a1prime
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    EVnext=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d23*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d23*N_a1,N_u]

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz(:,:,d4_c));
            % Build per-d4 (d1,d3) gridvals with d4 baked in
            d13_with_d4=[repmat(d13_gridvals,1,1),repmat(d4_gridvals(d4_c,:),N_d13,1)];

            % EV integrated over bothz' (zprime)
            EV=EVnext.*shiftdim(pi_bothz',-1); % [N_a,N_bothz,N_bothz']
            EV(isnan(EV))=0;
            EV=sum(EV,2); % sum over bothz', singular 2nd dim
            EV=reshape(EV,[N_a,N_bothz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_bothz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_bothz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a1,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a1,N_u,N_bothz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2); % [N_d23*N_a1,1,N_bothz]
            EV=reshape(EV,[N_d23*N_a1,N_bothz]);

            % Refine d2: max over d2
            EVres=reshape(EV,[N_d2,N_d3*N_a1,N_bothz]);
            [EV_onlyd3,d2index]=max(EVres,[],1); % [1,N_d3*N_a1,N_bothz]
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_bothz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_bothz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_bothz]);

            % Level1: top points
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_bothz, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            % [N_d13, level1n, N_a1, N_a2, N_bothz]
            RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_bothz]);
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_bothz]);
            entireRHS_ii=RM+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_bothz]);

            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
            pol_d13_a1=shiftdim(maxindex2,1);
            d1part=rem(pol_d13_a1-1,N_d1)+1;
            d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
            a1primepart=ceil(pol_d13_a1/N_d13);
            [npts,nz]=size(pol_d13_a1);
            zidx=repmat(gpuArray(1:nz),npts,1);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
            d1_ford4_jj(curraindex,:,d4_c)=d1part;
            d3_ford4_jj(curraindex,:,d4_c)=d3part;
            a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
            d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);

            % Maxgap loop
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d1part=rem(pol_d13_a1-1,N_d1)+1;
                    d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    [npts,nz]=size(pol_d13_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d1_ford4_jj(curraindex,:,d4_c)=d1part;
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,n_bothz, d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d1part=rem(pol_d13_a1-1,N_d1)+1;
                    d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    [npts,nz]=size(pol_d13_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d1_ford4_jj(curraindex,:,d4_c)=d1part;
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz(:,:,d4_c));
            d13_with_d4=[repmat(d13_gridvals,1,1),repmat(d4_gridvals(d4_c,:),N_d13,1)];

            EV=EVnext.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_bothz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_bothz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_bothz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a1,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a1,N_u,N_bothz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a1,N_bothz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a1,N_bothz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_bothz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_bothz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_bothz]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                zBindblock=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                DiscountedEVblock=DiscountedEV(:,:,:,:,semizblock);
                d2index_reshblock=d2index_resh(:,:,semizblock);

                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                DEV=reshape(DiscountedEVblock,[1,N_d3,N_a1,1,1,N_semiz]);
                entireRHS_ii=RM+DEV;
                entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz]);

                [~,maxindex1]=max(entireRHS_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford4_jj(curraindex,semizblock,d4_c)=shiftdim(Vtempii,1);
                pol_d13_a1=shiftdim(maxindex2,1);
                d1part=rem(pol_d13_a1-1,N_d1)+1;
                d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                a1primepart=ceil(pol_d13_a1/N_d13);
                [npts,nz]=size(pol_d13_a1);
                zidx=repmat(gpuArray(1:nz),npts,1);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                d1_ford4_jj(curraindex,semizblock,d4_c)=d1part;
                d3_ford4_jj(curraindex,semizblock,d4_c)=d3part;
                a1prime_ford4_jj(curraindex,semizblock,d4_c)=a1primepart;
                d2_ford4_jj(curraindex,semizblock,d4_c)=d2index_reshblock(lin);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBindblock,-2);
                        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVblock(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford4_jj(curraindex,semizblock,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBindblock;
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        [npts,nz]=size(pol_d13_a1);
                        zidx=repmat(gpuArray(1:nz),npts,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                        d1_ford4_jj(curraindex,semizblock,d4_c)=d1part;
                        d3_ford4_jj(curraindex,semizblock,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,semizblock,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,semizblock,d4_c)=d2index_reshblock(lin);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBindblock,-2);
                        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVblock(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford4_jj(curraindex,semizblock,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBindblock;
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        [npts,nz]=size(pol_d13_a1);
                        zidx=repmat(gpuArray(1:nz),npts,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                        d1_ford4_jj(curraindex,semizblock,d4_c)=d1part;
                        d3_ford4_jj(curraindex,semizblock,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,semizblock,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,semizblock,d4_c)=d2index_reshblock(lin);
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz(:,:,d4_c));
            d13_with_d4=[repmat(d13_gridvals,1,1),repmat(d4_gridvals(d4_c,:),N_d13,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                EV_z=EVnext.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2); % [N_a,1]
                EV_z=reshape(EV_z,[N_a,1]);

                skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
                aprimeProbs=repmat(a2primeProbs,N_a1,1);
                aprimeProbs(skipinterp)=0;
                aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);

                EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
                EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
                EV_z=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2); % [N_d23*N_a1,1]

                EVres=reshape(EV_z,[N_d2,N_d3*N_a1]);
                [EV_onlyd3,d2index]=max(EVres,[],1);
                EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,1]);
                d2index_z=reshape(d2index,[N_d3,N_a1]);

                DiscountedEV_z=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1]);

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,ones(1,length(n_bothz)), d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii_z,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2]);
                DEV=reshape(DiscountedEV_z,[1,N_d3,N_a1,1,1]);
                entireRHS_ii_z=RM+DEV;
                entireRHS_ii_z=reshape(entireRHS_ii_z,[N_d13,N_a1,vfoptions.level1n,N_a2]);

                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                pol_d13_a1=shiftdim(maxindex2,1);
                d1part=rem(pol_d13_a1-1,N_d1)+1;
                d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                a1primepart=ceil(pol_d13_a1/N_d13);
                lin=d3part+N_d3*(a1primepart-1);
                d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);

                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,ones(1,length(n_bothz)), d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d3aprime=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        lin=d3part+N_d3*(a1primepart-1);
                        d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,ones(1,length(n_bothz)), d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d3aprime=d3ind+N_d3*(loweredge-1);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        lin=d3part+N_d3*(a1primepart-1);
                        d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    end
                end
            end
        end
    end

    % Now max over d4
    [Vbest,d4winner]=max(V_ford4_jj,[],3); % [N_a,N_bothz]
    V(:,:,N_j)=Vbest;
    linidx_d4=(1:1:N_a*N_bothz)'+(N_a*N_bothz)*(reshape(d4winner,[N_a*N_bothz,1])-1);
    Policy(1,:,:,N_j)=reshape(d1_ford4_jj(linidx_d4),[1,N_a,N_bothz]);
    Policy(2,:,:,N_j)=reshape(d2_ford4_jj(linidx_d4),[1,N_a,N_bothz]);
    Policy(3,:,:,N_j)=reshape(d3_ford4_jj(linidx_d4),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(d4winner,[1,N_a,N_bothz]);
    Policy(5,:,:,N_j)=reshape(a1prime_ford4_jj(linidx_d4),[1,N_a,N_bothz]);
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

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d23*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d23*N_a1,N_u]

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz(:,:,d4_c));
            % Build per-d4 (d1,d3) gridvals with d4 baked in
            d13_with_d4=[repmat(d13_gridvals,1,1),repmat(d4_gridvals(d4_c,:),N_d13,1)];

            % EV integrated over bothz' (zprime)
            EV=EVnext.*shiftdim(pi_bothz',-1); % [N_a,N_bothz,N_bothz']
            EV(isnan(EV))=0;
            EV=sum(EV,2); % sum over bothz', singular 2nd dim
            EV=reshape(EV,[N_a,N_bothz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_bothz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_bothz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a1,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a1,N_u,N_bothz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2); % [N_d23*N_a1,1,N_bothz]
            EV=reshape(EV,[N_d23*N_a1,N_bothz]);

            % Refine d2: max over d2
            EVres=reshape(EV,[N_d2,N_d3*N_a1,N_bothz]);
            [EV_onlyd3,d2index]=max(EVres,[],1); % [1,N_d3*N_a1,N_bothz]
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_bothz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_bothz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_bothz]);

            % Level1: top points
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_bothz, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
            % [N_d13, level1n, N_a1, N_a2, N_bothz]
            RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_bothz]);
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_bothz]);
            entireRHS_ii=RM+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_bothz]);

            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
            pol_d13_a1=shiftdim(maxindex2,1);
            d1part=rem(pol_d13_a1-1,N_d1)+1;
            d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
            a1primepart=ceil(pol_d13_a1/N_d13);
            [npts,nz]=size(pol_d13_a1);
            zidx=repmat(gpuArray(1:nz),npts,1);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
            d1_ford4_jj(curraindex,:,d4_c)=d1part;
            d3_ford4_jj(curraindex,:,d4_c)=d3part;
            a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
            d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);

            % Maxgap loop
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d1part=rem(pol_d13_a1-1,N_d1)+1;
                    d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    [npts,nz]=size(pol_d13_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d1_ford4_jj(curraindex,:,d4_c)=d1part;
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,n_bothz, d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d1part=rem(pol_d13_a1-1,N_d1)+1;
                    d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    [npts,nz]=size(pol_d13_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d1_ford4_jj(curraindex,:,d4_c)=d1part;
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz(:,:,d4_c));
            d13_with_d4=[repmat(d13_gridvals,1,1),repmat(d4_gridvals(d4_c,:),N_d13,1)];

            EV=EVnext.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_bothz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_bothz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_bothz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a1,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a1,N_u,N_bothz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a1,N_bothz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a1,N_bothz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_bothz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_bothz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_bothz]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                zBindblock=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                DiscountedEVblock=DiscountedEV(:,:,:,:,semizblock);
                d2index_reshblock=d2index_resh(:,:,semizblock);

                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                DEV=reshape(DiscountedEVblock,[1,N_d3,N_a1,1,1,N_semiz]);
                entireRHS_ii=RM+DEV;
                entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz]);

                [~,maxindex1]=max(entireRHS_ii,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford4_jj(curraindex,semizblock,d4_c)=shiftdim(Vtempii,1);
                pol_d13_a1=shiftdim(maxindex2,1);
                d1part=rem(pol_d13_a1-1,N_d1)+1;
                d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                a1primepart=ceil(pol_d13_a1/N_d13);
                [npts,nz]=size(pol_d13_a1);
                zidx=repmat(gpuArray(1:nz),npts,1);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                d1_ford4_jj(curraindex,semizblock,d4_c)=d1part;
                d3_ford4_jj(curraindex,semizblock,d4_c)=d3part;
                a1prime_ford4_jj(curraindex,semizblock,d4_c)=a1primepart;
                d2_ford4_jj(curraindex,semizblock,d4_c)=d2index_reshblock(lin);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBindblock,-2);
                        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVblock(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford4_jj(curraindex,semizblock,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBindblock;
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        [npts,nz]=size(pol_d13_a1);
                        zidx=repmat(gpuArray(1:nz),npts,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                        d1_ford4_jj(curraindex,semizblock,d4_c)=d1part;
                        d3_ford4_jj(curraindex,semizblock,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,semizblock,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,semizblock,d4_c)=d2index_reshblock(lin);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBindblock,-2);
                        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVblock(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford4_jj(curraindex,semizblock,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBindblock;
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        [npts,nz]=size(pol_d13_a1);
                        zidx=repmat(gpuArray(1:nz),npts,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                        d1_ford4_jj(curraindex,semizblock,d4_c)=d1part;
                        d3_ford4_jj(curraindex,semizblock,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,semizblock,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,semizblock,d4_c)=d2index_reshblock(lin);
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz(:,:,d4_c));
            d13_with_d4=[repmat(d13_gridvals,1,1),repmat(d4_gridvals(d4_c,:),N_d13,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

                EV_z=EVnext.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2); % [N_a,1]
                EV_z=reshape(EV_z,[N_a,1]);

                skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
                aprimeProbs=repmat(a2primeProbs,N_a1,1);
                aprimeProbs(skipinterp)=0;
                aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);

                EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
                EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
                EV_z=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2); % [N_d23*N_a1,1]

                EVres=reshape(EV_z,[N_d2,N_d3*N_a1]);
                [EV_onlyd3,d2index]=max(EVres,[],1);
                EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,1]);
                d2index_z=reshape(d2index,[N_d3,N_a1]);

                DiscountedEV_z=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1]);

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,ones(1,length(n_bothz)), d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii_z,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2]);
                DEV=reshape(DiscountedEV_z,[1,N_d3,N_a1,1,1]);
                entireRHS_ii_z=RM+DEV;
                entireRHS_ii_z=reshape(entireRHS_ii_z,[N_d13,N_a1,vfoptions.level1n,N_a2]);

                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                pol_d13_a1=shiftdim(maxindex2,1);
                d1part=rem(pol_d13_a1-1,N_d1)+1;
                d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                a1primepart=ceil(pol_d13_a1/N_d13);
                lin=d3part+N_d3*(a1primepart-1);
                d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);

                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,ones(1,length(n_bothz)), d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d3aprime=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        lin=d3part+N_d3*(a1primepart-1);
                        d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,ones(1,length(n_bothz)), d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d3aprime=d3ind+N_d3*(loweredge-1);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        lin=d3part+N_d3*(a1primepart-1);
                        d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    end
                end
            end
        end
    end

    % Now max over d4
    [Vbest,d4winner]=max(V_ford4_jj,[],3); % [N_a,N_bothz]
    V(:,:,jj)=Vbest;
    linidx_d4=(1:1:N_a*N_bothz)'+(N_a*N_bothz)*(reshape(d4winner,[N_a*N_bothz,1])-1);
    Policy(1,:,:,jj)=reshape(d1_ford4_jj(linidx_d4),[1,N_a,N_bothz]);
    Policy(2,:,:,jj)=reshape(d2_ford4_jj(linidx_d4),[1,N_a,N_bothz]);
    Policy(3,:,:,jj)=reshape(d3_ford4_jj(linidx_d4),[1,N_a,N_bothz]);
    Policy(4,:,:,jj)=reshape(d4winner,[1,N_a,N_bothz]);
    Policy(5,:,:,jj)=reshape(a1prime_ford4_jj(linidx_d4),[1,N_a,N_bothz]);
end


end
