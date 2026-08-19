function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC2A_nod1_noz_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_a3,n_semiz,n_u,N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, a3_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_nod1_noz_raw.
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% No d1, no z.
%
% a1: standard endogenous state, this is the one divide-and-conquer is applied to
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% The EV pipeline is unchanged from the DC1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset.
% Policy output has the choices on the first dimension: (d2,d3,d4,a1prime,a2prime).

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_u=prod(n_u);

N_a12=N_a1*N_a2; % the two standard assets, carried forward directly

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy=zeros(5,N_a,N_semiz,N_j,'gpuArray'); % d2, d3, d4, a1prime, a2prime

%%
u_grid=gpuArray(u_grid);
a3_grid=gpuArray(a3_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
d23_grid=gpuArray(d23_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d3_gridvals=gpuArray(CreateGridvals(n_d3,d3_grid,1));
d3d4a1a2_gridvals=gpuArray(CreateGridvals([n_d3,n_d4,n_a1,n_a2],[d3_grid;d4_grid;a1_grid;a2_grid],1));
a1a2a3_gridvals=gpuArray(CreateGridvals([n_a1,n_a2,n_a3],[a1_grid;a2_grid;a3_grid],1));

pi_u_col=pi_u(:);

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2Bind=gpuArray(0:1:N_a2-1);
a3Bind=gpuArray(0:1:N_a3-1);
zBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
d3ind=(1:1:N_d3)';
a2ind=gpuArray(0:1:N_a2-1)';
a3ind=gpuArray(0:1:N_a3-1)';
a2pcol=reshape(0:1:N_a2-1,[1,1,N_a2]); % [1,1,N_a2prime]

% Accumulators for the choice of d4 (max is taken across d4 at the end of each period)
V_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
d2_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
d3_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
a1a2prime_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_d4,n_a1,n_a2], [n_a1,n_a2,n_a3], n_semiz, d3d4a1a2_gridvals, a1a2a3_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=shiftdim(Vtemp,1);
        dindex=rem(maxindex-1,N_d3*N_d4)+1;
        Policy(1,:,:,N_j)=1; % d2 is meaningless in the terminal period
        Policy(2,:,:,N_j)=rem(dindex-1,N_d3)+1; % d3
        Policy(3,:,:,N_j)=ceil(dindex/N_d3); % d4
        a1a2primepart=ceil(maxindex/(N_d3*N_d4));
        Policy(4,:,:,N_j)=rem(a1a2primepart-1,N_a1)+1; % a1prime
        Policy(5,:,:,N_j)=floor((a1a2primepart-1)/N_a1)+1; % a2prime
    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_d4,n_a1,n_a2], [n_a1,n_a2,n_a3], special_n_semiz, d3d4a1a2_gridvals, a1a2a3_gridvals, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtemp,1);
            dindex=rem(maxindex-1,N_d3*N_d4)+1;
            Policy(1,:,z_c,N_j)=1; % d2 is meaningless in the terminal period
            Policy(2,:,z_c,N_j)=rem(dindex-1,N_d3)+1; % d3
            Policy(3,:,z_c,N_j)=ceil(dindex/N_d3); % d4
            a1a2primepart=ceil(maxindex/(N_d3*N_d4));
            Policy(4,:,z_c,N_j)=rem(a1a2primepart-1,N_a1)+1; % a1prime
            Policy(5,:,z_c,N_j)=floor((a1a2primepart-1)/N_a1)+1; % a2prime
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    EVnext=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    aprimeIndex=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

            EV=EVnext.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a3primeProbs,N_a12,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a12,N_semiz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a12,N_semiz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_semiz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_semiz]);

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,n_semiz, d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            % [N_d3, N_a1prime, N_a2prime, level1n, N_a2, N_a3, N_semiz]
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d3*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
            pol_d3_a1a2=shiftdim(maxindex2,1);
            d3part     =rem(pol_d3_a1a2-1,N_d3)+1;
            a1primepart=rem(floor((pol_d3_a1a2-1)/N_d3),N_a1)+1;
            a2primepart=floor((pol_d3_a1a2-1)/(N_d3*N_a1))+1;
            zidx=repmat(gpuArray(1:N_semiz),size(pol_d3_a1a2,1),1);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*(zidx-1);
            d3_ford4_jj(curraindex,:,d4_c)=d3part;
            a1a2prime_ford4_jj(curraindex,:,d4_c)=a1primepart+N_a1*(a2primepart-1);
            d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                a2Bind_flat=repmat(repelem(a2Bind,1,level1iidiff(ii)),1,N_a3);
                a3Bind_flat=repelem(a3Bind,1,level1iidiff(ii)*N_a2);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % [N_d3,1,N_a2prime,1,N_a2,N_a3,N_semiz]
                    a1primeindexes=loweredge+(0:1:maxgap(ii));                % [N_d3,maxgap+1,N_a2prime,1,N_a2,N_a3,N_semiz]
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,n_semiz, d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*a2pcol+N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    d3part     =rem(maxindex-1,N_d3)+1;
                    a1localind =rem(floor((maxindex-1)/N_d3),maxgap(ii)+1)+1;
                    a2primepart=floor((maxindex-1)/(N_d3*(maxgap(ii)+1)))+1;
                    allind=d3part+N_d3*(a2primepart-1)+N_d3*N_a2*a2Bind_flat+N_d3*N_a2*N_a2*a3Bind_flat+N_d3*N_a2*N_a2*N_a3*zBind;
                    a1primepart=a1localind+loweredge(allind)-1;
                    d3part=shiftdim(d3part,1);
                    a1primepart=shiftdim(a1primepart,1);
                    a2primepart=shiftdim(a2primepart,1);
                    zidx=repmat(gpuArray(1:N_semiz),size(d3part,1),1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*(zidx-1);
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1a2prime_ford4_jj(curraindex,:,d4_c)=a1primepart+N_a1*(a2primepart-1);
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,n_semiz, d3_with_d4, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*a2pcol+N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    d3part     =rem(maxindex-1,N_d3)+1;
                    a2primepart=floor((maxindex-1)/N_d3)+1;
                    allind=d3part+N_d3*(a2primepart-1)+N_d3*N_a2*a2Bind_flat+N_d3*N_a2*N_a2*a3Bind_flat+N_d3*N_a2*N_a2*N_a3*zBind;
                    a1primepart=loweredge(allind);
                    d3part=shiftdim(d3part,1);
                    a1primepart=shiftdim(a1primepart,1);
                    a2primepart=shiftdim(a2primepart,1);
                    zidx=repmat(gpuArray(1:N_semiz),size(d3part,1),1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*(zidx-1);
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1a2prime_ford4_jj(curraindex,:,d4_c)=a1primepart+N_a1*(a2primepart-1);
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);

                EV_z=EVnext.*pi_semizd4(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);
                EV_z=reshape(EV_z,[N_a,1]);

                skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
                aprimeProbs=repmat(a3primeProbs,N_a12,1);
                aprimeProbs(skipinterp)=0;
                aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u]);

                EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a12,N_u]).*aprimeProbs;
                EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a12,N_u]).*(1-aprimeProbs);
                EV_z=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);

                EVres=reshape(EV_z,[N_d2,N_d3*N_a12]);
                [EV_onlyd3,d2index]=max(EVres,[],1);
                EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,1]);
                d2index_z=reshape(d2index,[N_d3,N_a1,N_a2]);

                DiscountedEV_z=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1]);

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,ones(1,length(n_semiz)), d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec,1);
                entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z;

                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d3*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                pol_d3_a1a2=shiftdim(maxindex2,1);
                d3part     =rem(pol_d3_a1a2-1,N_d3)+1;
                a1primepart=rem(floor((pol_d3_a1a2-1)/N_d3),N_a1)+1;
                a2primepart=floor((pol_d3_a1a2-1)/(N_d3*N_a1))+1;
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                a1a2prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart+N_a1*(a2primepart-1);
                d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    a2Bind_flat=repmat(repelem(a2Bind,1,level1iidiff(ii)),1,N_a3);
                    a3Bind_flat=repelem(a3Bind,1,level1iidiff(ii)*N_a2);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % [N_d3,1,N_a2prime,1,N_a2,N_a3]
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,ones(1,length(n_semiz)), d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec,3);
                        d3aprime=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*a2pcol;
                        entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d3*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        d3part     =rem(maxindex-1,N_d3)+1;
                        a1localind =rem(floor((maxindex-1)/N_d3),maxgap(ii)+1)+1;
                        a2primepart=floor((maxindex-1)/(N_d3*(maxgap(ii)+1)))+1;
                        allind=d3part+N_d3*(a2primepart-1)+N_d3*N_a2*a2Bind_flat+N_d3*N_a2*N_a2*a3Bind_flat;
                        a1primepart=a1localind+loweredge(allind)-1;
                        d3part=shiftdim(d3part,1);
                        a1primepart=shiftdim(a1primepart,1);
                        a2primepart=shiftdim(a2primepart,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1a2prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart+N_a1*(a2primepart-1);
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,ones(1,length(n_semiz)), d3_with_d4, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec,3);
                        d3aprime=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*a2pcol;
                        entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d3*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        d3part     =rem(maxindex-1,N_d3)+1;
                        a2primepart=floor((maxindex-1)/N_d3)+1;
                        allind=d3part+N_d3*(a2primepart-1)+N_d3*N_a2*a2Bind_flat+N_d3*N_a2*N_a2*a3Bind_flat;
                        a1primepart=loweredge(allind);
                        d3part=shiftdim(d3part,1);
                        a1primepart=shiftdim(a1primepart,1);
                        a2primepart=shiftdim(a2primepart,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1a2prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart+N_a1*(a2primepart-1);
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    end
                end
            end
        end
    end

    [Vbest,d4winner]=max(V_ford4_jj,[],3);
    V(:,:,N_j)=Vbest;
    linidx_d4=(1:1:N_a*N_semiz)'+(N_a*N_semiz)*(reshape(d4winner,[N_a*N_semiz,1])-1);
    Policy(1,:,:,N_j)=reshape(d2_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(2,:,:,N_j)=reshape(d3_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(d4winner,[1,N_a,N_semiz]);
    a1a2primepart=reshape(a1a2prime_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=rem(a1a2primepart-1,N_a1)+1; % a1prime
    Policy(5,:,:,N_j)=floor((a1a2primepart-1)/N_a1)+1; % a2prime
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
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);

    EVnext=V(:,:,jj+1);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    aprimeIndex=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

            EV=EVnext.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a3primeProbs,N_a12,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a12,N_semiz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a12,N_semiz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_semiz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_semiz]);

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,n_semiz, d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d3*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
            pol_d3_a1a2=shiftdim(maxindex2,1);
            d3part     =rem(pol_d3_a1a2-1,N_d3)+1;
            a1primepart=rem(floor((pol_d3_a1a2-1)/N_d3),N_a1)+1;
            a2primepart=floor((pol_d3_a1a2-1)/(N_d3*N_a1))+1;
            zidx=repmat(gpuArray(1:N_semiz),size(pol_d3_a1a2,1),1);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*(zidx-1);
            d3_ford4_jj(curraindex,:,d4_c)=d3part;
            a1a2prime_ford4_jj(curraindex,:,d4_c)=a1primepart+N_a1*(a2primepart-1);
            d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                a2Bind_flat=repmat(repelem(a2Bind,1,level1iidiff(ii)),1,N_a3);
                a3Bind_flat=repelem(a3Bind,1,level1iidiff(ii)*N_a2);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,n_semiz, d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*a2pcol+N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    d3part     =rem(maxindex-1,N_d3)+1;
                    a1localind =rem(floor((maxindex-1)/N_d3),maxgap(ii)+1)+1;
                    a2primepart=floor((maxindex-1)/(N_d3*(maxgap(ii)+1)))+1;
                    allind=d3part+N_d3*(a2primepart-1)+N_d3*N_a2*a2Bind_flat+N_d3*N_a2*N_a2*a3Bind_flat+N_d3*N_a2*N_a2*N_a3*zBind;
                    a1primepart=a1localind+loweredge(allind)-1;
                    d3part=shiftdim(d3part,1);
                    a1primepart=shiftdim(a1primepart,1);
                    a2primepart=shiftdim(a2primepart,1);
                    zidx=repmat(gpuArray(1:N_semiz),size(d3part,1),1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*(zidx-1);
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1a2prime_ford4_jj(curraindex,:,d4_c)=a1primepart+N_a1*(a2primepart-1);
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,n_semiz, d3_with_d4, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*a2pcol+N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    d3part     =rem(maxindex-1,N_d3)+1;
                    a2primepart=floor((maxindex-1)/N_d3)+1;
                    allind=d3part+N_d3*(a2primepart-1)+N_d3*N_a2*a2Bind_flat+N_d3*N_a2*N_a2*a3Bind_flat+N_d3*N_a2*N_a2*N_a3*zBind;
                    a1primepart=loweredge(allind);
                    d3part=shiftdim(d3part,1);
                    a1primepart=shiftdim(a1primepart,1);
                    a2primepart=shiftdim(a2primepart,1);
                    zidx=repmat(gpuArray(1:N_semiz),size(d3part,1),1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*(zidx-1);
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1a2prime_ford4_jj(curraindex,:,d4_c)=a1primepart+N_a1*(a2primepart-1);
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);

                EV_z=EVnext.*pi_semizd4(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);
                EV_z=reshape(EV_z,[N_a,1]);

                skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
                aprimeProbs=repmat(a3primeProbs,N_a12,1);
                aprimeProbs(skipinterp)=0;
                aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u]);

                EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a12,N_u]).*aprimeProbs;
                EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a12,N_u]).*(1-aprimeProbs);
                EV_z=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);

                EVres=reshape(EV_z,[N_d2,N_d3*N_a12]);
                [EV_onlyd3,d2index]=max(EVres,[],1);
                EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,1]);
                d2index_z=reshape(d2index,[N_d3,N_a1,N_a2]);

                DiscountedEV_z=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1]);

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,ones(1,length(n_semiz)), d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec,1);
                entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z;

                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d3*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                pol_d3_a1a2=shiftdim(maxindex2,1);
                d3part     =rem(pol_d3_a1a2-1,N_d3)+1;
                a1primepart=rem(floor((pol_d3_a1a2-1)/N_d3),N_a1)+1;
                a2primepart=floor((pol_d3_a1a2-1)/(N_d3*N_a1))+1;
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                a1a2prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart+N_a1*(a2primepart-1);
                d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    a2Bind_flat=repmat(repelem(a2Bind,1,level1iidiff(ii)),1,N_a3);
                    a3Bind_flat=repelem(a3Bind,1,level1iidiff(ii)*N_a2);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,ones(1,length(n_semiz)), d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec,3);
                        d3aprime=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*a2pcol;
                        entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d3*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        d3part     =rem(maxindex-1,N_d3)+1;
                        a1localind =rem(floor((maxindex-1)/N_d3),maxgap(ii)+1)+1;
                        a2primepart=floor((maxindex-1)/(N_d3*(maxgap(ii)+1)))+1;
                        allind=d3part+N_d3*(a2primepart-1)+N_d3*N_a2*a2Bind_flat+N_d3*N_a2*N_a2*a3Bind_flat;
                        a1primepart=a1localind+loweredge(allind)-1;
                        d3part=shiftdim(d3part,1);
                        a1primepart=shiftdim(a1primepart,1);
                        a2primepart=shiftdim(a2primepart,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1a2prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart+N_a1*(a2primepart-1);
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0,[n_d3,special_n_d4],n_a2,n_a3,ones(1,length(n_semiz)), d3_with_d4, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec,3);
                        d3aprime=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*a2pcol;
                        entireRHS_ii_z=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d3*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        d3part     =rem(maxindex-1,N_d3)+1;
                        a2primepart=floor((maxindex-1)/N_d3)+1;
                        allind=d3part+N_d3*(a2primepart-1)+N_d3*N_a2*a2Bind_flat+N_d3*N_a2*N_a2*a3Bind_flat;
                        a1primepart=loweredge(allind);
                        d3part=shiftdim(d3part,1);
                        a1primepart=shiftdim(a1primepart,1);
                        a2primepart=shiftdim(a2primepart,1);
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1a2prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart+N_a1*(a2primepart-1);
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    end
                end
            end
        end
    end

    [Vbest,d4winner]=max(V_ford4_jj,[],3);
    V(:,:,jj)=Vbest;
    linidx_d4=(1:1:N_a*N_semiz)'+(N_a*N_semiz)*(reshape(d4winner,[N_a*N_semiz,1])-1);
    Policy(1,:,:,jj)=reshape(d2_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(2,:,:,jj)=reshape(d3_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(3,:,:,jj)=reshape(d4winner,[1,N_a,N_semiz]);
    a1a2primepart=reshape(a1a2prime_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(4,:,:,jj)=rem(a1a2primepart-1,N_a1)+1; % a1prime
    Policy(5,:,:,jj)=floor((a1a2primepart-1)/N_a1)+1; % a2prime
end


end
