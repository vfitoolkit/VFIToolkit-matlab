function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_noz_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% No z; e iid.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
special_n_d4=ones(1,length(n_d4));
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);
N_u=prod(n_u);

N_d13=N_d1*N_d3;

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz,N_e,N_j,'gpuArray');
d2Policy=ones(1,N_a,N_semiz,N_e,N_j,'gpuArray');
d4Policy=ones(1,N_a,N_semiz,N_e,N_j,'gpuArray');

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
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

pi_u_col=pi_u(:);

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
d3ind=repelem(gpuArray(1:1:N_d3)',N_d1,1);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_semiz=ones(1,length(n_semiz));
end

V_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
Policy3_ford4_jj=zeros(3,N_a,N_semiz,N_e,N_d4,'gpuArray');
flag_ford4_jj=2*ones(N_a,N_semiz,N_e,N_d4,'gpuArray');
d2_ford4_jj=ones(N_a,N_semiz,N_e,N_d4,'gpuArray');


%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, n_e, d1d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,:,N_j)=shiftdim(Vtemp,1);
    dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
    d1d3_ind=rem(dindex-1,N_d13)+1;
    d1part=rem(d1d3_ind-1,N_d1)+1;
    d3part=ceil(d1d3_ind/N_d1);
    d4part=ceil(dindex/N_d13);
    a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
    Policy3(1,:,:,:,N_j)=shiftdim(d1part+N_d1*(d3part-1),-1);
    Policy3(2,:,:,:,N_j)=shiftdim(a1primepart,-1);
    Policy3(3,:,:,:,N_j)=n2short+2;
    d4Policy(1,:,:,:,N_j)=shiftdim(d4part,-1);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]);
    EVpre=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j),-2),3);
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end
    semiz_gridvals=semiz_gridvals_J(:,:,N_j);
    e_gridvals=e_gridvals_J(:,:,N_j);

    V_jj=zeros(N_a,N_semiz,N_e,'gpuArray');
    Policy3_jj=zeros(3,N_a,N_semiz,N_e,'gpuArray');
    PolicyL2flag_jj=2*ones(1,N_a,N_semiz,N_e,'gpuArray');
    d2Policy_jj=ones(1,N_a,N_semiz,N_e,'gpuArray');
    d4Policy_jj=ones(1,N_a,N_semiz,N_e,'gpuArray');

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    for d4_c=1:N_d4
        pi_semizd4=pi_semiz(:,:,d4_c);
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        % EV / d2index / DiscountedEV are independent of e — compute once per d4
        EV=EVpre.*shiftdim(pi_semizd4',-1);
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
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

        if vfoptions.lowmemory==0
            midpoint=zeros(N_d13,1,N_a1,N_a2,N_semiz,N_e,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_semiz,1]);
            entireRHS_ii=RM+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz,n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,2,0);
            da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBind,-2);
            entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz,N_e])+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_semiz,N_e]),[N_d13*n2long,N_a1*N_a2,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zBind+N_d13*N_a*N_semiz*eBind;
            Policy3_ford4_jj(1,:,:,:,d4_c)=d_ind;
            Policy3_ford4_jj(2,:,:,:,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford4_jj(3,:,:,:,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

            L2offset      = ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBind + N_d13*n2long*N_a*N_semiz*eBind;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind + N_d13*n2long*N_a*N_semiz*eBind;
            ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz,N_e]);
            isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
            a1mid=squeeze(midpoint(allind));
            zidx=repmat(gpuArray(reshape(1:N_semiz,[1,N_semiz,1])),N_a,1,N_e);
            linlookup=d3part+N_d3*(a1mid-1)+N_d3*N_a1*(zidx-1);
            d2_ford4_jj(:,:,:,d4_c)=d2index_resh(linlookup);

        elseif vfoptions.lowmemory==1
            % Loop over e inside d4 to reduce memory footprint
            special_n_e=ones(1,length(n_e));
            for e_c=1:N_e
                e_val=e_gridvals(e_c,:);
                midpoint=zeros(N_d13,1,N_a1,N_a2,N_semiz,'gpuArray');

                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii_e,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_semiz]);
                entireRHS_ii_e=RM+DEV;
                entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz]);

                [~,maxindex1]=max(entireRHS_ii_e,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                        entireRHS_ii_e=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                        [~,maxindex]=max(entireRHS_ii_e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,2,0);
                da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBind,-2);
                entireRHS_ii_e=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz])+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_semiz]),[N_d13*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind+N_d13*N_a*zBind;
                Policy3_ford4_jj(1,:,:,e_c,d4_c)=d_ind;
                Policy3_ford4_jj(2,:,:,e_c,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford4_jj(3,:,:,e_c,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                L2offset      = ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
                ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz]);
                isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
                a1mid=squeeze(midpoint(allind));
                zidx=repmat(gpuArray(1:N_semiz),N_a,1);
                linlookup=d3part+N_d3*(a1mid-1)+N_d3*N_a1*(zidx-1);
                d2_ford4_jj(:,:,e_c,d4_c)=d2index_resh(linlookup);
            end
        elseif vfoptions.lowmemory==2
            % Loop over semiz (outer) and e (inner) to reduce memory footprint
            special_n_e=ones(1,length(n_e));
            for z_c=1:N_semiz
                semiz_val=semiz_gridvals(z_c,:);
                DiscountedEV_zc=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_zc=DiscountedEVinterp(:,:,:,:,z_c);
                d2index_resh_zc=d2index_resh(:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    midpoint=zeros(N_d13,1,N_a1,N_a2,1,'gpuArray');

                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,1,0);
                    RM=reshape(ReturnMatrix_ii_e,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,1]);
                    DEV=reshape(DiscountedEV_zc,[1,N_d3,N_a1,1,1,1]);
                    entireRHS_ii_e=RM+DEV;
                    entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,1]);

                    [~,maxindex1]=max(entireRHS_ii_e,[],2);
                    midpoint(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,3,0);
                            d3aprimez=d3ind+N_d3*(a1primeindexes-1);
                            entireRHS_ii_e=ReturnMatrix_ii+DiscountedEV_zc(d3aprimez);
                            [~,maxindex]=max(entireRHS_ii_e,[],2);
                            midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_semiz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,2,0);
                    da1primez=d3ind+N_d3*(a1primeindexesfine-1);
                    entireRHS_ii_e=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,1])+reshape(DiscountedEVinterp_zc(da1primez),[N_d13,n2long,N_a1,N_a2,1]),[N_d13*n2long,N_a1*N_a2,1]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d13)+1;
                    allind=d_ind+N_d13*aind;
                    Policy3_ford4_jj(1,:,z_c,e_c,d4_c)=d_ind;
                    Policy3_ford4_jj(2,:,z_c,e_c,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford4_jj(3,:,z_c,e_c,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                    L2offset      = ceil(maxindexL2/N_d13);
                    linidx_lower  = d_ind                    + N_d13*n2long*aind;
                    linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                    ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,1]);
                    isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                    d3part=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                    a1mid=midpoint(allind);
                    linlookup=d3part+N_d3*(a1mid-1);
                    d2_ford4_jj(:,z_c,e_c,d4_c)=d2index_resh_zc(linlookup);
                end
            end
        end
    end

    % Cross-d4 max
    [V_jj,d4winner]=max(V_ford4_jj,[],4);
    N=N_a*N_semiz*N_e;
    P1=reshape(Policy3_ford4_jj(1,:,:,:,:),[N,N_d4]);
    P2=reshape(Policy3_ford4_jj(2,:,:,:,:),[N,N_d4]);
    P3=reshape(Policy3_ford4_jj(3,:,:,:,:),[N,N_d4]);
    F =reshape(flag_ford4_jj,[N,N_d4]);
    D2=reshape(d2_ford4_jj,[N,N_d4]);
    rowidx=(1:1:N)';
    gather_idx=rowidx+N*(reshape(d4winner,[N,1])-1);
    Policy3_jj(1,:,:,:)=shiftdim(reshape(P1(gather_idx),[N_a,N_semiz,N_e]),-1);
    Policy3_jj(2,:,:,:)=shiftdim(reshape(P2(gather_idx),[N_a,N_semiz,N_e]),-1);
    Policy3_jj(3,:,:,:)=shiftdim(reshape(P3(gather_idx),[N_a,N_semiz,N_e]),-1);
    PolicyL2flag_jj(1,:,:,:)=shiftdim(reshape(F(gather_idx),[N_a,N_semiz,N_e]),-1);
    d2Policy_jj(1,:,:,:)=shiftdim(reshape(D2(gather_idx),[N_a,N_semiz,N_e]),-1);
    d4Policy_jj(1,:,:,:)=shiftdim(d4winner,-1);

    V(:,:,:,N_j)=V_jj;
    Policy3(:,:,:,:,N_j)=Policy3_jj;
    PolicyL2flag(:,:,:,:,N_j)=PolicyL2flag_jj;
    d2Policy(:,:,:,:,N_j)=d2Policy_jj;
    d4Policy(:,:,:,:,N_j)=d4Policy_jj;
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

    EVnext=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end
    semiz_gridvals=semiz_gridvals_J(:,:,jj);
    e_gridvals=e_gridvals_J(:,:,jj);

    V_jj=zeros(N_a,N_semiz,N_e,'gpuArray');
    Policy3_jj=zeros(3,N_a,N_semiz,N_e,'gpuArray');
    PolicyL2flag_jj=2*ones(1,N_a,N_semiz,N_e,'gpuArray');
    d2Policy_jj=ones(1,N_a,N_semiz,N_e,'gpuArray');
    d4Policy_jj=ones(1,N_a,N_semiz,N_e,'gpuArray');

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    for d4_c=1:N_d4
        pi_semizd4=pi_semiz(:,:,d4_c);
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        % EV / d2index / DiscountedEV are independent of e — compute once per d4
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
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

        if vfoptions.lowmemory==0
            midpoint=zeros(N_d13,1,N_a1,N_a2,N_semiz,N_e,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_semiz,1]);
            entireRHS_ii=RM+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz,n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,2,0);
            da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBind,-2);
            entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz,N_e])+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_semiz,N_e]),[N_d13*n2long,N_a1*N_a2,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zBind+N_d13*N_a*N_semiz*eBind;
            Policy3_ford4_jj(1,:,:,:,d4_c)=d_ind;
            Policy3_ford4_jj(2,:,:,:,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford4_jj(3,:,:,:,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

            L2offset      = ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBind + N_d13*n2long*N_a*N_semiz*eBind;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind + N_d13*n2long*N_a*N_semiz*eBind;
            ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz,N_e]);
            isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
            a1mid=squeeze(midpoint(allind));
            zidx=repmat(gpuArray(reshape(1:N_semiz,[1,N_semiz,1])),N_a,1,N_e);
            linlookup=d3part+N_d3*(a1mid-1)+N_d3*N_a1*(zidx-1);
            d2_ford4_jj(:,:,:,d4_c)=d2index_resh(linlookup);

        elseif vfoptions.lowmemory==1
            % Loop over e inside d4 to reduce memory footprint
            special_n_e=ones(1,length(n_e));
            for e_c=1:N_e
                e_val=e_gridvals(e_c,:);
                midpoint=zeros(N_d13,1,N_a1,N_a2,N_semiz,'gpuArray');

                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii_e,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_semiz]);
                entireRHS_ii_e=RM+DEV;
                entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz]);

                [~,maxindex1]=max(entireRHS_ii_e,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                        entireRHS_ii_e=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                        [~,maxindex]=max(entireRHS_ii_e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,2,0);
                da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBind,-2);
                entireRHS_ii_e=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz])+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_semiz]),[N_d13*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind+N_d13*N_a*zBind;
                Policy3_ford4_jj(1,:,:,e_c,d4_c)=d_ind;
                Policy3_ford4_jj(2,:,:,e_c,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford4_jj(3,:,:,e_c,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                L2offset      = ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
                ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz]);
                isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
                a1mid=squeeze(midpoint(allind));
                zidx=repmat(gpuArray(1:N_semiz),N_a,1);
                linlookup=d3part+N_d3*(a1mid-1)+N_d3*N_a1*(zidx-1);
                d2_ford4_jj(:,:,e_c,d4_c)=d2index_resh(linlookup);
            end
        elseif vfoptions.lowmemory==2
            % Loop over semiz (outer) and e (inner) to reduce memory footprint
            special_n_e=ones(1,length(n_e));
            for z_c=1:N_semiz
                semiz_val=semiz_gridvals(z_c,:);
                DiscountedEV_zc=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_zc=DiscountedEVinterp(:,:,:,:,z_c);
                d2index_resh_zc=d2index_resh(:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    midpoint=zeros(N_d13,1,N_a1,N_a2,1,'gpuArray');

                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,1,0);
                    RM=reshape(ReturnMatrix_ii_e,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,1]);
                    DEV=reshape(DiscountedEV_zc,[1,N_d3,N_a1,1,1,1]);
                    entireRHS_ii_e=RM+DEV;
                    entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,1]);

                    [~,maxindex1]=max(entireRHS_ii_e,[],2);
                    midpoint(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,3,0);
                            d3aprimez=d3ind+N_d3*(a1primeindexes-1);
                            entireRHS_ii_e=ReturnMatrix_ii+DiscountedEV_zc(d3aprimez);
                            [~,maxindex]=max(entireRHS_ii_e,[],2);
                            midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_semiz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,2,0);
                    da1primez=d3ind+N_d3*(a1primeindexesfine-1);
                    entireRHS_ii_e=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,1])+reshape(DiscountedEVinterp_zc(da1primez),[N_d13,n2long,N_a1,N_a2,1]),[N_d13*n2long,N_a1*N_a2,1]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d13)+1;
                    allind=d_ind+N_d13*aind;
                    Policy3_ford4_jj(1,:,z_c,e_c,d4_c)=d_ind;
                    Policy3_ford4_jj(2,:,z_c,e_c,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford4_jj(3,:,z_c,e_c,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                    L2offset      = ceil(maxindexL2/N_d13);
                    linidx_lower  = d_ind                    + N_d13*n2long*aind;
                    linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                    ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,1]);
                    isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                    d3part=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                    a1mid=midpoint(allind);
                    linlookup=d3part+N_d3*(a1mid-1);
                    d2_ford4_jj(:,z_c,e_c,d4_c)=d2index_resh_zc(linlookup);
                end
            end
        end
    end

    % Cross-d4 max
    [V_jj,d4winner]=max(V_ford4_jj,[],4);
    N=N_a*N_semiz*N_e;
    P1=reshape(Policy3_ford4_jj(1,:,:,:,:),[N,N_d4]);
    P2=reshape(Policy3_ford4_jj(2,:,:,:,:),[N,N_d4]);
    P3=reshape(Policy3_ford4_jj(3,:,:,:,:),[N,N_d4]);
    F =reshape(flag_ford4_jj,[N,N_d4]);
    D2=reshape(d2_ford4_jj,[N,N_d4]);
    rowidx=(1:1:N)';
    gather_idx=rowidx+N*(reshape(d4winner,[N,1])-1);
    Policy3_jj(1,:,:,:)=shiftdim(reshape(P1(gather_idx),[N_a,N_semiz,N_e]),-1);
    Policy3_jj(2,:,:,:)=shiftdim(reshape(P2(gather_idx),[N_a,N_semiz,N_e]),-1);
    Policy3_jj(3,:,:,:)=shiftdim(reshape(P3(gather_idx),[N_a,N_semiz,N_e]),-1);
    PolicyL2flag_jj(1,:,:,:)=shiftdim(reshape(F(gather_idx),[N_a,N_semiz,N_e]),-1);
    d2Policy_jj(1,:,:,:)=shiftdim(reshape(D2(gather_idx),[N_a,N_semiz,N_e]),-1);
    d4Policy_jj(1,:,:,:)=shiftdim(d4winner,-1);

    V(:,:,:,jj)=V_jj;
    Policy3(:,:,:,:,jj)=Policy3_jj;
    PolicyL2flag(:,:,:,:,jj)=PolicyL2flag_jj;
    d2Policy(:,:,:,:,jj)=d2Policy_jj;
    d4Policy(:,:,:,:,jj)=d4Policy_jj;
end


%% Switch Policy3(2,:) from 'midpoint' to 'lower grid index'
adjust=(Policy3(3,:,:,:,:)<1+n2short+1);
Policy3(2,:,:,:,:)=Policy3(2,:,:,:,:)-adjust;
Policy3(3,:,:,:,:)=adjust.*Policy3(3,:,:,:,:)+(1-adjust).*(Policy3(3,:,:,:,:)-n2short-1);

%% Encode Policy as component rows (with d1, no z, with e)
Policy=zeros(7,N_a,N_semiz,N_e,N_j,'gpuArray');
d13=Policy3(1,:,:,:,:);
Policy(1,:,:,:,:)=rem(d13-1,N_d1)+1;
Policy(2,:,:,:,:)=d2Policy;
Policy(3,:,:,:,:)=rem(ceil(d13/N_d1)-1,N_d3)+1;
Policy(4,:,:,:,:)=d4Policy;
Policy(5,:,:,:,:)=Policy3(2,:,:,:,:);
Policy(6,:,:,:,:)=Policy3(3,:,:,:,:);
Policy(7,:,:,:,:)=PolicyL2flag;

end
