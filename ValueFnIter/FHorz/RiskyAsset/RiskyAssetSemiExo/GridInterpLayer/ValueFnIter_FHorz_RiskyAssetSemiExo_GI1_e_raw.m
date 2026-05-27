function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% e: iid start-of-period shock (integrated out of EV before d2 refine)

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
N_d=N_d1*N_d2*N_d3;

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

special_n_d4=ones(1,length(n_d4)); %#ok<NASGU>
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

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

pi_u_col=pi_u(:);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_bothz-1),-3);
zindB=shiftdim(gpuArray(0:1:N_bothz-1),-1);
zeindB=zindB+N_bothz*shiftdim((0:1:N_e-1),-2);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);

V_ford4_jj=zeros(N_a,N_bothz,N_e,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_bothz,N_e,N_d4,'gpuArray');
flag_ford4_jj=2*ones(N_a,N_bothz,N_e,N_d4,'gpuArray');
d2index_ford4_jj=ones(N_a,N_bothz,N_e,N_d4,'gpuArray');


%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,n_bothz,n_e, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            [~,maxindex_d4]=max(ReturnMatrix_d4,[],2);

            midpoint_d4=max(min(maxindex_d4,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_d4+(midpoint_d4-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_bothz,n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zeindB;
            mid_at=shiftdim(squeeze(midpoint_d4(allind)),-1);
            L2offset=shiftdim(ceil(maxindexL2/N_d13),-1);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zeindB;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zeindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
            d2index_ford4_jj(:,:,:,d4_c)=1;
        end
    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d4e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,n_bothz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
                [~,maxindex_d4e]=max(ReturnMatrix_d4e,[],2);

                midpoint_d4e=max(min(maxindex_d4e,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_d4e+(midpoint_d4e-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_bothz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
                mid_at=shiftdim(squeeze(midpoint_d4e(allind)),-1);
                L2offset=shiftdim(ceil(maxindexL2/N_d13),-1);
                linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
                d2index_ford4_jj(:,:,e_c,d4_c)=1;
            end
        end
    elseif vfoptions.lowmemory==2
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                    [~,maxindex_ze]=max(ReturnMatrix_ze,[],2);

                    midpoint_ze=max(min(maxindex_ze,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_ze+(midpoint_ze-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,special_n_bothz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d13)+1;
                    allind=d_ind+N_d13*aind;
                    mid_at=midpoint_ze(allind);
                    L2offset=ceil(maxindexL2/N_d13);
                    linidx_lower  = d_ind                    + N_d13*n2long*aind;
                    linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                    Policy_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
                    d2index_ford4_jj(:,z_c,e_c,d4_c)=1;
                end
            end
        end
    end
    [V(:,:,:,N_j),Policy(:,:,:,N_j),~]=combine_across_d4_e(V_ford4_jj,Policy_ford4_jj,d2index_ford4_jj,flag_ford4_jj,N_a,N_bothz,N_e,N_d4,N_d1,N_d2,N_d3,N_d13,N_d,N_a1,n2short);
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
        n_d1,n_d3,n_a1,n_a2,n_bothz,n_e,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d,N_a1,N_a2,N_a,N_bothz,N_e,N_u,N_a1prime,...
        d13_gridvals,d4_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,bothz_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),pi_z_J(:,:,N_j),pi_semiz,pi_u_col,...
        aind,zind,zindB,zeindB,a2ind,n2short,n2long,V_ford4_jj,Policy_ford4_jj,flag_ford4_jj,d2index_ford4_jj,vfoptions);
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
        n_d1,n_d3,n_a1,n_a2,n_bothz,n_e,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d,N_a1,N_a2,N_a,N_bothz,N_e,N_u,N_a1prime,...
        d13_gridvals,d4_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,bothz_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),pi_z_J(:,:,jj),pi_semiz,pi_u_col,...
        aind,zind,zindB,zeindB,a2ind,n2short,n2long,V_ford4_jj,Policy_ford4_jj,flag_ford4_jj,d2index_ford4_jj,vfoptions);
end


end


%% Per-period inner with e (e already integrated out of EVnext)
function [V_jj,Policy_jj]=internal_per_j_e(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d1,n_d3,n_a1,n_a2,n_bothz,n_e,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d,N_a1,N_a2,N_a,N_bothz_count,N_e_count,N_u,N_a1prime,...
    d13_gridvals,d4_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,bothz_gridvals,e_gridvals,pi_z,pi_semiz,pi_u_col,...
    aind,zind,zindB,zeindB,a2ind,n2short,n2long,V_ford4_jj,Policy_ford4_jj,flag_ford4_jj,d2index_ford4_jj,vfoptions)

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
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
        DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
        DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

        ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,n_bothz,n_e, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec,1,0);

        entireRHS=ReturnMatrix_d4+DiscountedEV_d13; % broadcast a2,e

        [~,maxindex]=max(entireRHS,[],2);

        midpoint=max(min(maxindex,n_a1(1)-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_bothz,n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec,2,0);
        da1primez=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1)+N_d13*N_a1prime*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_d13(da1primez(:)),[N_d13*n2long,N_a1*N_a2,N_bothz_count,N_e_count]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d13)+1;
        allind=d_ind+N_d13*aind+N_d13*N_a*zeindB;
        mid_at=shiftdim(squeeze(midpoint(allind)),-1);
        L2offset=shiftdim(ceil(maxindexL2/N_d13),-1);
        linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zeindB;
        linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zeindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        flag_ford4_jj(:,:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        Policy_ford4_jj(:,:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
        % d2 lookup: d2index_resh is [N_d3,N_a1,N_bothz] (no e dep)
        d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a,N_bothz,N_e]
        a1opt_mid=midpoint(allind); % [1,N_a,N_bothz,N_e]
        zlin=shiftdim(gpuArray(0:N_bothz_count-1),-1);
        lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
        d2index_ford4_jj(:,:,:,d4_c)=shiftdim(d2index_resh(lin),1);
    end

elseif vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
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
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
        DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
        DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

        for e_c=1:N_e_count
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,n_bothz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_val, ReturnFnParamsVec,1,0);
            entireRHS_e=ReturnMatrix_e+DiscountedEV_d13;
            [~,maxindex]=max(entireRHS_e,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_bothz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals, e_val, ReturnFnParamsVec,2,0);
            da1primez=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1)+N_d13*N_a1prime*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_d13(da1primez(:)),[N_d13*n2long,N_a1*N_a2,N_bothz_count]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
            mid_at=shiftdim(squeeze(midpoint(allind)),-1);
            L2offset=shiftdim(ceil(maxindexL2/N_d13),-1);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
            d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a,N_bothz]
            a1opt_mid=midpoint(allind);
            zlin=shiftdim(gpuArray(0:N_bothz_count-1),-1);
            lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
            d2index_ford4_jj(:,:,e_c,d4_c)=shiftdim(d2index_resh(lin),1);
        end
    end
end

[V_jj,Policy_jj,~]=combine_across_d4_e(V_ford4_jj,Policy_ford4_jj,d2index_ford4_jj,flag_ford4_jj,N_a,N_bothz_count,N_e_count,N_d4,N_d1,N_d2,N_d3,N_d13,N_d,N_a1,n2short);

end


%% Cross-d4 max + final encoding (with e dim)
function [V_jj,Policy_jj,d4winner]=combine_across_d4_e(V_ford4,Policy_ford4,d2idx_ford4,flag_ford4,N_a,N_bothz,N_e,N_d4,N_d1,N_d2,N_d3,N_d13,N_d,N_a1,n2short)
% Inputs are [N_a,N_bothz,N_e,N_d4]; max over dim 4
[V_jj,d4winner]=max(V_ford4,[],4);
N=N_a*N_bothz*N_e;
linidx=(1:1:N)'+N*(reshape(d4winner,[N,1])-1);
polenc=reshape(Policy_ford4(linidx),[N_a,N_bothz,N_e]);
d2winner=reshape(d2idx_ford4(linidx),[N_a,N_bothz,N_e]);
flagwinner=reshape(flag_ford4(linidx),[N_a,N_bothz,N_e]);

d13part=rem(polenc-1,N_d13)+1;
tmp=ceil(polenc/N_d13);
midpart=rem(tmp-1,N_a1)+1;
L2offset=ceil(tmp/N_a1);

adjust=(L2offset<1+n2short+1);
a1prime_low=midpart-adjust;
L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);

d1part=rem(d13part-1,N_d1)+1;
d3part=rem(ceil(d13part/N_d1)-1,N_d3)+1;
d2part=d2winner;
d4part=d4winner;

Policy_jj=d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d*(d4part-1)+N_d*N_d4*(a1prime_low-1)+N_d*N_d4*N_a1*(L2ind-1)+N_d*N_d4*N_a1*(n2short+2)*(flagwinner-1);

end
