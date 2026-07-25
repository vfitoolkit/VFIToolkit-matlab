function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_nod1_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_e,n_u,N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% No d1, with e.

n_bothz=[n_semiz,n_z];

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

N_d=N_d2*N_d3;

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

special_n_d4=ones(1,length(n_d4)); %#ok<NASGU>
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(6,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
u_grid=gpuArray(u_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
d23_grid=gpuArray(d23_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d3_gridvals=gpuArray(CreateGridvals(n_d3,d3_grid,1));

pi_u_col=pi_u(:);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>=2
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
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];
            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,n_bothz,n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            [~,maxindex_d4]=max(ReturnMatrix_d4,[],2);

            midpoint_d4=max(min(maxindex_d4,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_d4+(midpoint_d4-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_bothz,n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d_ind+N_d3*aind+N_d3*N_a*zeindB;
            mid_at=shiftdim(squeeze(midpoint_d4(allind)),-1);
            L2offset=ceil(maxindexL2/N_d3);
            linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
            linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
            d2index_ford4_jj(:,:,:,d4_c)=1;
        end
    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d4e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,n_bothz,special_n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
                [~,maxindex_d4e]=max(ReturnMatrix_d4e,[],2);

                midpoint_d4e=max(min(maxindex_d4e,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_d4e+(midpoint_d4e-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_bothz,special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d3)+1;
                allind=d_ind+N_d3*aind+N_d3*N_a*zindB;
                mid_at=shiftdim(squeeze(midpoint_d4e(allind)),-1);
                L2offset=ceil(maxindexL2/N_d3);
                linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
                linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
                d2index_ford4_jj(:,:,e_c,d4_c)=1;
            end
        end
    elseif vfoptions.lowmemory==2
        for d4_c=1:N_d4
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d4e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);
                    [~,maxindex_d4e]=max(ReturnMatrix_d4e,[],2);

                    midpoint_d4e=max(min(maxindex_d4e,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_d4e+(midpoint_d4e-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    V_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d3)+1;
                    allind=d_ind+N_d3*aind+N_d3*N_a*semizBind;
                    mid_at=shiftdim(squeeze(midpoint_d4e(allind)),-1);
                    L2offset=ceil(maxindexL2/N_d3);
                    linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*semizBind;
                    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*semizBind;
                    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                    Policy_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
                    d2index_ford4_jj(:,semizblock,e_c,d4_c)=1;
                end
            end
        end
    elseif vfoptions.lowmemory==3
        for d4_c=1:N_d4
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                    [~,maxindex_ze]=max(ReturnMatrix_ze,[],2);

                    midpoint_ze=max(min(maxindex_ze,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_ze+(midpoint_ze-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d3)+1;
                    allind=d_ind+N_d3*aind;
                    mid_at=midpoint_ze(allind);
                    L2offset=ceil(maxindexL2/N_d3);
                    linidx_lower  = d_ind                   + N_d3*n2long*aind;
                    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
                    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                    Policy_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
                    d2index_ford4_jj(:,z_c,e_c,d4_c)=1;
                end
            end
        end
    end
    [Vbest,d4winner]=max(V_ford4_jj,[],4);
    V(:,:,:,N_j)=Vbest;
    Ncomb=N_a*N_bothz*N_e;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_bothz,N_e]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_bothz,N_e]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_bothz,N_e]);
    d3part=rem(polenc-1,N_d3)+1;
    tmp=ceil(polenc/N_d3);
    midpart=rem(tmp-1,N_a1)+1;
    L2offset=ceil(tmp/N_a1);
    adjust=(L2offset<1+n2short+1);
    a1prime_low=midpart-adjust;
    L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);
    Policy(1,:,:,:,N_j)=reshape(d2winner,[1,N_a,N_bothz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(d3part,[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(d4winner,[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(a1prime_low,[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(L2ind,[1,N_a,N_bothz,N_e]);
    Policy(6,:,:,:,N_j)=reshape(flagwinner,[1,N_a,N_bothz,N_e]);
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

    bothz_gridvals=bothz_gridvals_J(:,:,N_j);
    e_gridvals=e_gridvals_J(:,:,N_j);
    pi_z=pi_z_J(:,:,N_j);
    EVnext=EVpre;

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

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
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,n_bothz,n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec,1,0);

            entireRHS=ReturnMatrix_d4+DiscountedEV; % broadcast a2,e

            [~,maxindex]=max(entireRHS,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_bothz,n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec,2,0);
            da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d_ind+N_d3*aind+N_d3*N_a*zeindB;
            mid_at=shiftdim(squeeze(midpoint(allind)),-1);
            L2offset=ceil(maxindexL2/N_d3);
            linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
            linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
            d3opt=d_ind;
            a1opt_mid=midpoint(allind);
            zlin=shiftdim(gpuArray(0:N_bothz-1),-1);
            lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
            d2index_ford4_jj(:,:,:,d4_c)=shiftdim(d2index_resh(lin),1);
        end

    elseif vfoptions.lowmemory==1
        special_n_e=ones(1,length(n_e));
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

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
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for e_c=1:N_e
                e_val=e_gridvals(e_c,:);
                ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,n_bothz,special_n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_val, ReturnFnParamsVec,1,0);
                entireRHS_e=ReturnMatrix_e+DiscountedEV;
                [~,maxindex]=max(entireRHS_e,[],2);

                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_bothz,special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals, e_val, ReturnFnParamsVec,2,0);
                da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*zind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d3)+1;
                allind=d_ind+N_d3*aind+N_d3*N_a*zindB;
                mid_at=shiftdim(squeeze(midpoint(allind)),-1);
                L2offset=ceil(maxindexL2/N_d3);
                linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
                linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
                d3opt=d_ind;
                a1opt_mid=midpoint(allind);
                zlin=shiftdim(gpuArray(0:N_bothz-1),-1);
                lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
                d2index_ford4_jj(:,:,e_c,d4_c)=shiftdim(d2index_resh(lin),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

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
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals(semizblock,:);
                semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3);
                semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                DiscountedEVblock=DiscountedEV(:,:,:,:,semizblock);
                DiscountedEVinterpblock=DiscountedEVinterp(:,:,:,:,semizblock);
                d2index_reshblock=d2index_resh(:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);
                    entireRHS_e=ReturnMatrix_e+DiscountedEVblock;
                    [~,maxindex]=max(entireRHS_e,[],2);

                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*semizind;
                    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterpblock(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d3)+1;
                    allind=d_ind+N_d3*aind+N_d3*N_a*semizBind;
                    mid_at=shiftdim(squeeze(midpoint(allind)),-1);
                    L2offset=ceil(maxindexL2/N_d3);
                    linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*semizBind;
                    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*semizBind;
                    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                    Policy_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
                    d3opt=d_ind;
                    a1opt_mid=midpoint(allind);
                    zlin=shiftdim(gpuArray(0:N_semiz-1),-1);
                    lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
                    d2index_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(d2index_reshblock(lin),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

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
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals(z_c,:);
                DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);
                d2index_resh_z=d2index_resh(:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                    entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z;
                    [~,maxindex]=max(entireRHS_ze,[],2);

                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    da1prime=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1);
                    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(da1prime(:)),[N_d3*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d3)+1;
                    allind=d_ind+N_d3*aind;
                    mid_at=midpoint(allind);
                    L2offset=ceil(maxindexL2/N_d3);
                    linidx_lower  = d_ind                   + N_d3*n2long*aind;
                    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
                    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                    Policy_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
                    d3opt=d_ind;
                    a1opt_mid=midpoint(allind);
                    lin=d3opt+N_d3*(a1opt_mid-1);
                    d2index_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(d2index_resh_z(lin),1);
                end
            end
        end
    end

    [Vbest,d4winner]=max(V_ford4_jj,[],4);
    V(:,:,:,N_j)=Vbest;
    Ncomb=N_a*N_bothz*N_e;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_bothz,N_e]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_bothz,N_e]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_bothz,N_e]);
    d3part=rem(polenc-1,N_d3)+1;
    tmp=ceil(polenc/N_d3);
    midpart=rem(tmp-1,N_a1)+1;
    L2offset=ceil(tmp/N_a1);
    adjust=(L2offset<1+n2short+1);
    a1prime_low=midpart-adjust;
    L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);
    Policy(1,:,:,:,N_j)=reshape(d2winner,[1,N_a,N_bothz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(d3part,[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(d4winner,[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(a1prime_low,[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(L2ind,[1,N_a,N_bothz,N_e]);
    Policy(6,:,:,:,N_j)=reshape(flagwinner,[1,N_a,N_bothz,N_e]);
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

    bothz_gridvals=bothz_gridvals_J(:,:,jj);
    e_gridvals=e_gridvals_J(:,:,jj);
    pi_z=pi_z_J(:,:,jj);

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

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
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,n_bothz,n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec,1,0);

            entireRHS=ReturnMatrix_d4+DiscountedEV; % broadcast a2,e

            [~,maxindex]=max(entireRHS,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_bothz,n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec,2,0);
            da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d_ind+N_d3*aind+N_d3*N_a*zeindB;
            mid_at=shiftdim(squeeze(midpoint(allind)),-1);
            L2offset=ceil(maxindexL2/N_d3);
            linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
            linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
            d3opt=d_ind;
            a1opt_mid=midpoint(allind);
            zlin=shiftdim(gpuArray(0:N_bothz-1),-1);
            lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
            d2index_ford4_jj(:,:,:,d4_c)=shiftdim(d2index_resh(lin),1);
        end

    elseif vfoptions.lowmemory==1
        special_n_e=ones(1,length(n_e));
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

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
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for e_c=1:N_e
                e_val=e_gridvals(e_c,:);
                ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,n_bothz,special_n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_val, ReturnFnParamsVec,1,0);
                entireRHS_e=ReturnMatrix_e+DiscountedEV;
                [~,maxindex]=max(entireRHS_e,[],2);

                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_bothz,special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals, e_val, ReturnFnParamsVec,2,0);
                da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*zind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d3)+1;
                allind=d_ind+N_d3*aind+N_d3*N_a*zindB;
                mid_at=shiftdim(squeeze(midpoint(allind)),-1);
                L2offset=ceil(maxindexL2/N_d3);
                linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
                linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
                d3opt=d_ind;
                a1opt_mid=midpoint(allind);
                zlin=shiftdim(gpuArray(0:N_bothz-1),-1);
                lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
                d2index_ford4_jj(:,:,e_c,d4_c)=shiftdim(d2index_resh(lin),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

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
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals(semizblock,:);
                semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3);
                semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                DiscountedEVblock=DiscountedEV(:,:,:,:,semizblock);
                DiscountedEVinterpblock=DiscountedEVinterp(:,:,:,:,semizblock);
                d2index_reshblock=d2index_resh(:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);
                    entireRHS_e=ReturnMatrix_e+DiscountedEVblock;
                    [~,maxindex]=max(entireRHS_e,[],2);

                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*semizind;
                    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterpblock(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d3)+1;
                    allind=d_ind+N_d3*aind+N_d3*N_a*semizBind;
                    mid_at=shiftdim(squeeze(midpoint(allind)),-1);
                    L2offset=ceil(maxindexL2/N_d3);
                    linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*semizBind;
                    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*semizBind;
                    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                    Policy_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
                    d3opt=d_ind;
                    a1opt_mid=midpoint(allind);
                    zlin=shiftdim(gpuArray(0:N_semiz-1),-1);
                    lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
                    d2index_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(d2index_reshblock(lin),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

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
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals(z_c,:);
                DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);
                d2index_resh_z=d2index_resh(:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d3_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                    entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z;
                    [~,maxindex]=max(entireRHS_ze,[],2);

                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    da1prime=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1);
                    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(da1prime(:)),[N_d3*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d3)+1;
                    allind=d_ind+N_d3*aind;
                    mid_at=midpoint(allind);
                    L2offset=ceil(maxindexL2/N_d3);
                    linidx_lower  = d_ind                   + N_d3*n2long*aind;
                    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
                    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                    Policy_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(d_ind,1)+N_d3*(shiftdim(mid_at,1)-1)+N_d3*N_a1*(shiftdim(L2offset,1)-1);
                    d3opt=d_ind;
                    a1opt_mid=midpoint(allind);
                    lin=d3opt+N_d3*(a1opt_mid-1);
                    d2index_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(d2index_resh_z(lin),1);
                end
            end
        end
    end

    [Vbest,d4winner]=max(V_ford4_jj,[],4);
    V(:,:,:,jj)=Vbest;
    Ncomb=N_a*N_bothz*N_e;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_bothz,N_e]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_bothz,N_e]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_bothz,N_e]);
    d3part=rem(polenc-1,N_d3)+1;
    tmp=ceil(polenc/N_d3);
    midpart=rem(tmp-1,N_a1)+1;
    L2offset=ceil(tmp/N_a1);
    adjust=(L2offset<1+n2short+1);
    a1prime_low=midpart-adjust;
    L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);
    Policy(1,:,:,:,jj)=reshape(d2winner,[1,N_a,N_bothz,N_e]);
    Policy(2,:,:,:,jj)=reshape(d3part,[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,jj)=reshape(d4winner,[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(a1prime_low,[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,jj)=reshape(L2ind,[1,N_a,N_bothz,N_e]);
    Policy(6,:,:,:,jj)=reshape(flagwinner,[1,N_a,N_bothz,N_e]);
end


end
