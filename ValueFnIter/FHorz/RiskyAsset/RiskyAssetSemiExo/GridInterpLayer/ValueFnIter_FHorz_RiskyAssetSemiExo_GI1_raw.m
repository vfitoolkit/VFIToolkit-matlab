function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
%
% Plain GI + d4 outer loop. Inside each d4: refine d2 out of EV, then GI midpoint+L2 over a1 with d1+d3+a1prime.
% After d4 loop: max over d4 and look up per-d4 slabs.

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
N_u=prod(n_u);

N_d13=N_d1*N_d3;
N_d=N_d1*N_d2*N_d3; % per-d4 d-block; final d-block is N_d*N_d4

% For ReturnFn (d1,d3) per-d4
n_d13=[n_d1,n_d3]; %#ok<NASGU>
d13_grid=[d1_grid;d3_grid]; %#ok<NASGU>
% For aprimeFn (d2,d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

% Variant of d4 (single slice) and d4 gridvals
special_n_d4=ones(1,length(n_d4)); %#ok<NASGU>
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

V=zeros(N_a,N_bothz,N_j,'gpuArray');
% Final Case2-FHorz Kron index over (d1,d2,d3,d4,a1prime_low,L2ind,L2flag)
Policy=zeros(N_a,N_bothz,N_j,'gpuArray');

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

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_bothz-1),-3);
zindB=shiftdim(gpuArray(0:1:N_bothz-1),-1);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);

% Preallocate per-d4 slabs
V_ford4_jj=zeros(N_a,N_bothz,N_d4,'gpuArray');
% Per-d4 policy encoding (no d4 baked in): d13 + N_d13*(a1prime_mid-1) + N_d13*N_a1*(L2ind-1)
Policy_ford4_jj=zeros(N_a,N_bothz,N_d4,'gpuArray');
flag_ford4_jj=2*ones(N_a,N_bothz,N_d4,'gpuArray');
d2index_ford4_jj=ones(N_a,N_bothz,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % Terminal: only ReturnFn matters; d2 meaningless (set to 1). Run GI search on ReturnMatrix alone, looping over d4 to mirror SemiExo baseline.
    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,n_bothz, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % [N_d13,N_a1prime,N_a1,N_a2,N_bothz]
            [~,maxindex_d4]=max(ReturnMatrix_d4,[],2);

            midpoint_d4=max(min(maxindex_d4,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_d4+(midpoint_d4-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford4_jj(:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
            mid_at=shiftdim(squeeze(midpoint_d4(allind)),-1); % [1,N_a,N_bothz]
            L2offset=shiftdim(ceil(maxindexL2/N_d13),-1); % [1,N_a,N_bothz]
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            % Per-d4 policy encoding stores d13 + N_d13*(mid-1) + N_d13*N_a1*(L2offset-1)
            pol_combined=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
            Policy_ford4_jj(:,:,d4_c)=pol_combined;
            % d2 meaningless at j=N_j
            d2index_ford4_jj(:,:,d4_c)=1;
        end
    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d4z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,special_n_bothz, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                [~,maxindex_d4z]=max(ReturnMatrix_d4z,[],2);

                midpoint_d4z=max(min(maxindex_d4z,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_d4z+(midpoint_d4z-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,special_n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1; % [1,1,N_a]
                allind=d_ind+N_d13*aind;
                mid_at=midpoint_d4z(allind); % [1,N_a]
                L2offset=ceil(maxindexL2/N_d13); % [1,1,N_a]
                linidx_lower  = d_ind                    + N_d13*n2long*aind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,z_c,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                pol_combined=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
                Policy_ford4_jj(:,z_c,d4_c)=pol_combined;
                d2index_ford4_jj(:,z_c,d4_c)=1;
            end
        end
    end
    % Cross-d4 max
    [V(:,:,N_j),Policy(:,:,N_j),~]=combine_across_d4(V_ford4_jj,Policy_ford4_jj,d2index_ford4_jj,flag_ford4_jj,N_a,N_bothz,N_d4,N_d1,N_d2,N_d3,N_d13,N_d,N_a1,n2short);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    [V(:,:,N_j),Policy(:,:,N_j)]=internal_per_j(EVpre,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_d4,n_a1,n_a2,n_bothz,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d,N_a1,N_a2,N_a,N_bothz,N_u,N_semiz,N_z,N_a1prime,...
        d13_gridvals,d4_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,bothz_gridvals_J(:,:,N_j),pi_z_J(:,:,N_j),pi_semiz,pi_u_col,...
        aind,zind,zindB,a2ind,n2short,n2long,V_ford4_jj,Policy_ford4_jj,flag_ford4_jj,d2index_ford4_jj,vfoptions);
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

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    [V(:,:,jj),Policy(:,:,jj)]=internal_per_j(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_d4,n_a1,n_a2,n_bothz,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d,N_a1,N_a2,N_a,N_bothz,N_u,N_semiz,N_z,N_a1prime,...
        d13_gridvals,d4_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,bothz_gridvals_J(:,:,jj),pi_z_J(:,:,jj),pi_semiz,pi_u_col,...
        aind,zind,zindB,a2ind,n2short,n2long,V_ford4_jj,Policy_ford4_jj,flag_ford4_jj,d2index_ford4_jj,vfoptions);
end


end


%% Per-period inner
function [V_jj,Policy_jj]=internal_per_j(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d1,n_d3,n_d4,n_a1,n_a2,n_bothz,N_d1,N_d2,N_d3,N_d4,N_d13,N_d23,N_d,N_a1,N_a2,N_a,N_bothz_count,N_u,N_semiz,N_z,N_a1prime,...
    d13_gridvals,d4_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,bothz_gridvals,pi_z,pi_semiz,pi_u_col,...
    aind,zind,zindB,a2ind,n2short,n2long,V_ford4_jj,Policy_ford4_jj,flag_ford4_jj,d2index_ford4_jj,vfoptions)

aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

if vfoptions.lowmemory==0
    for d4_c=1:N_d4
        pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        % EV integrated over bothz'
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

        % Refine d2
        EVres=reshape(EV,[N_d2,N_d3*N_a1,N_bothz_count]);
        [EV_onlyd3,d2index]=max(EVres,[],1);
        EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_bothz_count]);
        d2index_resh=reshape(d2index,[N_d3,N_a1,N_bothz_count]);

        DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_bothz_count]); % [N_d3,N_a1,1,1,N_bothz]
        % Interpolate to fine a1prime grid
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d3,N_a1prime,1,1,N_bothz]
        DiscountedEV_d13=repelem(DiscountedEV,N_d1,1); % [N_d13,N_a1,1,1,N_bothz]
        DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1); % [N_d13,N_a1prime,1,1,N_bothz]

        % Level-1 Return at coarse a1prime grid
        ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,n_bothz, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, ReturnFnParamsVec,1,0); % [N_d13,N_a1prime(=N_a1),N_a1,N_a2,N_bothz]

        entireRHS=ReturnMatrix_d4+DiscountedEV_d13; % broadcast a2

        [~,maxindex]=max(entireRHS,[],2);

        midpoint=max(min(maxindex,n_a1(1)-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals, ReturnFnParamsVec,2,0);
        da1primez=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1)+N_d13*N_a1prime*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_d13(da1primez(:)),[N_d13*n2long,N_a1*N_a2,N_bothz_count]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V_ford4_jj(:,:,d4_c)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d13)+1;
        allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
        mid_at=shiftdim(squeeze(midpoint(allind)),-1); % [1,N_a,N_bothz]
        L2offset=shiftdim(ceil(maxindexL2/N_d13),-1); % [1,N_a,N_bothz]
        linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
        linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        flag_ford4_jj(:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        % Per-d4 policy encoding: d13 + N_d13*(mid-1) + N_d13*N_a1*(L2offset-1)
        Policy_ford4_jj(:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
        % d2 lookup per (a,bothz) — uses the COARSE midpoint (matches Plain GI convention)
        d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a,N_bothz]
        a1opt_mid=midpoint(allind); % [1,N_a,N_bothz]
        zlin=shiftdim(gpuArray(0:N_bothz_count-1),-1); % [1,1,N_bothz]
        lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
        d2index_ford4_jj(:,:,d4_c)=shiftdim(d2index_resh(lin),1);
    end

elseif vfoptions.lowmemory==1
    special_n_bothz=ones(1,length(n_bothz));
    for d4_c=1:N_d4
        pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        for z_c=1:N_bothz_count
            z_val=bothz_gridvals(z_c,:);

            EV_z=EVnext.*pi_bothz(z_c,:);
            EV_z(isnan(EV_z))=0;
            EV_z=sum(EV_z,2);
            EV_z=reshape(EV_z,[N_a,1]);

            skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
            aprimeProbs=repmat(a2primeProbs,N_a1,1);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);

            EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
            EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
            EV_z=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);

            EVres=reshape(EV_z,[N_d2,N_d3*N_a1]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,1]);
            d2index_z=reshape(d2index,[N_d3,N_a1]);

            DiscountedEV_z=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1]);
            DiscountedEVinterp_z=permute(interp1(a1_gridvals,permute(DiscountedEV_z,[2,1,3,4]),a1prime_grid),[2,1,3,4]); % [N_d3,N_a1prime,1,1]
            DiscountedEV_d13_z=repelem(DiscountedEV_z,N_d1,1);
            DiscountedEVinterp_d13_z=repelem(DiscountedEVinterp_z,N_d1,1);

            ReturnMatrix_d4z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,special_n_bothz, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);

            entireRHS_z=ReturnMatrix_d4z+DiscountedEV_d13_z;

            [~,maxindex]=max(entireRHS_z,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,special_n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
            da1prime=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1);
            entireRHS_ii_z=ReturnMatrix_ii_z+reshape(DiscountedEVinterp_d13_z(da1prime(:)),[N_d13*n2long,N_a1*N_a2]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind;
            mid_at=midpoint(allind); % [1,N_a]
            L2offset=ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
            isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,z_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,z_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
            d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a]
            a1opt_mid=midpoint(allind);
            lin=d3opt+N_d3*(a1opt_mid-1);
            d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2index_z(lin),1);
        end
    end
end

% Cross-d4 max and combine
[V_jj,Policy_jj,~]=combine_across_d4(V_ford4_jj,Policy_ford4_jj,d2index_ford4_jj,flag_ford4_jj,N_a,N_bothz_count,N_d4,N_d1,N_d2,N_d3,N_d13,N_d,N_a1,n2short);

end


%% Cross-d4 max + final Policy encoding (Case2-FHorz Kron with L2flag and d4 included)
function [V_jj,Policy_jj,d4winner]=combine_across_d4(V_ford4,Policy_ford4,d2idx_ford4,flag_ford4,N_a,N_bothz,N_d4,N_d1,N_d2,N_d3,N_d13,N_d,N_a1,n2short)
% V_ford4: [N_a,N_bothz,N_d4]
% Policy_ford4: [N_a,N_bothz,N_d4] encoding d13 + N_d13*(mid-1) + N_d13*N_a1*(L2offset-1)
% d2idx_ford4: [N_a,N_bothz,N_d4]
% flag_ford4: [N_a,N_bothz,N_d4]
% Returns V_jj [N_a,N_bothz], Policy_jj [N_a,N_bothz] as single Case2-FHorz Kron index.

[V_jj,d4winner]=max(V_ford4,[],3); % [N_a,N_bothz]
N=N_a*N_bothz;
linidx=(1:1:N)'+N*(reshape(d4winner,[N,1])-1);
polenc=reshape(Policy_ford4(linidx),[N_a,N_bothz]);
d2winner=reshape(d2idx_ford4(linidx),[N_a,N_bothz]);
flagwinner=reshape(flag_ford4(linidx),[N_a,N_bothz]);

% Decode polenc into (d13, mid, L2offset)
d13part=rem(polenc-1,N_d13)+1;
tmp=ceil(polenc/N_d13); % mid + N_a1*(L2offset-1)+1 ... actually = (mid-1)+N_a1*(L2offset-1)+1
midpart=rem(tmp-1,N_a1)+1;
L2offset=ceil(tmp/N_a1);

% Switch midpoint -> lower grid point
adjust=(L2offset<1+n2short+1);
a1prime_low=midpart-adjust;
L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);

% Decompose d13 into d1,d3
d1part=rem(d13part-1,N_d1)+1;
d3part=rem(ceil(d13part/N_d1)-1,N_d3)+1;
d2part=d2winner;
d4part=d4winner;

% Final encoding: d1 + N_d1*(d2-1) + N_d1*N_d2*(d3-1) + N_d1*N_d2*N_d3*(d4-1)
%                 + N_d*N_d4*(a1prime_low-1) + N_d*N_d4*N_a1*(L2ind-1) + N_d*N_d4*N_a1*(n2short+2)*(flag-1)
Policy_jj=d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d*(d4part-1)+N_d*N_d4*(a1prime_low-1)+N_d*N_d4*N_a1*(L2ind-1)+N_d*N_d4*N_a1*(n2short+2)*(flagwinner-1);

end
