function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
%
% Splices:
%   - SemiExo DC: d4 outer loop, per-d4 d2-refinement, cross-d4 max with slab lookup
%   - Plain DC_GI: DC level1n outer + GI midpoint+L2 fine inner; L2flag scaffold

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
special_n_d4=ones(1,length(n_d4));
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_u=prod(n_u);

N_d13=N_d1*N_d3;
N_d1d2d3=N_d1*N_d2*N_d3;

% For ReturnFn (d1 and d3 inside the level1 helper)
% For aprimeFn (d2 and d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_bothz,N_j,'gpuArray');
% Stores (d13, a1prime midpoint, L2ind) packed; d2 and d4 added after cross-d4 max
Policy3=zeros(3,N_a,N_bothz,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_bothz,N_j,'gpuArray');
d2Policy=ones(1,N_a,N_bothz,N_j,'gpuArray');
d4Policy=ones(1,N_a,N_bothz,N_j,'gpuArray');

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

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
d3ind=repelem(gpuArray(1:1:N_d3)',N_d1,1); % [N_d13,1]

if vfoptions.lowmemory>=1
    special_n_semiz=ones(1,length(n_semiz));
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate per-d4 slabs
V_ford4_jj=zeros(N_a,N_bothz,N_d4,'gpuArray');
Policy3_ford4_jj=zeros(3,N_a,N_bothz,N_d4,'gpuArray');
flag_ford4_jj=2*ones(N_a,N_bothz,N_d4,'gpuArray');
d2_ford4_jj=ones(N_a,N_bothz,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % Terminal: only ReturnFn matters; d2 is meaningless (set to 1).
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_bothz, d1d3d4a1_gridvals, a1a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=shiftdim(Vtemp,1);
    dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
    d1d3_ind=rem(dindex-1,N_d13)+1;
    d1part=rem(d1d3_ind-1,N_d1)+1;
    d3part=ceil(d1d3_ind/N_d1);
    d4part=ceil(dindex/N_d13);
    a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
    % Stash into Policy3 + d4Policy + d2Policy + PolicyL2flag so end-of-file encoding works.
    % d13 packed as d1+N_d1*(d3-1); midpoint=a1prime; L2ind=n2short+2 (maps to "lower=midpoint, L2flag=2")
    Policy3(1,:,:,N_j)=shiftdim(d1part+N_d1*(d3part-1),-1);
    Policy3(2,:,:,N_j)=shiftdim(a1primepart,-1);
    Policy3(3,:,:,N_j)=n2short+2; % adjust=0 in final block => stays n2short+2-n2short-1=1 (lower index 1 of GI)
    d4Policy(1,:,:,N_j)=shiftdim(d4part,-1);
    % (PolicyL2flag stays 2; d2Policy stays 1)
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

    bothz_gridvals=bothz_gridvals_J(:,:,N_j);
    pi_z=pi_z_J(:,:,N_j);

    V_jj=zeros(N_a,N_bothz,'gpuArray');
    Policy3_jj=zeros(3,N_a,N_bothz,'gpuArray');
    PolicyL2flag_jj=2*ones(1,N_a,N_bothz,'gpuArray');
    d2Policy_jj=ones(1,N_a,N_bothz,'gpuArray');
    d4Policy_jj=ones(1,N_a,N_bothz,'gpuArray');

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    for d4_c=1:N_d4
        pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        % EV integrated over bothz' (zprime)
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

        % Refine d2: max over d2
        EVres=reshape(EV,[N_d2,N_d3*N_a1,N_bothz]);
        [EV_onlyd3,d2index]=max(EVres,[],1);
        EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_bothz]);
        d2index_resh=reshape(d2index,[N_d3,N_a1,N_bothz]);

        DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_bothz]);
        % Interpolate EV over a1prime fine grid
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d3,N_a1prime,1,1,N_bothz]

        if vfoptions.lowmemory==0
            midpoint=zeros(N_d13,1,N_a1,N_a2,N_bothz,'gpuArray');

            % n-Monotonicity (coarse DC search at level1ii midpoints)
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_bothz, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals, ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_bothz]);
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_bothz]);
            entireRHS_ii=RM+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_bothz]);

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals, ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % GI fine search at n2long interpolated points either side of each midpoint
            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals, ReturnFnParamsVec,2,0);
            da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBind,-2);
            entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_bothz])+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_bothz]),[N_d13*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zBind;
            Policy3_ford4_jj(1,:,:,d4_c)=d_ind; % d13 index
            Policy3_ford4_jj(2,:,:,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford4_jj(3,:,:,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

            % L2 flag detection
            L2offset      = ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
            ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_bothz]);
            isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            % d2 lookup: d2 = d2index_resh(d3part, a1primemidpoint, z)
            d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
            a1mid=squeeze(midpoint(allind));
            zidx=repmat(gpuArray(1:N_bothz),N_a,1);
            linlookup=d3part+N_d3*(a1mid-1)+N_d3*N_a1*(zidx-1);
            d2_ford4_jj(:,:,d4_c)=d2index_resh(linlookup);
        elseif vfoptions.lowmemory==1
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals(semizblock,:);
                zBindblock=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                DiscountedEVblock=DiscountedEV(:,:,:,:,semizblock);
                DiscountedEVinterpblock=DiscountedEVinterp(:,:,:,:,semizblock);
                d2index_reshblock=d2index_resh(:,:,semizblock);

                midpoint=zeros(N_d13,1,N_a1,N_a2,N_semiz,'gpuArray');

                % n-Monotonicity (coarse DC search at level1ii midpoints)
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                DEV=reshape(DiscountedEVblock,[1,N_d3,N_a1,1,1,N_semiz]);
                entireRHS_ii=RM+DEV;
                entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz]);

                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBindblock,-2);
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEVblock(d3aprimez);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                % GI fine search at n2long interpolated points either side of each midpoint
                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBindblock,-2);
                entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz])+reshape(DiscountedEVinterpblock(da1primez),[N_d13,n2long,N_a1,N_a2,N_semiz]),[N_d13*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,semizblock,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind+N_d13*N_a*zBindblock;
                Policy3_ford4_jj(1,:,semizblock,d4_c)=d_ind; % d13 index
                Policy3_ford4_jj(2,:,semizblock,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford4_jj(3,:,semizblock,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                % L2 flag detection
                L2offset      = ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBindblock;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBindblock;
                ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz]);
                isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,semizblock,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                % d2 lookup: d2 = d2index_reshblock(d3part, a1primemidpoint, z)
                d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
                a1mid=squeeze(midpoint(allind));
                zidx=repmat(gpuArray(1:N_semiz),N_a,1);
                linlookup=d3part+N_d3*(a1mid-1)+N_d3*N_a1*(zidx-1);
                d2_ford4_jj(:,semizblock,d4_c)=d2index_reshblock(linlookup);
            end
        elseif vfoptions.lowmemory>=2 % lm2 already does the most-looped variant, so it also serves the higher lowmemory values
            for z_c=1:N_bothz
                z_val=bothz_gridvals(z_c,:);
                DiscountedEV_zc=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_zc=DiscountedEVinterp(:,:,:,:,z_c);
                d2index_resh_zc=d2index_resh(:,:,z_c);

                midpoint=zeros(N_d13,1,N_a1,N_a2,1,'gpuArray');

                % n-Monotonicity (coarse DC search at level1ii midpoints)
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,special_n_bothz, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,1]);
                DEV=reshape(DiscountedEV_zc,[1,N_d3,N_a1,1,1,1]);
                entireRHS_ii=RM+DEV;
                entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,1]);

                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_zc(d3aprimez);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                % GI fine search at n2long interpolated points either side of each midpoint
                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                da1primez=d3ind+N_d3*(a1primeindexesfine-1);
                entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,1])+reshape(DiscountedEVinterp_zc(da1primez),[N_d13,n2long,N_a1,N_a2,1]),[N_d13*n2long,N_a1*N_a2,1]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind;
                Policy3_ford4_jj(1,:,z_c,d4_c)=d_ind; % d13 index
                Policy3_ford4_jj(2,:,z_c,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford4_jj(3,:,z_c,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                % L2 flag detection
                L2offset      = ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,1]);
                isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,z_c,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                d3part=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                a1mid=midpoint(allind);
                linlookup=d3part+N_d3*(a1mid-1);
                d2_ford4_jj(:,z_c,d4_c)=d2index_resh_zc(linlookup);
            end
        end
    end

    % Cross-d4 max
    [V_jj,d4winner]=max(V_ford4_jj,[],3);
    linidx_d4=(1:1:N_a*N_bothz)'+(N_a*N_bothz)*(reshape(d4winner,[N_a*N_bothz,1])-1);
    % Gather per-d4 slabs at winning d4
    % Policy3_ford4_jj is [3,N_a,N_bothz,N_d4]; we want per-(a,z) gather along dim 4
    P1=reshape(Policy3_ford4_jj(1,:,:,:),[N_a*N_bothz,N_d4]);
    P2=reshape(Policy3_ford4_jj(2,:,:,:),[N_a*N_bothz,N_d4]);
    P3=reshape(Policy3_ford4_jj(3,:,:,:),[N_a*N_bothz,N_d4]);
    F =reshape(flag_ford4_jj,[N_a*N_bothz,N_d4]);
    D2=reshape(d2_ford4_jj,[N_a*N_bothz,N_d4]);
    rowidx=(1:1:N_a*N_bothz)';
    gather_idx=rowidx+(N_a*N_bothz)*(reshape(d4winner,[N_a*N_bothz,1])-1);
    Policy3_jj(1,:,:)=shiftdim(reshape(P1(gather_idx),[N_a,N_bothz]),-1);
    Policy3_jj(2,:,:)=shiftdim(reshape(P2(gather_idx),[N_a,N_bothz]),-1);
    Policy3_jj(3,:,:)=shiftdim(reshape(P3(gather_idx),[N_a,N_bothz]),-1);
    PolicyL2flag_jj(1,:,:)=shiftdim(reshape(F(gather_idx),[N_a,N_bothz]),-1);
    d2Policy_jj(1,:,:)=shiftdim(reshape(D2(gather_idx),[N_a,N_bothz]),-1);
    d4Policy_jj(1,:,:)=shiftdim(d4winner,-1);

    V(:,:,N_j)=V_jj;
    Policy3(:,:,:,N_j)=Policy3_jj;
    PolicyL2flag(:,:,:,N_j)=PolicyL2flag_jj;
    d2Policy(:,:,:,N_j)=d2Policy_jj;
    d4Policy(:,:,:,N_j)=d4Policy_jj;
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

    bothz_gridvals=bothz_gridvals_J(:,:,jj);
    pi_z=pi_z_J(:,:,jj);

    V_jj=zeros(N_a,N_bothz,'gpuArray');
    Policy3_jj=zeros(3,N_a,N_bothz,'gpuArray');
    PolicyL2flag_jj=2*ones(1,N_a,N_bothz,'gpuArray');
    d2Policy_jj=ones(1,N_a,N_bothz,'gpuArray');
    d4Policy_jj=ones(1,N_a,N_bothz,'gpuArray');

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    for d4_c=1:N_d4
        pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        % EV integrated over bothz' (zprime)
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

        % Refine d2: max over d2
        EVres=reshape(EV,[N_d2,N_d3*N_a1,N_bothz]);
        [EV_onlyd3,d2index]=max(EVres,[],1);
        EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_bothz]);
        d2index_resh=reshape(d2index,[N_d3,N_a1,N_bothz]);

        DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_bothz]);
        % Interpolate EV over a1prime fine grid
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d3,N_a1prime,1,1,N_bothz]

        if vfoptions.lowmemory==0
            midpoint=zeros(N_d13,1,N_a1,N_a2,N_bothz,'gpuArray');

            % n-Monotonicity (coarse DC search at level1ii midpoints)
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_bothz, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals, ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_bothz]);
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_bothz]);
            entireRHS_ii=RM+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_bothz]);

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals, ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % GI fine search at n2long interpolated points either side of each midpoint
            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals, ReturnFnParamsVec,2,0);
            da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBind,-2);
            entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_bothz])+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_bothz]),[N_d13*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zBind;
            Policy3_ford4_jj(1,:,:,d4_c)=d_ind; % d13 index
            Policy3_ford4_jj(2,:,:,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford4_jj(3,:,:,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

            % L2 flag detection
            L2offset      = ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
            ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_bothz]);
            isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            % d2 lookup: d2 = d2index_resh(d3part, a1primemidpoint, z)
            d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
            a1mid=squeeze(midpoint(allind));
            zidx=repmat(gpuArray(1:N_bothz),N_a,1);
            linlookup=d3part+N_d3*(a1mid-1)+N_d3*N_a1*(zidx-1);
            d2_ford4_jj(:,:,d4_c)=d2index_resh(linlookup);
        elseif vfoptions.lowmemory==1
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals(semizblock,:);
                zBindblock=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                DiscountedEVblock=DiscountedEV(:,:,:,:,semizblock);
                DiscountedEVinterpblock=DiscountedEVinterp(:,:,:,:,semizblock);
                d2index_reshblock=d2index_resh(:,:,semizblock);

                midpoint=zeros(N_d13,1,N_a1,N_a2,N_semiz,'gpuArray');

                % n-Monotonicity (coarse DC search at level1ii midpoints)
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                DEV=reshape(DiscountedEVblock,[1,N_d3,N_a1,1,1,N_semiz]);
                entireRHS_ii=RM+DEV;
                entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz]);

                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBindblock,-2);
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEVblock(d3aprimez);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                % GI fine search at n2long interpolated points either side of each midpoint
                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBindblock,-2);
                entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz])+reshape(DiscountedEVinterpblock(da1primez),[N_d13,n2long,N_a1,N_a2,N_semiz]),[N_d13*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,semizblock,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind+N_d13*N_a*zBindblock;
                Policy3_ford4_jj(1,:,semizblock,d4_c)=d_ind; % d13 index
                Policy3_ford4_jj(2,:,semizblock,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford4_jj(3,:,semizblock,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                % L2 flag detection
                L2offset      = ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBindblock;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBindblock;
                ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz]);
                isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,semizblock,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                % d2 lookup: d2 = d2index_reshblock(d3part, a1primemidpoint, z)
                d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
                a1mid=squeeze(midpoint(allind));
                zidx=repmat(gpuArray(1:N_semiz),N_a,1);
                linlookup=d3part+N_d3*(a1mid-1)+N_d3*N_a1*(zidx-1);
                d2_ford4_jj(:,semizblock,d4_c)=d2index_reshblock(linlookup);
            end
        elseif vfoptions.lowmemory>=2 % lm2 already does the most-looped variant, so it also serves the higher lowmemory values
            for z_c=1:N_bothz
                z_val=bothz_gridvals(z_c,:);
                DiscountedEV_zc=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_zc=DiscountedEVinterp(:,:,:,:,z_c);
                d2index_resh_zc=d2index_resh(:,:,z_c);

                midpoint=zeros(N_d13,1,N_a1,N_a2,1,'gpuArray');

                % n-Monotonicity (coarse DC search at level1ii midpoints)
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,special_n_bothz, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,1]);
                DEV=reshape(DiscountedEV_zc,[1,N_d3,N_a1,1,1,1]);
                entireRHS_ii=RM+DEV;
                entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,1]);

                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_zc(d3aprimez);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                % GI fine search at n2long interpolated points either side of each midpoint
                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                da1primez=d3ind+N_d3*(a1primeindexesfine-1);
                entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,1])+reshape(DiscountedEVinterp_zc(da1primez),[N_d13,n2long,N_a1,N_a2,1]),[N_d13*n2long,N_a1*N_a2,1]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind;
                Policy3_ford4_jj(1,:,z_c,d4_c)=d_ind; % d13 index
                Policy3_ford4_jj(2,:,z_c,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford4_jj(3,:,z_c,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                % L2 flag detection
                L2offset      = ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,1]);
                isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,z_c,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                d3part=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                a1mid=midpoint(allind);
                linlookup=d3part+N_d3*(a1mid-1);
                d2_ford4_jj(:,z_c,d4_c)=d2index_resh_zc(linlookup);
            end
        end
    end

    % Cross-d4 max
    [V_jj,d4winner]=max(V_ford4_jj,[],3);
    linidx_d4=(1:1:N_a*N_bothz)'+(N_a*N_bothz)*(reshape(d4winner,[N_a*N_bothz,1])-1);
    % Gather per-d4 slabs at winning d4
    % Policy3_ford4_jj is [3,N_a,N_bothz,N_d4]; we want per-(a,z) gather along dim 4
    P1=reshape(Policy3_ford4_jj(1,:,:,:),[N_a*N_bothz,N_d4]);
    P2=reshape(Policy3_ford4_jj(2,:,:,:),[N_a*N_bothz,N_d4]);
    P3=reshape(Policy3_ford4_jj(3,:,:,:),[N_a*N_bothz,N_d4]);
    F =reshape(flag_ford4_jj,[N_a*N_bothz,N_d4]);
    D2=reshape(d2_ford4_jj,[N_a*N_bothz,N_d4]);
    rowidx=(1:1:N_a*N_bothz)';
    gather_idx=rowidx+(N_a*N_bothz)*(reshape(d4winner,[N_a*N_bothz,1])-1);
    Policy3_jj(1,:,:)=shiftdim(reshape(P1(gather_idx),[N_a,N_bothz]),-1);
    Policy3_jj(2,:,:)=shiftdim(reshape(P2(gather_idx),[N_a,N_bothz]),-1);
    Policy3_jj(3,:,:)=shiftdim(reshape(P3(gather_idx),[N_a,N_bothz]),-1);
    PolicyL2flag_jj(1,:,:)=shiftdim(reshape(F(gather_idx),[N_a,N_bothz]),-1);
    d2Policy_jj(1,:,:)=shiftdim(reshape(D2(gather_idx),[N_a,N_bothz]),-1);
    d4Policy_jj(1,:,:)=shiftdim(d4winner,-1);

    V(:,:,jj)=V_jj;
    Policy3(:,:,:,jj)=Policy3_jj;
    PolicyL2flag(:,:,:,jj)=PolicyL2flag_jj;
    d2Policy(:,:,:,jj)=d2Policy_jj;
    d4Policy(:,:,:,jj)=d4Policy_jj;
end


%% With grid interpolation, switch Policy3(2,:) from 'midpoint' to 'lower grid index'
adjust=(Policy3(3,:,:,:)<1+n2short+1);
Policy3(2,:,:,:)=Policy3(2,:,:,:)-adjust;
Policy3(3,:,:,:)=adjust.*Policy3(3,:,:,:)+(1-adjust).*(Policy3(3,:,:,:)-n2short-1);

%% Encode Policy (7 rows: d1,d2,d3,d4,a1prime_low,L2,L2flag)
Policy=zeros(7,N_a,N_bothz,N_j,'gpuArray');
d13=Policy3(1,:,:,:);
Policy(1,:,:,:)=rem(d13-1,N_d1)+1;
Policy(2,:,:,:)=d2Policy;
Policy(3,:,:,:)=rem(ceil(d13/N_d1)-1,N_d3)+1;
Policy(4,:,:,:)=d4Policy;
Policy(5,:,:,:)=Policy3(2,:,:,:);
Policy(6,:,:,:)=Policy3(3,:,:,:);
Policy(7,:,:,:)=PolicyL2flag;

end
