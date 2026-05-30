function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_GI1_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_e,n_u,N_j, d2_grid, d3_grid, a1_grid, a2_grid, e_gridvals_J, u_grid, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No d1, no z, with e. e treated as z-slot in Return helper.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_e=prod(n_e);
N_u=prod(n_u);

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_e,N_j,'gpuArray'); % (1)=d2, (2)=d3, (3)=midpoint, (4)=L2ind
PolicyL2flag=2*ones(1,N_a,N_e,N_j,'gpuArray');
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d3_gridvals=CreateGridvals(n_d3,d3_grid,1);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e),'gpuArray');
end

% Setup for GI
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

% Precompute
aind=gpuArray(0:1:N_a-1);
eindB=shiftdim((0:1:N_e-1),-1); % [1,1,N_e]
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % e as z
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

        % Grid interpolation layer
        aprimeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_e, d3_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d3_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d3_ind+N_d3*aind+N_d3*N_a*eindB;
        Policy(2,:,:,N_j)=d3_ind;                                       % d3
        Policy(3,:,:,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);    % midpoint
        Policy(4,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);           % L2ind

        % L2flag
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*eindB;
        linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*eindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,1,0);
            [~,maxindex]=max(ReturnMatrix_e,[],2);
            midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

            % Grid interpolation layer
            aprimeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,special_n_e, d3_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            d3_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d3_ind+N_d3*aind;
            Policy(2,:,e_c,N_j)=d3_ind;
            Policy(3,:,e_c,N_j)=midpoint_jj(allind);
            Policy(4,:,e_c,N_j)=ceil(maxindexL2/N_d3);

            % L2flag
            L2offset      = ceil(maxindexL2/N_d3);
            linidx_lower  = d3_ind                   + N_d3*n2long*aind;
            linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
            isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end

    % d2, which was not in ReturnFn
    Policy(1,:,:,N_j)=ones(1,N_a,N_e,'gpuArray'); % d2 (terminal: d2 doesn't matter, only in expectations)

else % V_Jplus1

    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);
    EV=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j),-1),2); % [N_a,1]

    % Build a2primeIndex and a2primeProbs for RisykAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Get EV in terms of next period endogenous states
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a2primeProbs,N_a1,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1]),[],1);
    EV=reshape(EV,[N_d3*N_a1,1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,1,1]);
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]);

    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % e in z-slot
        entireRHS=ReturnMatrix+DiscountedEV;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

        % Grid interpolation layer
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        da1prime=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1);
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1prime(:)),[N_d3*n2long,N_a1*N_a2,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d3_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d3_ind+N_d3*aind+N_d3*N_a*eindB;
        Policy(2,:,:,N_j)=d3_ind;
        Policy(3,:,:,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);
        Policy(4,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);

        % L2flag
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*eindB;
        linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*eindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

        % Get the d2Policy
        a1mid=midpoint_jj(allind); % [1,N_a,N_e]
        lin=d3_ind+N_d3*(a1mid-1);
        Policy(1,:,:,N_j)=reshape(d2index_resh(lin),[1,N_a,N_e]);

    elseif vfoptions.lowmemory>=1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,1,0);
            entireRHS_e=ReturnMatrix_e+DiscountedEV;
            [~,maxindex]=max(entireRHS_e,[],2);
            midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

            % Grid interpolation layer
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,special_n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0);
            da1prime=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1);
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1prime(:)),[N_d3*n2long,N_a1*N_a2]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            d3_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d3_ind+N_d3*aind;
            Policy(2,:,e_c,N_j)=d3_ind;
            Policy(3,:,e_c,N_j)=midpoint_jj(allind);
            Policy(4,:,e_c,N_j)=ceil(maxindexL2/N_d3);

            % L2flag
            L2offset      = ceil(maxindexL2/N_d3);
            linidx_lower  = d3_ind                   + N_d3*n2long*aind;
            linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % Get the d2Policy
            a1mid=midpoint_jj(allind);
            lin=d3_ind+N_d3*(a1mid-1);
            Policy(1,:,e_c,N_j)=d2index_resh(lin);
        end
    end
end

%% Iterate backwards
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj));

    % Build a2primeIndex and a2primeProbs for RisykAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Get EV in terms of next period endogenous states
    EV=sum(V(:,:,jj+1).*shiftdim(pi_e_J(:,jj),-1),2); % [N_a,1]
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a2primeProbs,N_a1,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1]),[],1);
    EV=reshape(EV,[N_d3*N_a1,1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,1,1]);
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]);

    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
        entireRHS=ReturnMatrix+DiscountedEV;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

        % Grid interpolation layer
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
        da1prime=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1);
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1prime(:)),[N_d3*n2long,N_a1*N_a2,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);
        d3_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d3_ind+N_d3*aind+N_d3*N_a*eindB;
        Policy(2,:,:,jj)=d3_ind;
        Policy(3,:,:,jj)=shiftdim(squeeze(midpoint_jj(allind)),-1);
        Policy(4,:,:,jj)=shiftdim(ceil(maxindexL2/N_d3),-1);

        % L2flag
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*eindB;
        linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*eindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

        % Get the d2Policy
        a1mid=midpoint_jj(allind); % [1,N_a,N_e]
        lin=d3_ind+N_d3*(a1mid-1);
        Policy(1,:,:,jj)=reshape(d2index_resh(lin),[1,N_a,N_e]);

    elseif vfoptions.lowmemory>=1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,1,0);
            entireRHS_e=ReturnMatrix_e+DiscountedEV;
            [~,maxindex]=max(entireRHS_e,[],2);
            midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

            % Grid interpolation layer
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,special_n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0);
            da1prime=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1);
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1prime(:)),[N_d3*n2long,N_a1*N_a2]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,e_c,jj)=shiftdim(Vtempii,1);
            d3_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d3_ind+N_d3*aind;
            Policy(2,:,e_c,jj)=d3_ind;
            Policy(3,:,e_c,jj)=midpoint_jj(allind);
            Policy(4,:,e_c,jj)=ceil(maxindexL2/N_d3);

            % L2flag
            L2offset      = ceil(maxindexL2/N_d3);
            linidx_lower  = d3_ind                   + N_d3*n2long*aind;
            linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % Get the d2Policy
            a1mid=midpoint_jj(allind);
            lin=d3_ind+N_d3*(a1mid-1);
            Policy(1,:,e_c,jj)=d2index_resh(lin);
        end
    end
end

%% Switch Policy(3,:) from 'midpoint' to 'lower grid index' (using L2ind side)
adjust=(Policy(4,:,:,:)<1+n2short+1);                                                  % L2ind strictly < n2short+2
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust;                                                % decrement midpoint when chosen-below
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);       % rebase L2ind to [1..n2short+2]

Policy=[Policy; PolicyL2flag];

end
