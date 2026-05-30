function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No z variant.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_u=prod(n_u);

n_d13=[n_d1,n_d3];
N_d13=N_d1*N_d3;
d13_grid=[d1_grid;d3_grid];
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(5,N_a,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray');
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

% Setup for DC
midpoint_jj=zeros(N_d13,1,N_a1,N_a2,'gpuArray');

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Setup for GI
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

% Precompute
aind=gpuArray(0:1:N_a-1);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
a2Bind=gpuArray(0:1:N_a2-1);
d3ind=repelem(gpuArray(1:1:N_d3)',N_d1,1);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % Layer 1
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, ReturnFnParamsVec,1,0);

    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    midpoint_jj(:,1,level1ii,:)=maxindex1;

    % Divide-and-conquer layer 2
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, ReturnFnParamsVec,3,0);
            [~,maxindex]=max(ReturnMatrix_ii,[],2);
            midpoint_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:);
            midpoint_jj(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
        end
    end

    % Grid interpolation layer
    midpoint_jj=max(min(midpoint_jj,n_a1(1)-1),2);
    a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, ReturnFnParamsVec,2,0);
    [Vtempii,maxindexL2]=max(reshape(ReturnMatrix_ii,[N_d13*n2long,N_a1*N_a2]),[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    d3_ind=rem(maxindexL2-1,N_d13)+1;
    allind=d3_ind+N_d13*aind;
    Policy(1,:,N_j)=rem(d3_ind-1,N_d1)+1; % d1
    Policy(3,:,N_j)=rem(ceil(d3_ind/N_d1)-1,N_d3)+1; % d3
    Policy(4,:,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);
    Policy(5,:,N_j)=shiftdim(ceil(maxindexL2/N_d13),-1);

    % L2flag
    L2offset      = ceil(maxindexL2/N_d13);
    linidx_lower  = d3_ind                    + N_d13*n2long*aind;
    linidx_upper  = d3_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
    ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2]);
    isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    % d2, which was not in ReturnFn
    Policy(2,:,N_j)=ones(1,N_a,'gpuArray'); % d2 (because this is terminal period, choice of d2 is not actually doing anything (as it is only in the expectations term)

else % V_Jplus1

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));

    % Build a2primeIndex and a2primeProbs for RisykAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Get EV in terms of next period endogenous states
    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a2primeProbs,N_a1,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    % EV now

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1]),[],1);
    EV=reshape(EV,[N_d3*N_a1,1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,1,1]);
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]); % [N_d3,N_a1prime,1,1]

    % Layer 1
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, ReturnFnParamsVec,1,0);
    % [N_d13, level1n, N_a1, N_a2]; broadcast DiscountedEV across d1 and a2
    RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2]);
    DEV=reshape(DiscountedEV,[1,N_d3,1,N_a1,1]);
    entireRHS_ii=RM+DEV;
    entireRHS_ii=reshape(entireRHS_ii,[N_d13,vfoptions.level1n,N_a1,N_a2]);

    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoint_jj(:,1,level1ii,:)=maxindex1;

    % Divide and conquer layer 2
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, ReturnFnParamsVec,3,0);
            d3aprime=d3ind+N_d3*(a1primeindexes-1); % [N_d13,maxgap+1,1,N_a2]
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprime);
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoint_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:);
            midpoint_jj(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
        end
    end

    % Grid interpolation layer
    midpoint_jj=max(min(midpoint_jj,n_a1(1)-1),2);
    a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, ReturnFnParamsVec,2,0);
    da1prime=d3ind+N_d3*(a1primeindexesfine-1); % [N_d13,n2long,N_a1,N_a2]
    entireRHS_ii=reshape(ReturnMatrix_ii+reshape(DiscountedEVinterp(da1prime),[N_d13,n2long,N_a1,N_a2]),[N_d13*n2long,N_a1*N_a2]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    d3_ind=rem(maxindexL2-1,N_d13)+1;
    allind=d3_ind+N_d13*aind;
    Policy(1,:,N_j)=rem(d3_ind-1,N_d1)+1; % d1
    Policy(3,:,N_j)=rem(ceil(d3_ind/N_d1)-1,N_d3)+1; % d3
    Policy(4,:,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);
    Policy(5,:,N_j)=shiftdim(ceil(maxindexL2/N_d13),-1);

    % L2flag
    L2offset      = ceil(maxindexL2/N_d13);
    linidx_lower  = d3_ind                    + N_d13*n2long*aind;
    linidx_upper  = d3_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
    ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2]);
    isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    % Get the d2Policy
    d3part=rem(ceil(shiftdim(d3_ind,1)/N_d1)-1,N_d3)+1; % [N_a,1]
    a1mid=squeeze(midpoint_jj(allind));
    linlookup=d3part+N_d3*(a1mid-1);
    Policy(2,:,N_j)=shiftdim(d2index_resh(linlookup),-1);
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
    EV=V(:,jj+1);
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a2primeProbs,N_a1,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    % EV now

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1]),[],1);
    EV=reshape(EV,[N_d3*N_a1,1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,1,1]);
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]); % [N_d3,N_a1prime,1,1]

    % Layer 1
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, ReturnFnParamsVec,1,0);
    % [N_d13, level1n, N_a1, N_a2]; broadcast DiscountedEV across d1 and a2
    RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2]);
    DEV=reshape(DiscountedEV,[1,N_d3,1,N_a1,1]);
    entireRHS_ii=RM+DEV;
    entireRHS_ii=reshape(entireRHS_ii,[N_d13,vfoptions.level1n,N_a1,N_a2]);

    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoint_jj(:,1,level1ii,:)=maxindex1;

    % Divide and conquer layer 2
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, ReturnFnParamsVec,3,0);
            d3aprime=d3ind+N_d3*(a1primeindexes-1); % [N_d13,maxgap+1,1,N_a2]
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprime);
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoint_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:);
            midpoint_jj(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
        end
    end

    % Grid interpolation layer
    midpoint_jj=max(min(midpoint_jj,n_a1(1)-1),2);
    a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, ReturnFnParamsVec,2,0);
    da1prime=d3ind+N_d3*(a1primeindexesfine-1); % [N_d13,n2long,N_a1,N_a2]
    entireRHS_ii=reshape(ReturnMatrix_ii+reshape(DiscountedEVinterp(da1prime),[N_d13,n2long,N_a1,N_a2]),[N_d13*n2long,N_a1*N_a2]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,jj)=shiftdim(Vtempii,1);
    d3_ind=rem(maxindexL2-1,N_d13)+1;
    allind=d3_ind+N_d13*aind;
    Policy(1,:,jj)=rem(d3_ind-1,N_d1)+1; % d1
    Policy(3,:,jj)=rem(ceil(d3_ind/N_d1)-1,N_d3)+1; % d3
    Policy(4,:,jj)=shiftdim(squeeze(midpoint_jj(allind)),-1);
    Policy(5,:,jj)=shiftdim(ceil(maxindexL2/N_d13),-1);

    % L2flag
    L2offset      = ceil(maxindexL2/N_d13);
    linidx_lower  = d3_ind                    + N_d13*n2long*aind;
    linidx_upper  = d3_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
    ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2]);
    isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,jj) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    % Get the d2Policy
    d3part=rem(ceil(shiftdim(d3_ind,1)/N_d1)-1,N_d3)+1; % [N_a,1]
    a1mid=squeeze(midpoint_jj(allind));
    linlookup=d3part+N_d3*(a1mid-1);
    Policy(2,:,jj)=shiftdim(d2index_resh(linlookup),-1);

end


%% Switch Policy(4,:) from 'midpoint' to 'lower grid index'
adjust=(Policy(5,:,)<1+n2short+1);
Policy(4,:,)=Policy(4,:,)-adjust;
Policy(5,:,)=adjust.*Policy(5,:,)+(1-adjust).*(Policy(5,:,)-n2short-1);

Policy=[Policy; PolicyL2flag];

end
