function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_a3,n_u,N_j, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAsset_DC1_GI1_nod1_noz_raw.
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No d1, no z.
%
% a1: standard endogenous state, this is the one divide-and-conquer (and then the grid interp layer) is applied to
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% The EV pipeline is unchanged from the DC1_GI1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset. DiscountedEV is (d3,a1prime,a2prime) with no a3 term.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_u=prod(n_u);

N_a12=N_a1*N_a2; % the two standard assets, carried forward directly

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(5,N_a,N_j,'gpuArray'); % (1)=d2, (2)=d3, (3)=a1prime midpoint, (4)=a2prime, (5)=a1prime L2
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray');
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d3_gridvals=CreateGridvals(n_d3,d3_grid,1);

% Setup for DC (over a1 only)
midpoint=zeros(N_d3,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Setup for GI
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% Precompute
aind=gpuArray(0:1:N_a-1);
d3col=(1:1:N_d3)';                     % [N_d3,1]
a2pcol=reshape(0:1:N_a2-1,[1,1,N_a2]); % [1,1,N_a2prime]

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % Layer 1
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d3, n_a2, n_a3, d3_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    % [N_d3, N_a1prime, N_a2prime, level1n, N_a2, N_a3]
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    midpoint(:,1,:,level1ii,:,:)=maxindex1;

    % Divide-and-conquer layer 2
    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % [N_d3,1,N_a2prime,1,N_a2,N_a3]
            a1primeindexes=loweredge+(0:1:maxgap(ii));              % [N_d3,maxgap+1,N_a2prime,1,N_a2,N_a3]
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d3, n_a2, n_a3, d3_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
            [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
        end
    end

    % Grid interpolation layer
    midpoint=max(min(midpoint,N_a1-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d3, n_a2, n_a3, d3_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);

    d3_ind      =rem(maxindexL2-1,N_d3)+1;
    maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
    maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

    allind=d3_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind;
    Policy(2,:,N_j)=d3_ind;
    Policy(3,:,N_j)=midpoint(allind);
    Policy(4,:,N_j)=maxindexL2a2;
    Policy(5,:,N_j)=maxindexL2a1;

    % L2flag
    linidx_lower=d3_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
    linidx_upper=d3_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    PolicyL2flag(1,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    % d2, which was not in ReturnFn
    Policy(1,:,N_j)=ones(1,N_a,'gpuArray'); % d2 (terminal: d2 doesn't matter since it's only in the expectations term)

else % V_Jplus1

    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states
    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a3primeProbs,N_a12,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a12,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a12,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a12]),[],1);
    EV=reshape(EV,[N_d3*N_a12,1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2]);

    % DiscountedEV: (d3,a1prime,a2prime), broadcast against the (a1,a2,a3) dims of ReturnMatrix_ii
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,N_a2]);
    % Interpolate EV over a1prime_grid
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3]),a1prime_grid),[2,1,3]); % [N_d3,N_a1fine,N_a2]

    % Layer 1
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d3, n_a2, n_a3, d3_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoint(:,1,:,level1ii,:,:)=maxindex1;

    % Divide and conquer layer 2
    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d3, n_a2, n_a3, d3_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
            d3aprime=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol; % linear index into DiscountedEV [N_d3,N_a1,N_a2]
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprime);
            [~,maxindex_inner]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
        end
    end

    % Grid interpolation layer
    midpoint=max(min(midpoint,N_a1-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d3, n_a2, n_a3, d3_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
    aprime=d3col + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol; % linear index into DiscountedEVinterp [N_d3,N_a1fine,N_a2]
    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprime),[N_d3*n2long*N_a2,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);

    d3_ind      =rem(maxindexL2-1,N_d3)+1;
    maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
    maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

    allind=d3_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind;
    a1mid=midpoint(allind);
    Policy(2,:,N_j)=d3_ind;
    Policy(3,:,N_j)=a1mid;
    Policy(4,:,N_j)=maxindexL2a2;
    Policy(5,:,N_j)=maxindexL2a1;

    % L2flag
    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d3*n2long*N_a2,N_a]);
    linidx_lower=d3_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
    linidx_upper=d3_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    PolicyL2flag(1,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    % Get the d2Policy
    lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
    Policy(1,:,N_j)=d2index_resh(lin);
end

%% Iterate backwards
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj));

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states
    EV=V(:,jj+1);
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a3primeProbs,N_a12,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a12,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a12,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a12]),[],1);
    EV=reshape(EV,[N_d3*N_a12,1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,N_a2]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3]),a1prime_grid),[2,1,3]); % [N_d3,N_a1fine,N_a2]

    % Layer 1
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d3, n_a2, n_a3, d3_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoint(:,1,:,level1ii,:,:)=maxindex1;

    % Divide and conquer layer 2
    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d3, n_a2, n_a3, d3_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
            d3aprime=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprime);
            [~,maxindex_inner]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
        end
    end

    % Grid interpolation layer
    midpoint=max(min(midpoint,N_a1-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d3, n_a2, n_a3, d3_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
    aprime=d3col + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol;
    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprime),[N_d3*n2long*N_a2,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,jj)=shiftdim(Vtempii,1);

    d3_ind      =rem(maxindexL2-1,N_d3)+1;
    maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
    maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

    allind=d3_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind;
    a1mid=midpoint(allind);
    Policy(2,:,jj)=d3_ind;
    Policy(3,:,jj)=a1mid;
    Policy(4,:,jj)=maxindexL2a2;
    Policy(5,:,jj)=maxindexL2a1;

    % L2flag
    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d3*n2long*N_a2,N_a]);
    linidx_lower=d3_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
    linidx_upper=d3_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    PolicyL2flag(1,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    % Get the d2Policy
    lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
    Policy(1,:,jj)=d2index_resh(lin);

end


%% Switch Policy(3,:) from 'midpoint' to 'lower grid index'
adjust=(Policy(5,:,:)<1+n2short+1);
Policy(3,:,:)=Policy(3,:,:)-adjust;
Policy(5,:,:)=adjust.*Policy(5,:,:)+(1-adjust).*(Policy(5,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

end
