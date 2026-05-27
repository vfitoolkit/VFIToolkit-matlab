function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
%
% Strategy: pre-refine d2 out of EV (max over d2 for each d3,a1prime,a2,z),
% then apply the ExpAssetu-style GI midpoint+L2 pattern with d as d13 (d1 broadcast over Return).

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_u=prod(n_u);

% For ReturnFn (d1 and d3 only)
n_d13=[n_d1,n_d3];
N_d13=N_d1*N_d3;
d13_grid=[d1_grid;d3_grid];
% For aprimeFn (d2 and d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d13 (combined d1*d3) and a1prime
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
Policyd2=ones(1,N_a,N_z,N_j,'gpuArray'); % d2 index recovered via lookup after GI search

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid; % already a column vector
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

pi_u_col=pi_u(:); % column

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1
zind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1
zindB=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1

d3ind=repelem((1:1:N_d3)',N_d1,1); % [N_d13,1]; maps full d13-index to d3-component (used for d2 lookup)

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No EV; d2 is meaningless. Just do GI search on Return alone (treat as ExpAssetu_GI with d=d13).
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,n_z, d13_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % [N_d13,N_a1prime,N_a1,N_a2,N_z]; Level=1, Refine=0
        [~,maxindex]=max(ReturnMatrix,[],2);

        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_z, d13_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d13,n2long,N_a1,N_a2,N_z]; Level=2, Refine=0
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d13)+1;
        allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
        Policy3(1,:,:,N_j)=d_ind; % d13 combined
        Policy3(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy3(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d13),-1); % a1primeL2ind
        % L2 flag
        L2offset      = ceil(maxindexL2/N_d13);
        linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
        linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        % d2 meaningless at j=N_j (no future), leave Policyd2=1

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,special_n_z, d13_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);
            [~,maxindex]=max(ReturnMatrix_z,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,special_n_z, d13_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind;
            Policy3(1,:,z_c,N_j)=d_ind;
            Policy3(2,:,z_c,N_j)=midpoint(allind);
            Policy3(3,:,z_c,N_j)=ceil(maxindexL2/N_d13);
            L2offset      = ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
            isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,z_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    % a2primeIndex,a2primeProbs are [N_d23,N_a2,N_u]

    [V(:,:,N_j),Policy3(:,:,:,N_j),PolicyL2flag(:,:,:,N_j),Policyd2(:,:,:,N_j)]=internal_per_j(EVpre,a2primeIndex,a2primeProbs,...
        ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_a1,n_a2,n_z,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_z,N_u,N_a1prime,...
        d13_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,z_gridvals_J(:,:,N_j),pi_z_J(:,:,N_j),pi_u_col,...
        aind,zind,zindB,a2ind,d3ind,n2short,n2long,vfoptions);
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
    [V(:,:,jj),Policy3(:,:,:,jj),PolicyL2flag(:,:,:,jj),Policyd2(:,:,:,jj)]=internal_per_j(EVnext,a2primeIndex,a2primeProbs,...
        ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_a1,n_a2,n_z,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_z,N_u,N_a1prime,...
        d13_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,z_gridvals_J(:,:,jj),pi_z_J(:,:,jj),pi_u_col,...
        aind,zind,zindB,a2ind,d3ind,n2short,n2long,vfoptions);
end


%% With grid interpolation, switch midpoint -> lower grid point
adjust=(Policy3(3,:,:,:)<1+n2short+1);
Policy3(2,:,:,:)=Policy3(2,:,:,:)-adjust;
Policy3(3,:,:,:)=adjust.*Policy3(3,:,:,:)+(1-adjust).*(Policy3(3,:,:,:)-n2short-1);

%% Decompose d13 into d1 and d3 components, combine with d2 lookup
d13opt=Policy3(1,:,:,:);
d1part=rem(d13opt-1,N_d1)+1;
d3part=rem(ceil(d13opt/N_d1)-1,N_d3)+1;
d2part=Policyd2(1,:,:,:);

% Combined index: d1 + N_d1*(d2-1) + N_d1*N_d2*(d3-1) + N_d1*N_d2*N_d3*(a1prime_low-1) + N_d1*N_d2*N_d3*N_a1*(L2ind-1) + N_d1*N_d2*N_d3*N_a1*(n2short+2)*(PolicyL2flag-1)
N_d=N_d1*N_d2*N_d3;
Policy=shiftdim(d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d*(Policy3(2,:,:,:)-1)+N_d*N_a1*(Policy3(3,:,:,:)-1)+N_d*N_a1*(n2short+2)*(PolicyL2flag-1),1);


end


%% Per-period inner: compute V, Policy3, PolicyL2flag, Policyd2 at age jj using EVnext
function [V_jj,Policy3_jj,PolicyL2flag_jj,Policyd2_jj]=internal_per_j(EVnext,a2primeIndex,a2primeProbs,...
    ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d1,n_d3,n_a1,n_a2,n_z,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_z,N_u,N_a1prime,...
    d13_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,z_gridvals,pi_z,pi_u_col,...
    aind,zind,zindB,a2ind,d3ind,n2short,n2long,vfoptions)

V_jj=zeros(N_a,N_z,'gpuArray');
Policy3_jj=zeros(3,N_a,N_z,'gpuArray');
PolicyL2flag_jj=2*ones(1,N_a,N_z,'gpuArray');
Policyd2_jj=ones(1,N_a,N_z,'gpuArray');

% (a1prime,a2prime) interpolation indexes for full (d23,a1prime)
aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d23*N_a1,N_u] (a2 dim collapsed via column-major)
aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d23*N_a1,N_u]

% Compute EV: integrate zprime first
EV=EVnext.*shiftdim(pi_z',-1); % [N_a,N_z,N_z(prime)]
EV(isnan(EV))=0;
EV=sum(EV,2); % [N_a,1,N_z]
EV=reshape(EV,[N_a,N_z]); % [N_a,N_z]

skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
aprimeProbs=repmat(a2primeProbs,N_a1,N_z); % [N_d23*N_a1, N_u*N_z]
aprimeProbs(skipinterp)=0;
aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_z]);

EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbs;
EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbs);
EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2); % [N_d23*N_a1,1,N_z]
EV=reshape(EV,[N_d23*N_a1,N_z]); % [N_d23*N_a1,N_z]

% Refine d2: max over d2 for each (d3,a1prime,z)
EVres=reshape(EV,[N_d2,N_d3*N_a1,N_z]);
[EV_onlyd3,d2index]=max(EVres,[],1); % [1,N_d3*N_a1,N_z]
EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_z]);
d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]); % [N_d3,N_a1,N_z]

DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_z]); % [N_d3,N_a1,1,1,N_z]
% Interpolate EV_onlyd3 over a1prime grid (fine) — interp1 along the N_a1 dim
DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d3,N_a1prime,1,1,N_z]

% Broadcast d1 onto DiscountedEV by repelem along the first dim — turns N_d3 into N_d13 with d1 fastest
DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);% [N_d1*N_d3,N_a1,1,1,N_z] = [N_d13,N_a1,1,1,N_z]
DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1); % [N_d13,N_a1prime,1,1,N_z]

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,n_z, d13_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals, ReturnFnParamsVec,1,0); % [N_d13,N_a1prime,N_a1,N_a2,N_z]; Level=1, Refine=0

    entireRHS=ReturnMatrix+DiscountedEV_d13; % broadcast over a2 (which is dim 4)

    [~,maxindex]=max(entireRHS,[],2);

    midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals, ReturnFnParamsVec,2,0); % [N_d13,n2long,N_a1,N_a2,N_z]; Level=2, Refine=0
    % EV does not depend on a2, so the linear index into DiscountedEVinterp_d13 [N_d13,N_a1prime,1,1,N_z] omits an a2 offset
    % a1primeindexesfine has shape [N_d13,n2long,N_a1,N_a2,N_z]; index value is identical across the a2 axis
    da1primez=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1)+N_d13*N_a1prime*zind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_d13(da1primez(:)),[N_d13*n2long,N_a1*N_a2,N_z]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V_jj=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d13)+1; % d13 combined
    allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
    Policy3_jj(1,:,:)=d_ind; % d13
    Policy3_jj(2,:,:)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
    Policy3_jj(3,:,:)=shiftdim(ceil(maxindexL2/N_d13),-1); % a1primeL2ind
    % L2 flag
    L2offset      = ceil(maxindexL2/N_d13);
    linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
    linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag_jj(1,:,:) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
    % d2 lookup: d2index_resh(d3, a1prime_midpoint, z)
    d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a,N_z]
    a1opt_mid=midpoint(allind); % [1,N_a,N_z] a1prime midpoint at chosen d
    zlin=shiftdim(gpuArray(0:N_z-1),-1); % [1,1,N_z]
    lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin; % [1,N_a,N_z]
    Policyd2_jj(1,:,:)=d2index_resh(lin);

elseif vfoptions.lowmemory==1

    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV_d13(:,:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp_d13(:,:,:,:,z_c);
        d2index_z=d2index_resh(:,:,z_c); % [N_d3,N_a1]

        ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,ones(1,length(n_z)), d13_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);

        entireRHS_z=ReturnMatrix_z+DiscountedEV_z;

        [~,maxindex]=max(entireRHS_z,[],2);

        midpoint=max(min(maxindex,n_a1(1)-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,ones(1,length(n_z)), d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
        % EV does not depend on a2; index into DiscountedEVinterp_z [N_d13,N_a1prime,1,1]
        da1prime=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1);
        entireRHS_ii=ReturnMatrix_ii_z+reshape(DiscountedEVinterp_z(da1prime(:)),[N_d13*n2long,N_a1*N_a2]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V_jj(:,z_c)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d13)+1;
        allind=d_ind+N_d13*aind;
        Policy3_jj(1,:,z_c)=d_ind;
        Policy3_jj(2,:,z_c)=midpoint(allind);
        Policy3_jj(3,:,z_c)=ceil(maxindexL2/N_d13);
        L2offset      = ceil(maxindexL2/N_d13);
        linidx_lower  = d_ind                    + N_d13*n2long*aind;
        linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
        isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag_jj(1,:,z_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        % d2 lookup
        d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a]
        a1opt_mid=midpoint(allind); % [1,N_a]
        lin=d3opt+N_d3*(a1opt_mid-1);
        Policyd2_jj(1,:,z_c)=d2index_z(lin);
    end
end

end
