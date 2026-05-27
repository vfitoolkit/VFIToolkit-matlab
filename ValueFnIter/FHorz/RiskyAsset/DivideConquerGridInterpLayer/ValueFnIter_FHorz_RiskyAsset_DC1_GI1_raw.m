function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
%
% Splices:
%   - RiskyAsset DC: u-shock + refine_d (d1/d2/d3) + d2-refinement trick (max over d2 on EV)
%   - ExpAsset DC_GI: outer DC level1ii n-Monotonicity + inner GI midpoint+L2 fine grid
%   - ExpAsset GI: L2flag scaffold (PolicyL2flag with -Inf neighbour detection)

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
Policy3=zeros(3,N_a,N_z,N_j,'gpuArray'); % (1)=d13 index, (2)=a1prime midpoint, (3)=a1primeL2ind
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper
d2Policy=ones(1,N_a,N_z,N_j,'gpuArray'); % d2 chosen via lookup after the DC+GI search

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

pi_u_col=pi_u(:);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
    midpoint=zeros(N_d13,1,N_a1,N_a2,'gpuArray');
else
    midpoint=zeros(N_d13,1,N_a1,N_a2,N_z,'gpuArray');
end

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);
zindB=shiftdim(gpuArray(0:1:N_z-1),-1);

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
a2Bind=gpuArray(0:1:N_a2-1);
d3ind=repelem(gpuArray(1:1:N_d3)',N_d1,1); % [N_d13,1]; maps full d13-index to d3-component

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity (Return only, since no V_Jplus1)
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoint(:,1,level1ii,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Turn this into the 'midpoint' and run the fine GI search
        midpoint=max(min(midpoint,n_a1(1)-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        [Vtempii,maxindexL2]=max(reshape(ReturnMatrix_ii,[N_d13*n2long,N_a1*N_a2,N_z]),[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d13)+1; % d13 index
        allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
        Policy3(1,:,:,N_j)=d_ind;
        Policy3(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d13),-1);

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset      = ceil(maxindexL2/N_d13);
        linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
        linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
        ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_z]);
        isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        % d2 meaningless at terminal age without V_Jplus1; leave d2Policy=1.

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);

            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            midpoint(:,1,level1ii,:)=maxindex1;

            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                    [~,maxindex]=max(ReturnMatrix_ii_z,[],2);
                    midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,special_n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(reshape(ReturnMatrix_ii,[N_d13*n2long,N_a1*N_a2]),[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind;
            Policy3(1,:,z_c,N_j)=d_ind;
            Policy3(2,:,z_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3(3,:,z_c,N_j)=shiftdim(ceil(maxindexL2/N_d13),-1);

            L2offset      = ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
            ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2]);
            isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,z_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    [V(:,:,N_j),Policy3(:,:,:,N_j),PolicyL2flag(:,:,:,N_j),d2Policy(:,:,:,N_j)]=internal_per_j(EVpre,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_a1,n_a2,n_z,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a1prime,N_a2,N_a,N_z,N_u,...
        d13_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,z_gridvals_J(:,:,N_j),pi_z_J(:,:,N_j),pi_u_col,...
        level1ii,level1iidiff,n2short,n2long,aind,a2ind,a2Bind,zind,zindB,d3ind,vfoptions);
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
    [V(:,:,jj),Policy3(:,:,:,jj),PolicyL2flag(:,:,:,jj),d2Policy(:,:,:,jj)]=internal_per_j(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_a1,n_a2,n_z,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a1prime,N_a2,N_a,N_z,N_u,...
        d13_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,z_gridvals_J(:,:,jj),pi_z_J(:,:,jj),pi_u_col,...
        level1ii,level1iidiff,n2short,n2long,aind,a2ind,a2Bind,zind,zindB,d3ind,vfoptions);
end


%% With grid interpolation, switch Policy(2,:) from 'midpoint' to 'lower grid index'
adjust=(Policy3(3,:,:,:)<1+n2short+1);
Policy3(2,:,:,:)=Policy3(2,:,:,:)-adjust;
Policy3(3,:,:,:)=adjust.*Policy3(3,:,:,:)+(1-adjust).*(Policy3(3,:,:,:)-n2short-1);

%% Encode single-index Policy (Case2 FHorz) with d2 lookup and L2flag included.
% Policy3(1,:) holds d13 index; decompose into d1 (fastest) and d3 components.
d1part=rem(Policy3(1,:,:,:)-1,N_d1)+1;
d3part=rem(ceil(Policy3(1,:,:,:)/N_d1)-1,N_d3)+1;
d2part=d2Policy;
N_d=N_d1*N_d2*N_d3;
Policy=shiftdim(d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d1*N_d2*N_d3*(Policy3(2,:,:,:)-1)+N_d*N_a1*(Policy3(3,:,:,:)-1)+N_d*N_a1*(n2short+2)*(PolicyL2flag-1),1);

end


%% Per-period inner: V/Policy3/PolicyL2flag/d2Policy at age jj given EVnext (= V(:,:,jj+1) or V_Jplus1)
function [V_jj,Policy3_jj,PolicyL2flag_jj,d2Policy_jj]=internal_per_j(EVnext,a2primeIndex,a2primeProbs,ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d1,n_d3,n_a1,n_a2,n_z,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a1prime,N_a2,N_a,N_z,N_u,...
    d13_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,z_gridvals,pi_z,pi_u_col,...
    level1ii,level1iidiff,n2short,n2long,aind,a2ind,a2Bind,zind,zindB,d3ind,vfoptions)

V_jj=zeros(N_a,N_z,'gpuArray');
Policy3_jj=zeros(3,N_a,N_z,'gpuArray');
PolicyL2flag_jj=2*ones(1,N_a,N_z,'gpuArray');
d2Policy_jj=ones(1,N_a,N_z,'gpuArray');

% (a1prime,a2prime) interpolation indexes for full (d23,a1prime,u)
aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

% Compute EV integrated over u and zprime
EV=EVnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0;
EV=sum(EV,2);
EV=reshape(EV,[N_a,N_z]);

skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
aprimeProbs=repmat(a2primeProbs,N_a1,N_z);
aprimeProbs(skipinterp)=0;
aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_z]);

EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbs;
EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbs);
EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
EV=reshape(EV,[N_d23*N_a1,N_z]);

% Refine d2 out: maximize EV over d2 to get EV_onlyd3 [N_d3*N_a1, N_z] and d2index recording argmax d2
EVres=reshape(EV,[N_d2,N_d3*N_a1,N_z]);
[EV_onlyd3,d2index]=max(EVres,[],1);
EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_z]);
d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_z]);
% Interpolate EV over a1prime fine grid
DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);   % [N_d3,N_a1prime,1,1,N_z]

if vfoptions.lowmemory==0
    midpoint=zeros(N_d13,1,N_a1,N_a2,N_z,'gpuArray');

    % n-Monotonicity (coarse DC search at level1ii midpoints)
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals, ReturnFnParamsVec,1,0);
    % [N_d13, level1n, N_a1, N_a2, N_z]; broadcast DiscountedEV [N_d3,N_a1,1,1,N_z] over d1 and a2
    RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2,N_z]);
    DEV=reshape(DiscountedEV,[1,N_d3,1,N_a1,1,N_z]);
    entireRHS_ii=RM+DEV;
    entireRHS_ii=reshape(entireRHS_ii,[N_d13,vfoptions.level1n,N_a1,N_a2,N_z]);

    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoint(:,1,level1ii,:,:)=maxindex1;

    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals, ReturnFnParamsVec,3,0);
            d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*zindB; % [N_d13,maxgap+1,1,N_a2,N_z]; lin idx into DiscountedEV [N_d3,N_a1,1,1,N_z]
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
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % [N_d13,n2long,N_a1,N_a2,N_z]
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals, ReturnFnParamsVec,2,0);
    da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*zindB; % lin idx into DiscountedEVinterp [N_d3,N_a1prime,1,1,N_z]
    entireRHS_ii=reshape(ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_z]),[N_d13*n2long,N_a1*N_a2,N_z]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V_jj(:,:)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d13)+1;
    allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
    Policy3_jj(1,:,:)=d_ind; % d13 index
    Policy3_jj(2,:,:)=shiftdim(squeeze(midpoint(allind)),-1);
    Policy3_jj(3,:,:)=shiftdim(ceil(maxindexL2/N_d13),-1);

    % L2 flag detection on the coarse a1 neighbours
    L2offset      = ceil(maxindexL2/N_d13);
    linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
    linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
    ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_z]);
    isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag_jj(1,:,:) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    % d2 lookup: d2 = d2index_resh(d3part, a1primemidpoint, z)
    d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1; % [N_a, N_z]
    a1mid=squeeze(midpoint(allind)); % [N_a, N_z]
    zidx=repmat(gpuArray(1:N_z),N_a,1);
    linlookup=d3part+N_d3*(a1mid-1)+N_d3*N_a1*(zidx-1);
    d2Policy_jj(1,:,:)=shiftdim(d2index_resh(linlookup),-1);

elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d13,1,N_a1,N_a2,'gpuArray');
    special_n_z=ones(1,length(n_z));
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c); % [N_d3,N_a1,1,1]
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c); % [N_d3,N_a1prime,1,1]
        d2index_z=d2index_resh(:,:,z_c); % [N_d3,N_a1]

        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
        RM=reshape(ReturnMatrix_ii_z,[N_d1,N_d3,vfoptions.level1n,N_a1,N_a2]);
        DEV=reshape(DiscountedEV_z,[1,N_d3,1,N_a1,1]);
        entireRHS_ii_z=RM+DEV;
        entireRHS_ii_z=reshape(entireRHS_ii_z,[N_d13,vfoptions.level1n,N_a1,N_a2]);

        [~,maxindex1]=max(entireRHS_ii_z,[],2);
        midpoint(:,1,level1ii,:)=maxindex1;

        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                d3aprime=d3ind+N_d3*(a1primeindexes-1); % [N_d13,maxgap+1,1,N_a2]
                entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z(d3aprime);
                [~,maxindex]=max(entireRHS_ii_z,[],2);
                midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        midpoint=max(min(midpoint,n_a1(1)-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,special_n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
        da1prime=d3ind+N_d3*(a1primeindexesfine-1); % [N_d13,n2long,N_a1,N_a2]
        entireRHS_ii=reshape(ReturnMatrix_ii+reshape(DiscountedEVinterp_z(da1prime),[N_d13,n2long,N_a1,N_a2]),[N_d13*n2long,N_a1*N_a2]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V_jj(:,z_c)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d13)+1;
        allind=d_ind+N_d13*aind;
        Policy3_jj(1,:,z_c)=d_ind;
        Policy3_jj(2,:,z_c)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy3_jj(3,:,z_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

        L2offset      = ceil(maxindexL2/N_d13);
        linidx_lower  = d_ind                    + N_d13*n2long*aind;
        linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
        ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2]);
        isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag_jj(1,:,z_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

        d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
        a1mid=squeeze(midpoint(allind));
        linlookup=d3part+N_d3*(a1mid-1);
        d2Policy_jj(1,:,z_c)=shiftdim(d2index_z(linlookup),-1);
    end
end

end
