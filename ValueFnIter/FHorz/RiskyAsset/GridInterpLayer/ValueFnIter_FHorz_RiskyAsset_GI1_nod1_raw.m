function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_GI1_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_u,N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No d1.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_u=prod(n_u);

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_z,N_j,'gpuArray'); % slot 1 = d3 (no d1)
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray');
Policyd2=ones(1,N_a,N_z,N_j,'gpuArray');

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d3_gridvals=CreateGridvals(n_d3,d3_grid,1);

pi_u_col=pi_u(:);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);
zindB=shiftdim(gpuArray(0:1:N_z-1),-1);

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_z, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        [~,maxindex]=max(ReturnMatrix,[],2);

        midpoint=max(min(maxindex,n_a1(1)-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_z, d3_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d_ind+N_d3*aind+N_d3*N_a*zindB;
        Policy3(1,:,:,N_j)=d_ind;
        Policy3(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
        linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,special_n_z, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);
            [~,maxindex]=max(ReturnMatrix_z,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,special_n_z, d3_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d_ind+N_d3*aind;
            Policy3(1,:,z_c,N_j)=d_ind;
            Policy3(2,:,z_c,N_j)=midpoint(allind);
            Policy3(3,:,z_c,N_j)=ceil(maxindexL2/N_d3);
            L2offset      = ceil(maxindexL2/N_d3);
            linidx_lower  = d_ind                   + N_d3*n2long*aind;
            linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
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

    [V(:,:,N_j),Policy3(:,:,:,N_j),PolicyL2flag(:,:,:,N_j),Policyd2(:,:,:,N_j)]=internal_per_j_nod1(EVpre,a2primeIndex,a2primeProbs,...
        ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d3,n_a1,n_a2,n_z,N_d2,N_d3,N_d23,N_a1,N_a2,N_a,N_z,N_u,N_a1prime,...
        d3_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,z_gridvals_J(:,:,N_j),pi_z_J(:,:,N_j),pi_u_col,...
        aind,zind,zindB,a2ind,n2short,n2long,vfoptions);
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
    [V(:,:,jj),Policy3(:,:,:,jj),PolicyL2flag(:,:,:,jj),Policyd2(:,:,:,jj)]=internal_per_j_nod1(EVnext,a2primeIndex,a2primeProbs,...
        ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d3,n_a1,n_a2,n_z,N_d2,N_d3,N_d23,N_a1,N_a2,N_a,N_z,N_u,N_a1prime,...
        d3_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,z_gridvals_J(:,:,jj),pi_z_J(:,:,jj),pi_u_col,...
        aind,zind,zindB,a2ind,n2short,n2long,vfoptions);
end


%% Adjust midpoint -> lower grid point
adjust=(Policy3(3,:,:,:)<1+n2short+1);
Policy3(2,:,:,:)=Policy3(2,:,:,:)-adjust;
Policy3(3,:,:,:)=adjust.*Policy3(3,:,:,:)+(1-adjust).*(Policy3(3,:,:,:)-n2short-1);

%% Combine. No d1 here. d3 in Policy3(1), d2 in Policyd2.
% Combined index: d2 + N_d2*(d3-1) + N_d2*N_d3*(a1prime_low-1) + N_d2*N_d3*N_a1*(L2ind-1) + N_d2*N_d3*N_a1*(n2short+2)*(PolicyL2flag-1)
N_d=N_d2*N_d3;
Policy=shiftdim(Policyd2(1,:,:,:)+N_d2*(Policy3(1,:,:,:)-1)+N_d*(Policy3(2,:,:,:)-1)+N_d*N_a1*(Policy3(3,:,:,:)-1)+N_d*N_a1*(n2short+2)*(PolicyL2flag-1),1);


end


%% Per-period inner (no d1)
function [V_jj,Policy3_jj,PolicyL2flag_jj,Policyd2_jj]=internal_per_j_nod1(EVnext,a2primeIndex,a2primeProbs,...
    ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d3,n_a1,n_a2,n_z,N_d2,N_d3,N_d23,N_a1,N_a2,N_a,N_z,N_u,N_a1prime,...
    d3_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,z_gridvals,pi_z,pi_u_col,...
    aind,zind,zindB,a2ind,n2short,n2long,vfoptions)

V_jj=zeros(N_a,N_z,'gpuArray');
Policy3_jj=zeros(3,N_a,N_z,'gpuArray');
PolicyL2flag_jj=2*ones(1,N_a,N_z,'gpuArray');
Policyd2_jj=ones(1,N_a,N_z,'gpuArray');

aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

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

EVres=reshape(EV,[N_d2,N_d3*N_a1,N_z]);
[EV_onlyd3,d2index]=max(EVres,[],1);
EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_z]);
d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_z]);
DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d3,N_a1prime,1,1,N_z]

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_z, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals, ReturnFnParamsVec,1,0);

    entireRHS=ReturnMatrix+DiscountedEV;

    [~,maxindex]=max(entireRHS,[],2);

    midpoint=max(min(maxindex,n_a1(1)-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_z, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals, ReturnFnParamsVec,2,0);
    da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*zind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_z]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V_jj=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d3)+1;
    allind=d_ind+N_d3*aind+N_d3*N_a*zindB;
    Policy3_jj(1,:,:)=d_ind;
    Policy3_jj(2,:,:)=shiftdim(squeeze(midpoint(allind)),-1);
    Policy3_jj(3,:,:)=shiftdim(ceil(maxindexL2/N_d3),-1);
    L2offset      = ceil(maxindexL2/N_d3);
    linidx_lower  = d_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag_jj(1,:,:) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
    % d2 lookup: d3 directly (no d1 to strip)
    d3opt=d_ind; % [1,N_a,N_z]
    a1opt_mid=midpoint(allind); % [1,N_a,N_z]
    zlin=shiftdim(gpuArray(0:N_z-1),-1); % [1,1,N_z]
    lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
    Policyd2_jj(1,:,:)=d2index_resh(lin);

elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);
        d2index_z=d2index_resh(:,:,z_c);

        ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,ones(1,length(n_z)), d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);

        entireRHS_z=ReturnMatrix_z+DiscountedEV_z;

        [~,maxindex]=max(entireRHS_z,[],2);

        midpoint=max(min(maxindex,n_a1(1)-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,ones(1,length(n_z)), d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
        da1prime=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1);
        entireRHS_ii=ReturnMatrix_ii_z+reshape(DiscountedEVinterp_z(da1prime(:)),[N_d3*n2long,N_a1*N_a2]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V_jj(:,z_c)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d_ind+N_d3*aind;
        Policy3_jj(1,:,z_c)=d_ind;
        Policy3_jj(2,:,z_c)=midpoint(allind);
        Policy3_jj(3,:,z_c)=ceil(maxindexL2/N_d3);
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d_ind                   + N_d3*n2long*aind;
        linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
        isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag_jj(1,:,z_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        % d2 lookup
        d3opt=d_ind;
        a1opt_mid=midpoint(allind);
        lin=d3opt+N_d3*(a1opt_mid-1);
        Policyd2_jj(1,:,z_c)=d2index_z(lin);
    end
end

end
