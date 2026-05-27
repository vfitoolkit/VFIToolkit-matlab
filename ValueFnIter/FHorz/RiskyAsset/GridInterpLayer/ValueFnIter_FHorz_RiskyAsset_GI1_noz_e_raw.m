function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_GI1_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, e_gridvals_J, u_grid, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No z, with e: iid start-of-period. e treated as z-slot in Return helper (no z exists).

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_e=prod(n_e);
N_u=prod(n_u);

n_d13=[n_d1,n_d3];
N_d13=N_d1*N_d3;
d13_grid=[d1_grid;d3_grid];
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_e,N_j,'gpuArray');
Policyd2=ones(1,N_a,N_e,N_j,'gpuArray');

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

pi_u_col=pi_u(:);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e),'gpuArray');
end

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
eindB=shiftdim((0:1:N_e-1),-1); % [1,1,N_e] (treated like zindB)

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,n_e, d13_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % treat e as z (no z exists)
        [~,maxindex]=max(ReturnMatrix,[],2);

        midpoint=max(min(maxindex,n_a1(1)-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_e, d13_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d13)+1;
        allind=d_ind+N_d13*aind+N_d13*N_a*eindB;
        Policy3(1,:,:,N_j)=d_ind;
        Policy3(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d13),-1);
        L2offset      = ceil(maxindexL2/N_d13);
        linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*eindB;
        linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*eindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,special_n_e, d13_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,1,0);
            [~,maxindex]=max(ReturnMatrix_e,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,special_n_e, d13_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind;
            Policy3(1,:,e_c,N_j)=d_ind;
            Policy3(2,:,e_c,N_j)=midpoint(allind);
            Policy3(3,:,e_c,N_j)=ceil(maxindexL2/N_d13);
            L2offset      = ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
            isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);
    EVpre=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j),-1),2); % [N_a,1]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    [V(:,:,N_j),Policy3(:,:,:,N_j),PolicyL2flag(:,:,:,N_j),Policyd2(:,:,:,N_j)]=internal_per_j_noz_e(EVpre,a2primeIndex,a2primeProbs,...
        ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_a1,n_a2,n_e,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_e,N_u,N_a1prime,...
        d13_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,e_gridvals_J(:,:,N_j),pi_u_col,...
        aind,eindB,a2ind,n2short,n2long,vfoptions);
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

    EVnext=sum(V(:,:,jj+1).*shiftdim(pi_e_J(:,jj),-1),2); % [N_a,1]
    [V(:,:,jj),Policy3(:,:,:,jj),PolicyL2flag(:,:,:,jj),Policyd2(:,:,:,jj)]=internal_per_j_noz_e(EVnext,a2primeIndex,a2primeProbs,...
        ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
        n_d1,n_d3,n_a1,n_a2,n_e,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_e,N_u,N_a1prime,...
        d13_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,e_gridvals_J(:,:,jj),pi_u_col,...
        aind,eindB,a2ind,n2short,n2long,vfoptions);
end


%% Adjust midpoint -> lower
adjust=(Policy3(3,:,:,:)<1+n2short+1);
Policy3(2,:,:,:)=Policy3(2,:,:,:)-adjust;
Policy3(3,:,:,:)=adjust.*Policy3(3,:,:,:)+(1-adjust).*(Policy3(3,:,:,:)-n2short-1);

%% Decompose d13 -> (d1,d3), combine with d2 lookup
d13opt=Policy3(1,:,:,:);
d1part=rem(d13opt-1,N_d1)+1;
d3part=rem(ceil(d13opt/N_d1)-1,N_d3)+1;
d2part=Policyd2(1,:,:,:);

N_d=N_d1*N_d2*N_d3;
Policy=shiftdim(d1part+N_d1*(d2part-1)+N_d1*N_d2*(d3part-1)+N_d*(Policy3(2,:,:,:)-1)+N_d*N_a1*(Policy3(3,:,:,:)-1)+N_d*N_a1*(n2short+2)*(PolicyL2flag-1),1);


end


%% Per-period inner (noz, with e). e treated as z-slot in Return helper.
function [V_jj,Policy3_jj,PolicyL2flag_jj,Policyd2_jj]=internal_per_j_noz_e(EVnext,a2primeIndex,a2primeProbs,...
    ReturnFn,DiscountFactorParamsVec,ReturnFnParamsVec,...
    n_d1,n_d3,n_a1,n_a2,n_e,N_d1,N_d2,N_d3,N_d13,N_d23,N_a1,N_a2,N_a,N_e,N_u,N_a1prime,...
    d13_gridvals,a1_gridvals,a1prime_grid,a2_gridvals,e_gridvals,pi_u_col,...
    aind,eindB,a2ind,n2short,n2long,vfoptions)

V_jj=zeros(N_a,N_e,'gpuArray');
Policy3_jj=zeros(3,N_a,N_e,'gpuArray');
PolicyL2flag_jj=2*ones(1,N_a,N_e,'gpuArray');
Policyd2_jj=ones(1,N_a,N_e,'gpuArray');

aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

EV=EVnext(:); % [N_a,1]
skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
aprimeProbs=repmat(a2primeProbs,N_a1,1);
aprimeProbs(skipinterp)=0;
aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);

EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2); % [N_d23*N_a1,1]

EVres=reshape(EV,[N_d2,N_d3*N_a1]);
[EV_onlyd3,d2index]=max(EVres,[],1);
EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,1]);
d2index_resh=reshape(d2index,[N_d3,N_a1]);

DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1]); % [N_d3,N_a1,1,1]
DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]); % [N_d3,N_a1prime,1,1]
DiscountedEV_d13=repelem(DiscountedEV,N_d1,1); % [N_d13,N_a1,1,1]
DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1); % [N_d13,N_a1prime,1,1]

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,n_e, d13_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, e_gridvals, ReturnFnParamsVec,1,0); % e in z-slot; [N_d13,N_a1prime,N_a1,N_a2,N_e]

    entireRHS=ReturnMatrix+DiscountedEV_d13; % broadcast e

    [~,maxindex]=max(entireRHS,[],2);

    midpoint=max(min(maxindex,n_a1(1)-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,n_e, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_gridvals, ReturnFnParamsVec,2,0); % [N_d13,n2long,N_a1,N_a2,N_e]
    % EV does not depend on a2 nor e; index into [N_d13,N_a1prime,1,1]
    da1prime=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1);
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_d13(da1prime(:)),[N_d13*n2long,N_a1*N_a2,N_e]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V_jj=shiftdim(Vtempii,1); % [N_a,N_e]
    d_ind=rem(maxindexL2-1,N_d13)+1; % [1,N_a,N_e]
    allind=d_ind+N_d13*aind+N_d13*N_a*eindB;
    Policy3_jj(1,:,:)=d_ind;
    Policy3_jj(2,:,:)=shiftdim(squeeze(midpoint(allind)),-1);
    Policy3_jj(3,:,:)=shiftdim(ceil(maxindexL2/N_d13),-1);
    L2offset      = ceil(maxindexL2/N_d13);
    linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*eindB;
    linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*eindB;
    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag_jj(1,:,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    % d2 lookup (no z dim in d2index_resh)
    d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a,N_e]
    a1opt_mid=midpoint(allind); % [1,N_a,N_e]
    lin=d3opt+N_d3*(a1opt_mid-1); % [1,N_a,N_e]
    Policyd2_jj(1,:,:)=d2index_resh(lin);

elseif vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n_a1,n_a1,n_a2,special_n_e, d13_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,1,0);
        entireRHS_e=ReturnMatrix_e+DiscountedEV_d13;
        [~,maxindex]=max(entireRHS_e,[],2);

        midpoint=max(min(maxindex,n_a1(1)-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d3,n2long,n_a1,n_a2,special_n_e, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0);
        da1prime=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1);
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_d13(da1prime(:)),[N_d13*n2long,N_a1*N_a2]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V_jj(:,e_c)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d13)+1;
        allind=d_ind+N_d13*aind;
        Policy3_jj(1,:,e_c)=d_ind;
        Policy3_jj(2,:,e_c)=midpoint(allind);
        Policy3_jj(3,:,e_c)=ceil(maxindexL2/N_d13);
        L2offset      = ceil(maxindexL2/N_d13);
        linidx_lower  = d_ind                    + N_d13*n2long*aind;
        linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag_jj(1,:,e_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
        a1opt_mid=midpoint(allind);
        lin=d3opt+N_d3*(a1opt_mid-1);
        Policyd2_jj(1,:,e_c)=d2index_resh(lin);
    end
end

end
