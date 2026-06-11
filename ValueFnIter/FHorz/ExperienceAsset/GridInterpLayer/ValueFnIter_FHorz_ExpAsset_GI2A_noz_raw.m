function [V,Policy]=ValueFnIter_FHorz_ExpAsset_GI2A_noz_raw(n_d1, n_d2, n_a1, n_a2, n_a3, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% with-d1 _noz analog of ValueFnIter_FHorz_ExpAsset_GI2A_raw.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(4,N_a,N_j,'gpuArray'); % 1=d (joint), 2=a1prime midpoint, 3=a2prime, 4=a1prime L2 fine
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray');

%% GI setup
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d, 1]
aind=gpuArray(0:1:N_a-1);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    [~,maxindex]=max(ReturnMatrix,[],2);
    midpoint=max(min(maxindex,N_a1-1),2);

    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);

    d_ind        =rem(maxindexL2-1,N_d)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

    allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=midpoint(allind);
    Policy(3,:,N_j)=maxindexL2a2;
    Policy(4,:,N_j)=maxindexL2a1;

    linidx_lower=d_ind                 + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    PolicyL2flag(1,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a3, d2_gridvals, a3_grid, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1);

    Vlower=reshape(EVpre(aprimeIndex(:)),    [N_d2*N_a1*N_a2,N_a3]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1*N_a2,N_a3]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    entireRHS=ReturnMatrix+repelem(DiscountedEV,N_d1,1,1,1,1,1);
    [~,maxindex]=max(entireRHS,[],2);
    midpoint=max(min(maxindex,N_a1-1),2);

    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
    aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);

    d_ind        =rem(maxindexL2-1,N_d)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

    allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=midpoint(allind);
    Policy(3,:,N_j)=maxindexL2a2;
    Policy(4,:,N_j)=maxindexL2a1;

    linidx_lower=d_ind                 + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    PolicyL2flag(1,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
end

%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a3, d2_gridvals, a3_grid, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1);

    Vlower=reshape(V(aprimeIndex(:),jj+1),    [N_d2*N_a1*N_a2,N_a3]);
    Vupper=reshape(V(aprimeplus1Index(:),jj+1),[N_d2*N_a1*N_a2,N_a3]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    entireRHS=ReturnMatrix+repelem(DiscountedEV,N_d1,1,1,1,1,1);
    [~,maxindex]=max(entireRHS,[],2);
    midpoint=max(min(maxindex,N_a1-1),2);

    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
    aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,jj)=shiftdim(Vtempii,1);

    d_ind        =rem(maxindexL2-1,N_d)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

    allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
    Policy(1,:,jj)=d_ind;
    Policy(2,:,jj)=midpoint(allind);
    Policy(3,:,jj)=maxindexL2a2;
    Policy(4,:,jj)=maxindexL2a1;

    linidx_lower=d_ind                 + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    PolicyL2flag(1,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
end


%% Post-process
adjust=(Policy(4,:,:)<1+n2short+1);
Policy(2,:,:)=Policy(2,:,:)-adjust;
Policy(4,:,:)=adjust.*Policy(4,:,:)+(1-adjust).*(Policy(4,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

end
