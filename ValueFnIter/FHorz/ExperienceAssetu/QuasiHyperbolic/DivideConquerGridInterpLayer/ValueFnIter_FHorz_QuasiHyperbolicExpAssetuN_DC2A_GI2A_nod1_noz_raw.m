function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC2A_GI2A_nod1_noz_raw(n_d2, n_a1, n_a2, n_a3, n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, u_gridvals, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAssetu_DC2A_GI2A_nod1_noz_raw.
% aprime(d2,a3,u) via CreateExperienceAssetuFnMatrix.
% a1=DC'd + grid-interpolated standard endogenous state, a2=folded standard
% endogenous state(s), a3=experience asset, u=i.i.d. shock on the a3 transition.
% GPU only.
%
% Naive:  Valt_j   = max u + beta*E[Valt_{j+1}]         (exponential discounter)
%         Vtilde_j = max u + beta_0*beta*E[Valt_{j+1}]  (agent's perceived choice)
% The two discount factors generally pick different DC brackets and different GI
% midpoints, so each pass re-derives its own bracket and midpoint; the level-1
% return matrix is shared, the beta pass uses maxgap_V and the beta0*beta pass
% plain maxgap, and level-2/level-3 return matrices are *_dc/*_gi (*alt for beta).

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_u=prod(n_u);
N_a=N_a1*N_a2*N_a3;

Valt=zeros(N_a,N_j,'gpuArray'); % exponential-discounter value fn (beta)
Policy=zeros(5,N_a,N_j,'gpuArray');
Policy(5,:,:)=2;
Policyalt=zeros(5,N_a,N_j,'gpuArray'); % exponential-discounter policy
Policyalt(5,:,:)=2;

aind=gpuArray(0:1:N_a-1);

midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
midpointalt=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');

% n-Monotonicity over a1
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% GI grid
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

pi_u=shiftdim(pi_u,-2); % put u into third dimension

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    midpoint(:,1,:,level1ii,:,:)=maxindex1;

    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
            [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
        end
    end

    midpoint=max(min(midpoint,N_a1-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    Valt(:,N_j)=shiftdim(Vtempii,1);

    d_ind        =rem(maxindexL2-1,N_d2)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=midpoint(allind);
    Policy(3,:,N_j)=maxindexL2a2;
    Policy(4,:,N_j)=maxindexL2a1;

    linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
    linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    Policy(5,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    Vtilde=Valt;
    Policyalt(:,:,N_j)=Policy(:,:,N_j); % terminal: QH and exp discounter coincide
    Policyalt(5,:,N_j)=Policy(5,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1);

    Vlower=reshape(EVpre(aprimeIndex(:)),    [N_d2*N_a1*N_a2,N_a3,N_u]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1*N_a2,N_a3,N_u]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);

    EV=sum(EV.*pi_u,3); % integrate out u
    entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3]); % undiscounted; beta/beta0beta applied at use sites
    entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

    Vtilde=zeros(N_a,N_j,'gpuArray');

    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
    entireRHS_iialt=ReturnMatrix_ii+beta*entireEV;
    [~,maxindex1_V]=max(entireRHS_iialt,[],2);
    midpointalt(:,1,:,level1ii,:,:)=maxindex1_V;

    maxgap_V=squeeze(max(max(max(max( maxindex1_V(:,1,:,2:end,:,:)-maxindex1_V(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inneralt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap_V(ii)>0
            loweredgealt=min(maxindex1_V(:,1,:,ii,:,:),N_a1-maxgap_V(ii));
            a1primeindexesalt=loweredgealt+(0:1:maxgap_V(ii));
            ReturnMatrix_ii_dcalt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1_grid(a1primeindexesalt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
            d2aprimealt=(1:1:N_d2)' + N_d2*(a1primeindexesalt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_iialt=ReturnMatrix_ii_dcalt+beta*entireEV(d2aprimealt);
            [~,maxindex_inneralt]=max(entireRHS_iialt,[],2);
            midpointalt(:,1,:,curra1inneralt,:,:)=maxindex_inneralt+(loweredgealt-1);
        else
            loweredgealt=maxindex1_V(:,1,:,ii,:,:);
            midpointalt(:,1,:,curra1inneralt,:,:)=repelem(loweredgealt,1,1,1,level1iidiff(ii),1,1);
        end
    end

    midpointalt=max(min(midpointalt,N_a1-1),2);
    a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii_gialt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
    aprimealt=(1:1:N_d2)' + N_d2*(a1primeindexesfinealt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
    entireRHS_iialt=reshape(ReturnMatrix_ii_gialt+beta*entireEVinterp(aprimealt),[N_d2*n2long*N_a2,N_a]);
    [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
    Valt(:,N_j)=shiftdim(Vtempiialt,1);

    d_indalt        =rem(maxindexL2alt-1,N_d2)+1;
    maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d2),n2long)+1;
    maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d2*n2long))+1;

    allindalt=d_indalt + N_d2*(maxindexL2a2alt-1) + N_d2*N_a2*aind;
    Policyalt(1,:,N_j)=d_indalt;
    Policyalt(2,:,N_j)=midpointalt(allindalt);
    Policyalt(3,:,N_j)=maxindexL2a2alt;
    Policyalt(4,:,N_j)=maxindexL2a1alt;

    ReturnMatrix_ii_flatalt=reshape(ReturnMatrix_ii_gialt,[N_d2*n2long*N_a2,N_a]);
    linidx_loweralt=d_indalt                   + N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind;
    linidx_upperalt=d_indalt + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind;
    isInfLoweralt=(ReturnMatrix_ii_flatalt(linidx_loweralt)==-Inf);
    isInfUpperalt=(ReturnMatrix_ii_flatalt(linidx_upperalt)==-Inf);
    inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
    inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
    Policyalt(5,:,N_j)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
    %% Vtilde (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+beta0beta*entireEV;
    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoint(:,1,:,level1ii,:,:)=maxindex1;

    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*entireEV(d2aprime);
            [~,maxindex_inner]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
        end
    end

    midpoint=max(min(midpoint,N_a1-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii_gi=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
    aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
    entireRHS_ii=reshape(ReturnMatrix_ii_gi+beta0beta*entireEVinterp(aprime),[N_d2*n2long*N_a2,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    Vtilde(:,N_j)=shiftdim(Vtempii,1);

    d_ind        =rem(maxindexL2-1,N_d2)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=midpoint(allind);
    Policy(3,:,N_j)=maxindexL2a2;
    Policy(4,:,N_j)=maxindexL2a1;

    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_gi,[N_d2*n2long*N_a2,N_a]);
    linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
    linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    Policy(5,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
end


%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1);

    Vlower=reshape(Valt(aprimeIndex(:),jj+1),    [N_d2*N_a1*N_a2,N_a3,N_u]);
    Vupper=reshape(Valt(aprimeplus1Index(:),jj+1),[N_d2*N_a1*N_a2,N_a3,N_u]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);

    EV=sum(EV.*pi_u,3); % integrate out u
    entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3]); % undiscounted; beta/beta0beta applied at use sites
    entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
    entireRHS_iialt=ReturnMatrix_ii+beta*entireEV;
    [~,maxindex1_V]=max(entireRHS_iialt,[],2);
    midpointalt(:,1,:,level1ii,:,:)=maxindex1_V;

    maxgap_V=squeeze(max(max(max(max( maxindex1_V(:,1,:,2:end,:,:)-maxindex1_V(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inneralt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap_V(ii)>0
            loweredgealt=min(maxindex1_V(:,1,:,ii,:,:),N_a1-maxgap_V(ii));
            a1primeindexesalt=loweredgealt+(0:1:maxgap_V(ii));
            ReturnMatrix_ii_dcalt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1_grid(a1primeindexesalt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
            d2aprimealt=(1:1:N_d2)' + N_d2*(a1primeindexesalt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_iialt=ReturnMatrix_ii_dcalt+beta*entireEV(d2aprimealt);
            [~,maxindex_inneralt]=max(entireRHS_iialt,[],2);
            midpointalt(:,1,:,curra1inneralt,:,:)=maxindex_inneralt+(loweredgealt-1);
        else
            loweredgealt=maxindex1_V(:,1,:,ii,:,:);
            midpointalt(:,1,:,curra1inneralt,:,:)=repelem(loweredgealt,1,1,1,level1iidiff(ii),1,1);
        end
    end

    midpointalt=max(min(midpointalt,N_a1-1),2);
    a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii_gialt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
    aprimealt=(1:1:N_d2)' + N_d2*(a1primeindexesfinealt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
    entireRHS_iialt=reshape(ReturnMatrix_ii_gialt+beta*entireEVinterp(aprimealt),[N_d2*n2long*N_a2,N_a]);
    [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
    Valt(:,jj)=shiftdim(Vtempiialt,1);

    d_indalt        =rem(maxindexL2alt-1,N_d2)+1;
    maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d2),n2long)+1;
    maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d2*n2long))+1;

    allindalt=d_indalt + N_d2*(maxindexL2a2alt-1) + N_d2*N_a2*aind;
    Policyalt(1,:,jj)=d_indalt;
    Policyalt(2,:,jj)=midpointalt(allindalt);
    Policyalt(3,:,jj)=maxindexL2a2alt;
    Policyalt(4,:,jj)=maxindexL2a1alt;

    ReturnMatrix_ii_flatalt=reshape(ReturnMatrix_ii_gialt,[N_d2*n2long*N_a2,N_a]);
    linidx_loweralt=d_indalt                   + N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind;
    linidx_upperalt=d_indalt + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind;
    isInfLoweralt=(ReturnMatrix_ii_flatalt(linidx_loweralt)==-Inf);
    isInfUpperalt=(ReturnMatrix_ii_flatalt(linidx_upperalt)==-Inf);
    inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
    inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
    Policyalt(5,:,jj)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
    %% Vtilde (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+beta0beta*entireEV;
    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoint(:,1,:,level1ii,:,:)=maxindex1;

    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*entireEV(d2aprime);
            [~,maxindex_inner]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:);
            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
        end
    end

    midpoint=max(min(midpoint,N_a1-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii_gi=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, 0, n_d2, n_a2, n_a3, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
    aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
    entireRHS_ii=reshape(ReturnMatrix_ii_gi+beta0beta*entireEVinterp(aprime),[N_d2*n2long*N_a2,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    Vtilde(:,jj)=shiftdim(Vtempii,1);

    d_ind        =rem(maxindexL2-1,N_d2)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
    Policy(1,:,jj)=d_ind;
    Policy(2,:,jj)=midpoint(allind);
    Policy(3,:,jj)=maxindexL2a2;
    Policy(4,:,jj)=maxindexL2a1;

    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_gi,[N_d2*n2long*N_a2,N_a]);
    linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
    linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    Policy(5,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
end


%% Post-process
adjust=(Policy(4,:,:)<1+n2short+1);
Policy(2,:,:)=Policy(2,:,:)-adjust;
Policy(4,:,:)=Policy(4,:,:)-(n2short+1)*(~adjust);

adjustalt=(Policyalt(4,:,:)<1+n2short+1);
Policyalt(2,:,:)=Policyalt(2,:,:)-adjustalt;
Policyalt(4,:,:)=Policyalt(4,:,:)-(n2short+1)*(~adjustalt);

end
