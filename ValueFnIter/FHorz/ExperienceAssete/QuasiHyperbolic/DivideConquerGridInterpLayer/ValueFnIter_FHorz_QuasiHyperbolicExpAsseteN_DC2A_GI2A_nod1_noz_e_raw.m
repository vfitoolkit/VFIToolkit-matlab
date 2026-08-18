function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_DC2A_GI2A_nod1_noz_e_raw(n_d2, n_a1, n_a2, n_a3, n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAssete_DC2A_GI2A_nod1_noz_e_raw.
% experienceassete: aprime(d2,a3,e) via CreateExperienceAsseteFnMatrix.
% a1=DC'd + grid-interpolated standard endogenous state, a2=folded standard
% endogenous state(s), a3=experience asset. GPU only.
%
% Naive:  Valt_j   = max u + beta*E[Valt_{j+1}]         (exponential discounter)
%         Vtilde_j = max u + beta_0*beta*E[Valt_{j+1}]  (agent's perceived choice)
% The two discount factors generally pick different DC brackets and different GI
% midpoints, so each pass re-derives its own bracket and midpoint; the level-1
% return matrix is shared, the beta pass uses maxgap_V and the beta0*beta pass
% plain maxgap, and level-2/level-3 return matrices are *_dc/*_gi (*alt for beta).
% lowmemory=0 and 1.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_e=prod(n_e);

Valt=zeros(N_a,N_e,N_j,'gpuArray'); % exponential-discounter value fn (beta)
Policy=zeros(4,N_a,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_e,N_j,'gpuArray');
Policyalt=zeros(4,N_a,N_e,N_j,'gpuArray'); % exponential-discounter policy
PolicyL2flagalt=2*ones(1,N_a,N_e,N_j,'gpuArray');

aind=gpuArray(0:1:N_a-1);
if vfoptions.lowmemory==0
    eindB=shiftdim(gpuArray(0:1:N_e-1),-1);
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_e,'gpuArray');
    midpointalt=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_e,'gpuArray');
else
    special_n_e=ones(1,length(n_e));
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
    midpointalt=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Valt(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d2)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

        allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*eindB;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,N_j)=maxindexL2a1;

        linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
        linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii_e,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
            Valt(:,e_c,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
            Policy(1,:,e_c,N_j)=d_ind;
            Policy(2,:,e_c,N_j)=midpoint(allind);
            Policy(3,:,e_c,N_j)=maxindexL2a2;
            Policy(4,:,e_c,N_j)=maxindexL2a1;

            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
            isInfLower=(ReturnMatrix_ii_e(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_e(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end

    Vtilde=Valt;
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j); % terminal: QH and exp discounter coincide
    PolicyL2flagalt(1,:,:,N_j)=PolicyL2flag(1,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=sum(pi_e_J(:,N_j+1)'.*reshape(vfoptions.V_Jplus1,[N_a,N_e]),2);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a3, n_e, d2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex and a3primeProbs are [N_d2,N_a3,N_e]   (N_e here is the current e)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1);

    Vlower=reshape(EVpre(aprimeIndex(:)),    [N_d2*N_a1*N_a2,N_a3,N_e]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1*N_a2,N_a3,N_e]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2*a1prime*a2prime,a3,e_cur)

    entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_e]); % undiscounted; beta/beta0beta applied at use sites
    entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    Vtilde=zeros(N_a,N_e,N_j,'gpuArray');

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHS_iialt=ReturnMatrix_ii+beta*entireEV;
        [~,maxindex1_V]=max(entireRHS_iialt,[],2);
        midpointalt(:,1,:,level1ii,:,:,:)=maxindex1_V;

        maxgap_V=squeeze(max(max(max(max(max( maxindex1_V(:,1,:,2:end,:,:,:)-maxindex1_V(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inneralt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap_V(ii)>0
                loweredgealt=min(maxindex1_V(:,1,:,ii,:,:,:),N_a1-maxgap_V(ii));
                a1primeindexesalt=loweredgealt+(0:1:maxgap_V(ii));
                ReturnMatrix_ii_dcalt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexesalt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimeealt=(1:1:N_d2)' + N_d2*(a1primeindexesalt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_iialt=ReturnMatrix_ii_dcalt+beta*entireEV(d2aprimeealt);
                [~,maxindex_inneralt]=max(entireRHS_iialt,[],2);
                midpointalt(:,1,:,curra1inneralt,:,:,:)=maxindex_inneralt+(loweredgealt-1);
            else
                loweredgealt=maxindex1_V(:,1,:,ii,:,:,:);
                midpointalt(:,1,:,curra1inneralt,:,:,:)=repelem(loweredgealt,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        midpointalt=max(min(midpointalt,N_a1-1),2);
        a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_gialt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        aprimeealt=(1:1:N_d2)' + N_d2*(a1primeindexesfinealt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
        entireRHS_iialt=reshape(ReturnMatrix_ii_gialt+beta*entireEVinterp(aprimeealt),[N_d2*n2long*N_a2,N_a,N_e]);
        [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
        Valt(:,:,N_j)=shiftdim(Vtempiialt,1);

        d_indalt        =rem(maxindexL2alt-1,N_d2)+1;
        maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d2),n2long)+1;
        maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d2*n2long))+1;

        allindalt=d_indalt + N_d2*(maxindexL2a2alt-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*eindB;
        Policyalt(1,:,:,N_j)=d_indalt;
        Policyalt(2,:,:,N_j)=midpointalt(allindalt);
        Policyalt(3,:,:,N_j)=maxindexL2a2alt;
        Policyalt(4,:,:,N_j)=maxindexL2a1alt;

        ReturnMatrix_ii_flatalt=reshape(ReturnMatrix_ii_gialt,[N_d2*n2long*N_a2,N_a,N_e]);
        linidx_loweralt=d_indalt                + N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
        linidx_upperalt=d_indalt + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
        isInfLoweralt=(ReturnMatrix_ii_flatalt(linidx_loweralt)==-Inf);
        isInfUpperalt=(ReturnMatrix_ii_flatalt(linidx_upperalt)==-Inf);
        inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
        inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
        PolicyL2flagalt(1,:,:,N_j)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+beta0beta*entireEV;
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*entireEV(d2aprimee);
                [~,maxindex_inner]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_gi=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        aprimee=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii_gi+beta0beta*entireEVinterp(aprimee),[N_d2*n2long*N_a2,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d2)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

        allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*eindB;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,N_j)=maxindexL2a1;

        ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_gi,[N_d2*n2long*N_a2,N_a,N_e]);
        linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
        linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
        isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            entireEV_e=entireEV(:,:,:,:,:,:,e_c);
            entireEVinterp_e=entireEVinterp(:,:,:,:,:,:,e_c);

            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS_ii_ealt=ReturnMatrix_ii_e+beta*entireEV_e;
            [~,maxindex1_V]=max(entireRHS_ii_ealt,[],2);
            midpointalt(:,1,:,level1ii,:,:)=maxindex1_V;

            maxgap_V=squeeze(max(max(max(max( maxindex1_V(:,1,:,2:end,:,:)-maxindex1_V(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inneralt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_V(ii)>0
                    loweredgealt=min(maxindex1_V(:,1,:,ii,:,:),N_a1-maxgap_V(ii));
                    a1primeindexesalt=loweredgealt+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_e_dcalt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexesalt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprimealt=(1:1:N_d2)' + N_d2*(a1primeindexesalt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_ealt=ReturnMatrix_ii_e_dcalt+beta*entireEV_e(d2aprimealt);
                    [~,maxindex_inneralt]=max(entireRHS_ii_ealt,[],2);
                    midpointalt(:,1,:,curra1inneralt,:,:)=maxindex_inneralt+(loweredgealt-1);
                else
                    loweredgealt=maxindex1_V(:,1,:,ii,:,:);
                    midpointalt(:,1,:,curra1inneralt,:,:)=repelem(loweredgealt,1,1,1,level1iidiff(ii),1,1);
                end
            end

            midpointalt=max(min(midpointalt,N_a1-1),2);
            a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_e_gialt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
            aprimealt=(1:1:N_d2)' + N_d2*(a1primeindexesfinealt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_ealt=reshape(ReturnMatrix_ii_e_gialt+beta*entireEVinterp_e(aprimealt),[N_d2*n2long*N_a2,N_a]);
            [Vtempiialt,maxindexL2alt]=max(entireRHS_ii_ealt,[],1);
            Valt(:,e_c,N_j)=shiftdim(Vtempiialt,1);

            d_indalt        =rem(maxindexL2alt-1,N_d2)+1;
            maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d2),n2long)+1;
            maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d2*n2long))+1;

            allindalt=d_indalt + N_d2*(maxindexL2a2alt-1) + N_d2*N_a2*aind;
            Policyalt(1,:,e_c,N_j)=d_indalt;
            Policyalt(2,:,e_c,N_j)=midpointalt(allindalt);
            Policyalt(3,:,e_c,N_j)=maxindexL2a2alt;
            Policyalt(4,:,e_c,N_j)=maxindexL2a1alt;

            ReturnMatrix_ii_flatalt=reshape(ReturnMatrix_ii_e_gialt,[N_d2*n2long*N_a2,N_a]);
            linidx_loweralt=d_indalt                + N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind;
            linidx_upperalt=d_indalt + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind;
            isInfLoweralt=(ReturnMatrix_ii_flatalt(linidx_loweralt)==-Inf);
            isInfUpperalt=(ReturnMatrix_ii_flatalt(linidx_upperalt)==-Inf);
            inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
            inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
            PolicyL2flagalt(1,:,e_c,N_j)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
            %% Vtilde (beta0*beta)
            entireRHS_ii_e=ReturnMatrix_ii_e+beta0beta*entireEV_e;
            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_e=ReturnMatrix_ii_e_dc+beta0beta*entireEV_e(d2aprime);
                    [~,maxindex_inner]=max(entireRHS_ii_e,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_e_gi=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
            aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_e=reshape(ReturnMatrix_ii_e_gi+beta0beta*entireEVinterp_e(aprime),[N_d2*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            Vtilde(:,e_c,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
            Policy(1,:,e_c,N_j)=d_ind;
            Policy(2,:,e_c,N_j)=midpoint(allind);
            Policy(3,:,e_c,N_j)=maxindexL2a2;
            Policy(4,:,e_c,N_j)=maxindexL2a1;

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_e_gi,[N_d2*n2long*N_a2,N_a]);
            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end
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
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EVpre=sum(pi_e_J(:,jj+1)'.*reshape(Valt(:,:,jj+1),[N_a,N_e]),2);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a3, n_e, d2_gridvals, a3_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex and a3primeProbs are [N_d2,N_a3,N_e]   (N_e here is the current e)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1);

    Vlower=reshape(EVpre(aprimeIndex(:)),    [N_d2*N_a1*N_a2,N_a3,N_e]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1*N_a2,N_a3,N_e]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2*a1prime*a2prime,a3,e_cur)

    entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_e]); % undiscounted; beta/beta0beta applied at use sites
    entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHS_iialt=ReturnMatrix_ii+beta*entireEV;
        [~,maxindex1_V]=max(entireRHS_iialt,[],2);
        midpointalt(:,1,:,level1ii,:,:,:)=maxindex1_V;

        maxgap_V=squeeze(max(max(max(max(max( maxindex1_V(:,1,:,2:end,:,:,:)-maxindex1_V(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inneralt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap_V(ii)>0
                loweredgealt=min(maxindex1_V(:,1,:,ii,:,:,:),N_a1-maxgap_V(ii));
                a1primeindexesalt=loweredgealt+(0:1:maxgap_V(ii));
                ReturnMatrix_ii_dcalt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexesalt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimeealt=(1:1:N_d2)' + N_d2*(a1primeindexesalt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_iialt=ReturnMatrix_ii_dcalt+beta*entireEV(d2aprimeealt);
                [~,maxindex_inneralt]=max(entireRHS_iialt,[],2);
                midpointalt(:,1,:,curra1inneralt,:,:,:)=maxindex_inneralt+(loweredgealt-1);
            else
                loweredgealt=maxindex1_V(:,1,:,ii,:,:,:);
                midpointalt(:,1,:,curra1inneralt,:,:,:)=repelem(loweredgealt,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        midpointalt=max(min(midpointalt,N_a1-1),2);
        a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_gialt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimeealt=(1:1:N_d2)' + N_d2*(a1primeindexesfinealt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
        entireRHS_iialt=reshape(ReturnMatrix_ii_gialt+beta*entireEVinterp(aprimeealt),[N_d2*n2long*N_a2,N_a,N_e]);
        [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
        Valt(:,:,jj)=shiftdim(Vtempiialt,1);

        d_indalt        =rem(maxindexL2alt-1,N_d2)+1;
        maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d2),n2long)+1;
        maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d2*n2long))+1;

        allindalt=d_indalt + N_d2*(maxindexL2a2alt-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*eindB;
        Policyalt(1,:,:,jj)=d_indalt;
        Policyalt(2,:,:,jj)=midpointalt(allindalt);
        Policyalt(3,:,:,jj)=maxindexL2a2alt;
        Policyalt(4,:,:,jj)=maxindexL2a1alt;

        ReturnMatrix_ii_flatalt=reshape(ReturnMatrix_ii_gialt,[N_d2*n2long*N_a2,N_a,N_e]);
        linidx_loweralt=d_indalt                + N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
        linidx_upperalt=d_indalt + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
        isInfLoweralt=(ReturnMatrix_ii_flatalt(linidx_loweralt)==-Inf);
        isInfUpperalt=(ReturnMatrix_ii_flatalt(linidx_upperalt)==-Inf);
        inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
        inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
        PolicyL2flagalt(1,:,:,jj)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+beta0beta*entireEV;
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*entireEV(d2aprimee);
                [~,maxindex_inner]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_gi=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimee=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii_gi+beta0beta*entireEVinterp(aprimee),[N_d2*n2long*N_a2,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,:,jj)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d2)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

        allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*eindB;
        Policy(1,:,:,jj)=d_ind;
        Policy(2,:,:,jj)=midpoint(allind);
        Policy(3,:,:,jj)=maxindexL2a2;
        Policy(4,:,:,jj)=maxindexL2a1;

        ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_gi,[N_d2*n2long*N_a2,N_a,N_e]);
        linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
        linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
        isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            entireEV_e=entireEV(:,:,:,:,:,:,e_c);
            entireEVinterp_e=entireEVinterp(:,:,:,:,:,:,e_c);

            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS_ii_ealt=ReturnMatrix_ii_e+beta*entireEV_e;
            [~,maxindex1_V]=max(entireRHS_ii_ealt,[],2);
            midpointalt(:,1,:,level1ii,:,:)=maxindex1_V;

            maxgap_V=squeeze(max(max(max(max( maxindex1_V(:,1,:,2:end,:,:)-maxindex1_V(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inneralt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_V(ii)>0
                    loweredgealt=min(maxindex1_V(:,1,:,ii,:,:),N_a1-maxgap_V(ii));
                    a1primeindexesalt=loweredgealt+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_e_dcalt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexesalt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprimealt=(1:1:N_d2)' + N_d2*(a1primeindexesalt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_ealt=ReturnMatrix_ii_e_dcalt+beta*entireEV_e(d2aprimealt);
                    [~,maxindex_inneralt]=max(entireRHS_ii_ealt,[],2);
                    midpointalt(:,1,:,curra1inneralt,:,:)=maxindex_inneralt+(loweredgealt-1);
                else
                    loweredgealt=maxindex1_V(:,1,:,ii,:,:);
                    midpointalt(:,1,:,curra1inneralt,:,:)=repelem(loweredgealt,1,1,1,level1iidiff(ii),1,1);
                end
            end

            midpointalt=max(min(midpointalt,N_a1-1),2);
            a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_e_gialt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
            aprimealt=(1:1:N_d2)' + N_d2*(a1primeindexesfinealt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_ealt=reshape(ReturnMatrix_ii_e_gialt+beta*entireEVinterp_e(aprimealt),[N_d2*n2long*N_a2,N_a]);
            [Vtempiialt,maxindexL2alt]=max(entireRHS_ii_ealt,[],1);
            Valt(:,e_c,jj)=shiftdim(Vtempiialt,1);

            d_indalt        =rem(maxindexL2alt-1,N_d2)+1;
            maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d2),n2long)+1;
            maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d2*n2long))+1;

            allindalt=d_indalt + N_d2*(maxindexL2a2alt-1) + N_d2*N_a2*aind;
            Policyalt(1,:,e_c,jj)=d_indalt;
            Policyalt(2,:,e_c,jj)=midpointalt(allindalt);
            Policyalt(3,:,e_c,jj)=maxindexL2a2alt;
            Policyalt(4,:,e_c,jj)=maxindexL2a1alt;

            ReturnMatrix_ii_flatalt=reshape(ReturnMatrix_ii_e_gialt,[N_d2*n2long*N_a2,N_a]);
            linidx_loweralt=d_indalt                + N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind;
            linidx_upperalt=d_indalt + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2alt-1) + N_d2*n2long*N_a2*aind;
            isInfLoweralt=(ReturnMatrix_ii_flatalt(linidx_loweralt)==-Inf);
            isInfUpperalt=(ReturnMatrix_ii_flatalt(linidx_upperalt)==-Inf);
            inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
            inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
            PolicyL2flagalt(1,:,e_c,jj)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
            %% Vtilde (beta0*beta)
            entireRHS_ii_e=ReturnMatrix_ii_e+beta0beta*entireEV_e;
            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_e=ReturnMatrix_ii_e_dc+beta0beta*entireEV_e(d2aprime);
                    [~,maxindex_inner]=max(entireRHS_ii_e,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_e_gi=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
            aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_e=reshape(ReturnMatrix_ii_e_gi+beta0beta*entireEVinterp_e(aprime),[N_d2*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            Vtilde(:,e_c,jj)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
            Policy(1,:,e_c,jj)=d_ind;
            Policy(2,:,e_c,jj)=midpoint(allind);
            Policy(3,:,e_c,jj)=maxindexL2a2;
            Policy(4,:,e_c,jj)=maxindexL2a1;

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_e_gi,[N_d2*n2long*N_a2,N_a]);
            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,e_c,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end
end


%% Post-process
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

adjustalt=(Policyalt(4,:,:,:)<1+n2short+1);
Policyalt(2,:,:,:)=Policyalt(2,:,:,:)-adjustalt;
Policyalt(4,:,:,:)=adjustalt.*Policyalt(4,:,:,:)+(1-adjustalt).*(Policyalt(4,:,:,:)-n2short-1);

Policyalt=[Policyalt;PolicyL2flagalt];

end
