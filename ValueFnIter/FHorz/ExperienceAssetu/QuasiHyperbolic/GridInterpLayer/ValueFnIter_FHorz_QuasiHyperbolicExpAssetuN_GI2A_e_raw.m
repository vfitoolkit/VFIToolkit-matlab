function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, n_e, n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, u_gridvals, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAssetu_GI2A_e_raw.
% a1=standard endogenous state carrying the grid interpolation layer, a2=folded
% standard endogenous state(s), a3=experience asset. GPU only.
% Policy is 4-channel: 1=d, 2=a1prime midpoint, 3=a2prime, 4=a1prime L2;
% the L2 flag is channel 5.
%
% Naive:  Valt_j   = max u + beta*E[Valt_{j+1}]         (exponential discounter)
%         Vtilde_j = max u + beta_0*beta*E[Valt_{j+1}]  (agent's perceived choice)
% The two discount factors generally pick different GI midpoints, so each pass
% re-derives its own midpoint and layer-2 return matrix, and each keeps its own
% a1prime midpoint / a1prime L2 / a2prime policy channels.
% lowmemory=0 full; lowmemory=1 loop z (e vectorized); lowmemory=2 nested z+e.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_u=prod(n_u);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);
N_e=prod(n_e);

Valt=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_z,N_e,N_j,'gpuArray');
Policy(5,:,:,:,:)=2;
Policyalt=zeros(5,N_a,N_z,N_e,N_j,'gpuArray'); % exponential discounter optimal choice
Policyalt(5,:,:,:,:)=2;

d2ind_vec=repelem((1:1:N_d2)',N_d1,1);

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
if vfoptions.lowmemory==0
    zindB=shiftdim(gpuArray(0:1:N_z-1),-1);
    eindB=shiftdim(gpuArray(0:1:N_e-1),-2);
elseif vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
    eindB=shiftdim(gpuArray(0:1:N_e-1),-2);
else
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
end

pi_u=shiftdim(pi_u,-2); % put u into third dimension

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,N_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Valt(:,:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
        Policy(1,:,:,:,N_j)=d_ind;
        Policy(2,:,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,:,N_j)=maxindexL2a1;

        linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        Policy(5,:,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_z,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            Valt(:,z_c,:,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

            allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*eindB;
            Policy(1,:,z_c,:,N_j)=d_ind;
            Policy(2,:,z_c,:,N_j)=midpoint(allind);
            Policy(3,:,z_c,:,N_j)=maxindexL2a2;
            Policy(4,:,z_c,:,N_j)=maxindexL2a1;

            linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            Policy(5,:,z_c,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_ze,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                Valt(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

                allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
                Policy(1,:,z_c,e_c,N_j)=d_ind;
                Policy(2,:,z_c,e_c,N_j)=midpoint(allind);
                Policy(3,:,z_c,e_c,N_j)=maxindexL2a2;
                Policy(4,:,z_c,e_c,N_j)=maxindexL2a1;

                linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                Policy(5,:,z_c,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end
    end

    Vtilde=Valt;
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j); % terminal: QH and exp discounter coincide
    Policyalt(5,:,:,:,N_j)=Policy(5,:,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3));

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    EV=squeeze(sum((EV.*pi_u),3)); % integrate out u
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,3));

    entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]); % undiscounted; beta/beta0beta applied at use sites
    entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    Vtilde=zeros(N_a,N_z,N_e,N_j,'gpuArray');

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHSalt=ReturnMatrix+beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
        [~,maxindexalt]=max(entireRHSalt,[],2);
        midpointalt=max(min(maxindexalt,N_a1-1),2);

        a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesalt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        aprimezalt=d2ind_vec + N_d2*(a1primeindexesalt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        entireRHS_iialt=reshape(ReturnMatrix_iialt+beta*entireEVinterp(aprimezalt),[N_d*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
        Valt(:,:,:,N_j)=shiftdim(Vtempii,1);

        d_indalt        =rem(maxindexL2alt-1,N_d)+1;
        maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d),n2long)+1;
        maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d*n2long))+1;

        allindalt=d_indalt + N_d*(maxindexL2a2alt-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
        Policyalt(1,:,:,:,N_j)=d_indalt;
        Policyalt(2,:,:,:,N_j)=midpointalt(allindalt);
        Policyalt(3,:,:,:,N_j)=maxindexL2a2alt;
        Policyalt(4,:,:,:,N_j)=maxindexL2a1alt;

        linidx_loweralt=d_indalt                + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        linidx_upperalt=d_indalt + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        isInfLoweralt   =(ReturnMatrix_iialt(linidx_loweralt)==-Inf);
        isInfUpperalt   =(ReturnMatrix_iialt(linidx_upperalt)==-Inf);
        inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
        inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
        Policyalt(5,:,:,:,N_j)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
        %% Vtilde (beta0*beta)
        entireRHS=ReturnMatrix+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,N_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii+beta0beta*entireEVinterp(aprimez),[N_d*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
        Policy(1,:,:,:,N_j)=d_ind;
        Policy(2,:,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,:,N_j)=maxindexL2a1;

        linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        Policy(5,:,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            entireEV_z=entireEV(:,:,:,:,:,:,z_c);
            entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS_zalt=ReturnMatrix_z+beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
            [~,maxindexalt]=max(entireRHS_zalt,[],2);
            midpointalt=max(min(maxindexalt,N_a1-1),2);

            a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_zalt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesalt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimealt=d2ind_vec + N_d2*(a1primeindexesalt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_zalt=reshape(ReturnMatrix_ii_zalt+beta*entireEVinterp_z(aprimealt),[N_d*n2long*N_a2,N_a,1,N_e]);
            [Vtempii,maxindexL2alt]=max(entireRHS_ii_zalt,[],1);
            Valt(:,z_c,:,N_j)=shiftdim(Vtempii,1);

            d_indalt        =rem(maxindexL2alt-1,N_d)+1;
            maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d),n2long)+1;
            maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d*n2long))+1;

            allindalt=d_indalt + N_d*(maxindexL2a2alt-1) + N_d*N_a2*aind + N_d*N_a2*N_a*eindB;
            Policyalt(1,:,z_c,:,N_j)=d_indalt;
            Policyalt(2,:,z_c,:,N_j)=midpointalt(allindalt);
            Policyalt(3,:,z_c,:,N_j)=maxindexL2a2alt;
            Policyalt(4,:,z_c,:,N_j)=maxindexL2a1alt;

            linidx_loweralt=d_indalt                + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            linidx_upperalt=d_indalt + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            isInfLoweralt   =(ReturnMatrix_ii_zalt(linidx_loweralt)==-Inf);
            isInfUpperalt   =(ReturnMatrix_ii_zalt(linidx_upperalt)==-Inf);
            inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
            inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
            Policyalt(5,:,z_c,:,N_j)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
            %% Vtilde (beta0*beta)
            entireRHS_z=ReturnMatrix_z+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS_z,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_z=reshape(ReturnMatrix_ii_z+beta0beta*entireEVinterp_z(aprime),[N_d*n2long*N_a2,N_a,1,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            Vtilde(:,z_c,:,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

            allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*eindB;
            Policy(1,:,z_c,:,N_j)=d_ind;
            Policy(2,:,z_c,:,N_j)=midpoint(allind);
            Policy(3,:,z_c,:,N_j)=maxindexL2a2;
            Policy(4,:,z_c,:,N_j)=maxindexL2a1;

            linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            Policy(5,:,z_c,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            entireEV_z=entireEV(:,:,:,:,:,:,z_c);
            entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
                entireRHS_zealt=ReturnMatrix_ze+beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
                [~,maxindexalt]=max(entireRHS_zealt,[],2);
                midpointalt=max(min(maxindexalt,N_a1-1),2);

                a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_zealt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1prime_grid(a1primeindexesalt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                aprimealt=d2ind_vec + N_d2*(a1primeindexesalt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_zealt=reshape(ReturnMatrix_ii_zealt+beta*entireEVinterp_z(aprimealt),[N_d*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2alt]=max(entireRHS_ii_zealt,[],1);
                Valt(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);

                d_indalt        =rem(maxindexL2alt-1,N_d)+1;
                maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d),n2long)+1;
                maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d*n2long))+1;

                allindalt=d_indalt + N_d*(maxindexL2a2alt-1) + N_d*N_a2*aind;
                Policyalt(1,:,z_c,e_c,N_j)=d_indalt;
                Policyalt(2,:,z_c,e_c,N_j)=midpointalt(allindalt);
                Policyalt(3,:,z_c,e_c,N_j)=maxindexL2a2alt;
                Policyalt(4,:,z_c,e_c,N_j)=maxindexL2a1alt;

                linidx_loweralt=d_indalt                + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind;
                linidx_upperalt=d_indalt + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind;
                isInfLoweralt   =(ReturnMatrix_ii_zealt(linidx_loweralt)==-Inf);
                isInfUpperalt   =(ReturnMatrix_ii_zealt(linidx_upperalt)==-Inf);
                inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
                inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
                Policyalt(5,:,z_c,e_c,N_j)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
                %% Vtilde (beta0*beta)
                entireRHS_ze=ReturnMatrix_ze+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS_ze,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+beta0beta*entireEVinterp_z(aprime),[N_d*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                Vtilde(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

                allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
                Policy(1,:,z_c,e_c,N_j)=d_ind;
                Policy(2,:,z_c,e_c,N_j)=midpoint(allind);
                Policy(3,:,z_c,e_c,N_j)=maxindexL2a2;
                Policy(4,:,z_c,e_c,N_j)=maxindexL2a1;

                linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                Policy(5,:,z_c,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
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
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3));

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    EV=squeeze(sum((EV.*pi_u),3)); % integrate out u
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,3));

    entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]); % undiscounted; beta/beta0beta applied at use sites
    entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHSalt=ReturnMatrix+beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
        [~,maxindexalt]=max(entireRHSalt,[],2);
        midpointalt=max(min(maxindexalt,N_a1-1),2);

        a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesalt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimezalt=d2ind_vec + N_d2*(a1primeindexesalt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        entireRHS_iialt=reshape(ReturnMatrix_iialt+beta*entireEVinterp(aprimezalt),[N_d*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
        Valt(:,:,:,jj)=shiftdim(Vtempii,1);

        d_indalt        =rem(maxindexL2alt-1,N_d)+1;
        maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d),n2long)+1;
        maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d*n2long))+1;

        allindalt=d_indalt + N_d*(maxindexL2a2alt-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
        Policyalt(1,:,:,:,jj)=d_indalt;
        Policyalt(2,:,:,:,jj)=midpointalt(allindalt);
        Policyalt(3,:,:,:,jj)=maxindexL2a2alt;
        Policyalt(4,:,:,:,jj)=maxindexL2a1alt;

        linidx_loweralt=d_indalt                + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        linidx_upperalt=d_indalt + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        isInfLoweralt   =(ReturnMatrix_iialt(linidx_loweralt)==-Inf);
        isInfUpperalt   =(ReturnMatrix_iialt(linidx_upperalt)==-Inf);
        inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
        inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
        Policyalt(5,:,:,:,jj)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
        %% Vtilde (beta0*beta)
        entireRHS=ReturnMatrix+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,N_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii+beta0beta*entireEVinterp(aprimez),[N_d*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,:,:,jj)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
        Policy(1,:,:,:,jj)=d_ind;
        Policy(2,:,:,:,jj)=midpoint(allind);
        Policy(3,:,:,:,jj)=maxindexL2a2;
        Policy(4,:,:,:,jj)=maxindexL2a1;

        linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        Policy(5,:,:,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            entireEV_z=entireEV(:,:,:,:,:,:,z_c);
            entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS_zalt=ReturnMatrix_z+beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
            [~,maxindexalt]=max(entireRHS_zalt,[],2);
            midpointalt=max(min(maxindexalt,N_a1-1),2);

            a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_zalt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesalt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimealt=d2ind_vec + N_d2*(a1primeindexesalt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_zalt=reshape(ReturnMatrix_ii_zalt+beta*entireEVinterp_z(aprimealt),[N_d*n2long*N_a2,N_a,1,N_e]);
            [Vtempii,maxindexL2alt]=max(entireRHS_ii_zalt,[],1);
            Valt(:,z_c,:,jj)=shiftdim(Vtempii,1);

            d_indalt        =rem(maxindexL2alt-1,N_d)+1;
            maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d),n2long)+1;
            maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d*n2long))+1;

            allindalt=d_indalt + N_d*(maxindexL2a2alt-1) + N_d*N_a2*aind + N_d*N_a2*N_a*eindB;
            Policyalt(1,:,z_c,:,jj)=d_indalt;
            Policyalt(2,:,z_c,:,jj)=midpointalt(allindalt);
            Policyalt(3,:,z_c,:,jj)=maxindexL2a2alt;
            Policyalt(4,:,z_c,:,jj)=maxindexL2a1alt;

            linidx_loweralt=d_indalt                + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            linidx_upperalt=d_indalt + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            isInfLoweralt   =(ReturnMatrix_ii_zalt(linidx_loweralt)==-Inf);
            isInfUpperalt   =(ReturnMatrix_ii_zalt(linidx_upperalt)==-Inf);
            inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
            inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
            Policyalt(5,:,z_c,:,jj)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
            %% Vtilde (beta0*beta)
            entireRHS_z=ReturnMatrix_z+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS_z,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_z=reshape(ReturnMatrix_ii_z+beta0beta*entireEVinterp_z(aprime),[N_d*n2long*N_a2,N_a,1,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            Vtilde(:,z_c,:,jj)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

            allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*eindB;
            Policy(1,:,z_c,:,jj)=d_ind;
            Policy(2,:,z_c,:,jj)=midpoint(allind);
            Policy(3,:,z_c,:,jj)=maxindexL2a2;
            Policy(4,:,z_c,:,jj)=maxindexL2a1;

            linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            Policy(5,:,z_c,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            entireEV_z=entireEV(:,:,:,:,:,:,z_c);
            entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
                entireRHS_zealt=ReturnMatrix_ze+beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
                [~,maxindexalt]=max(entireRHS_zealt,[],2);
                midpointalt=max(min(maxindexalt,N_a1-1),2);

                a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_zealt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1prime_grid(a1primeindexesalt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                aprimealt=d2ind_vec + N_d2*(a1primeindexesalt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_zealt=reshape(ReturnMatrix_ii_zealt+beta*entireEVinterp_z(aprimealt),[N_d*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2alt]=max(entireRHS_ii_zealt,[],1);
                Valt(:,z_c,e_c,jj)=shiftdim(Vtempii,1);

                d_indalt        =rem(maxindexL2alt-1,N_d)+1;
                maxindexL2a1alt =rem(floor((maxindexL2alt-1)/N_d),n2long)+1;
                maxindexL2a2alt =floor((maxindexL2alt-1)/(N_d*n2long))+1;

                allindalt=d_indalt + N_d*(maxindexL2a2alt-1) + N_d*N_a2*aind;
                Policyalt(1,:,z_c,e_c,jj)=d_indalt;
                Policyalt(2,:,z_c,e_c,jj)=midpointalt(allindalt);
                Policyalt(3,:,z_c,e_c,jj)=maxindexL2a2alt;
                Policyalt(4,:,z_c,e_c,jj)=maxindexL2a1alt;

                linidx_loweralt=d_indalt                + N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind;
                linidx_upperalt=d_indalt + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2alt-1) + N_d*n2long*N_a2*aind;
                isInfLoweralt   =(ReturnMatrix_ii_zealt(linidx_loweralt)==-Inf);
                isInfUpperalt   =(ReturnMatrix_ii_zealt(linidx_upperalt)==-Inf);
                inLowerStrictalt=(maxindexL2a1alt>=2)         & (maxindexL2a1alt<=n2short+1);
                inUpperStrictalt=(maxindexL2a1alt>=n2short+3) & (maxindexL2a1alt<=n2long-1);
                Policyalt(5,:,z_c,e_c,jj)=2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
                %% Vtilde (beta0*beta)
                entireRHS_ze=ReturnMatrix_ze+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS_ze,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+beta0beta*entireEVinterp_z(aprime),[N_d*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                Vtilde(:,z_c,e_c,jj)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

                allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
                Policy(1,:,z_c,e_c,jj)=d_ind;
                Policy(2,:,z_c,e_c,jj)=midpoint(allind);
                Policy(3,:,z_c,e_c,jj)=maxindexL2a2;
                Policy(4,:,z_c,e_c,jj)=maxindexL2a1;

                linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                Policy(5,:,z_c,e_c,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end
    end
end


%% Post-process
adjust=(Policy(4,:,:,:,:)<1+n2short+1);
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust;
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-(n2short+1)*(~adjust);

adjustalt=(Policyalt(4,:,:,:,:)<1+n2short+1);
Policyalt(2,:,:,:,:)=Policyalt(2,:,:,:,:)-adjustalt;
Policyalt(4,:,:,:,:)=Policyalt(4,:,:,:,:)-(n2short+1)*(~adjustalt);

end
