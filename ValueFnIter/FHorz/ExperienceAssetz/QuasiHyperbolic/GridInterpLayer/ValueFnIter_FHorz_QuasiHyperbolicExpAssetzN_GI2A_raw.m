function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI2A_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive QH + ExpAssetz, GI2A pattern (with d1).
% Two independent argmax passes per period: one with beta (Valt, Policyalt); one with beta0*beta (Vtilde, Policy).
% Policy/Policyalt channels: 1=d (joint), 2=a1prime lower coarse, 3=a2prime, 4=L2 fine, plus L2flag concatenated.
% lowmemory=0 full vectorization; lowmemory=1 loops z; lowmemory=2 errors (no e dim to nest).

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);

Valt=zeros(N_a,N_z,N_j,'gpuArray');
Vtilde=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_j,'gpuArray');
Policyalt=zeros(4,N_a,N_z,N_j,'gpuArray');
PolicyL2flag    =2*ones(1,N_a,N_z,N_j,'gpuArray');
PolicyaltL2flag =2*ones(1,N_a,N_z,N_j,'gpuArray');

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zindB=shiftdim(gpuArray(0:1:N_z-1),-1);
d2ind_vec=repelem((1:1:N_d2)',N_d1,1);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,N_a1-1),2);
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Valt(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;
        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB;
        Policyalt(1,:,:,N_j)=d_ind;
        Policyalt(2,:,:,N_j)=midpoint(allind);
        Policyalt(3,:,:,N_j)=maxindexL2a2;
        Policyalt(4,:,:,N_j)=maxindexL2a1;
        linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyaltL2flag(1,:,:,N_j)=2 + (inLowerStrict & (ReturnMatrix_ii(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii(linidx_upper)==-Inf));
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_z,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            Valt(:,z_c,N_j)=Vtempii(:);
            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;
            allind_z=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
            Policyalt(1,:,z_c,N_j)=d_ind;
            Policyalt(2,:,z_c,N_j)=midpoint(allind_z);
            Policyalt(3,:,z_c,N_j)=maxindexL2a2;
            Policyalt(4,:,z_c,N_j)=maxindexL2a1;
            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyaltL2flag(1,:,z_c,N_j)=2 + (inLowerStrict & (ReturnMatrix_ii_z(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii_z(linidx_upper)==-Inf));
        end
    elseif vfoptions.lowmemory==2
        error('lowmemory=2 not supported in QH+ExpAssetz _raw (no e dim to nest); use _e_raw variant')
    end

    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy(:,:,:,N_j)=Policyalt(:,:,:,N_j);
    PolicyL2flag(:,:,:,N_j)=PolicyaltL2flag(:,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,N_j),-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]);
    DiscountedEV_alt=beta*EVbase;
    DiscountedEV    =beta0beta*EVbase;
    DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
    DiscountedEVinterp    =permute(interp1(a1_grid,permute(DiscountedEV,    [2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

        entireRHS_alt=ReturnMatrix+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
        [~,maxindex_alt]=max(entireRHS_alt,[],2);
        midpoint_alt=max(min(maxindex_alt,N_a1-1),2);
        a1primeindexes_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1prime_grid(a1primeindexes_alt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        aprimez_alt=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+DiscountedEVinterp_alt(aprimez_alt),[N_d*n2long*N_a2,N_a,N_z]);
        [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_alt,[],1);
        Valt(:,:,N_j)=shiftdim(Vtempii_alt,1);
        d_ind_alt        =rem(maxindexL2_alt-1,N_d)+1;
        maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d),n2long)+1;
        maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d*n2long))+1;
        allind_alt=d_ind_alt + N_d*(maxindexL2a2_alt-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB;
        Policyalt(1,:,:,N_j)=d_ind_alt;
        Policyalt(2,:,:,N_j)=midpoint_alt(allind_alt);
        Policyalt(3,:,:,N_j)=maxindexL2a2_alt;
        Policyalt(4,:,:,N_j)=maxindexL2a1_alt;
        linidx_lower=d_ind_alt                  + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind_alt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        inLowerStrict=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
        inUpperStrict=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
        PolicyaltL2flag(1,:,:,N_j)=2 + (inLowerStrict & (ReturnMatrix_ii_alt(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii_alt(linidx_upper)==-Inf));

        entireRHS=ReturnMatrix+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,N_a1-1),2);
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d*n2long*N_a2,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;
        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,N_j)=maxindexL2a1;
        linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & (ReturnMatrix_ii(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii(linidx_upper)==-Inf));
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_alt_z       =DiscountedEV_alt      (:,:,:,:,:,:,z_c);
            DiscountedEV_z           =DiscountedEV          (:,:,:,:,:,:,z_c);
            DiscountedEVinterp_alt_z =DiscountedEVinterp_alt(:,:,:,:,:,:,z_c);
            DiscountedEVinterp_z     =DiscountedEVinterp    (:,:,:,:,:,:,z_c);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);

            entireRHS_alt_z=ReturnMatrix_z+repelem(DiscountedEV_alt_z,N_d1,1,1,1,1,1,1);
            [~,maxindex_alt]=max(entireRHS_alt_z,[],2);
            midpoint_alt=max(min(maxindex_alt,N_a1-1),2);
            a1primeindexes_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_alt_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1prime_grid(a1primeindexes_alt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprimez_alt_z=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_alt_z=reshape(ReturnMatrix_ii_alt_z+DiscountedEVinterp_alt_z(aprimez_alt_z),[N_d*n2long*N_a2,N_a]);
            [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_alt_z,[],1);
            Valt(:,z_c,N_j)=Vtempii_alt(:);
            d_ind_alt        =rem(maxindexL2_alt-1,N_d)+1;
            maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d),n2long)+1;
            maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d*n2long))+1;
            allind_alt_z=d_ind_alt + N_d*(maxindexL2a2_alt-1) + N_d*N_a2*aind;
            Policyalt(1,:,z_c,N_j)=d_ind_alt;
            Policyalt(2,:,z_c,N_j)=midpoint_alt(allind_alt_z);
            Policyalt(3,:,z_c,N_j)=maxindexL2a2_alt;
            Policyalt(4,:,z_c,N_j)=maxindexL2a1_alt;
            linidx_lower=d_ind_alt                  + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind;
            linidx_upper=d_ind_alt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind;
            inLowerStrict=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
            inUpperStrict=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
            PolicyaltL2flag(1,:,z_c,N_j)=2 + (inLowerStrict & (ReturnMatrix_ii_alt_z(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii_alt_z(linidx_upper)==-Inf));

            entireRHS_z=ReturnMatrix_z+repelem(DiscountedEV_z,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS_z,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprimez_z=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprimez_z),[N_d*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            Vtilde(:,z_c,N_j)=Vtempii(:);
            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;
            allind_z=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
            Policy(1,:,z_c,N_j)=d_ind;
            Policy(2,:,z_c,N_j)=midpoint(allind_z);
            Policy(3,:,z_c,N_j)=maxindexL2a2;
            Policy(4,:,z_c,N_j)=maxindexL2a1;
            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,N_j)=2 + (inLowerStrict & (ReturnMatrix_ii_z(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii_z(linidx_upper)==-Inf));
        end
    elseif vfoptions.lowmemory==2
        error('lowmemory=2 not supported in QH+ExpAssetz _raw (no e dim to nest); use _e_raw variant')
    end
end


%% Backward iteration
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

    EVpre=Valt(:,:,jj+1);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,jj),-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]);
    DiscountedEV_alt=beta*EVbase;
    DiscountedEV    =beta0beta*EVbase;
    DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
    DiscountedEVinterp    =permute(interp1(a1_grid,permute(DiscountedEV,    [2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);

        entireRHS_alt=ReturnMatrix+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
        [~,maxindex_alt]=max(entireRHS_alt,[],2);
        midpoint_alt=max(min(maxindex_alt,N_a1-1),2);
        a1primeindexes_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1prime_grid(a1primeindexes_alt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimez_alt=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+DiscountedEVinterp_alt(aprimez_alt),[N_d*n2long*N_a2,N_a,N_z]);
        [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_alt,[],1);
        Valt(:,:,jj)=shiftdim(Vtempii_alt,1);
        d_ind_alt        =rem(maxindexL2_alt-1,N_d)+1;
        maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d),n2long)+1;
        maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d*n2long))+1;
        allind_alt=d_ind_alt + N_d*(maxindexL2a2_alt-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB;
        Policyalt(1,:,:,jj)=d_ind_alt;
        Policyalt(2,:,:,jj)=midpoint_alt(allind_alt);
        Policyalt(3,:,:,jj)=maxindexL2a2_alt;
        Policyalt(4,:,:,jj)=maxindexL2a1_alt;
        linidx_lower=d_ind_alt                  + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind_alt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        inLowerStrict=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
        inUpperStrict=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
        PolicyaltL2flag(1,:,:,jj)=2 + (inLowerStrict & (ReturnMatrix_ii_alt(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii_alt(linidx_upper)==-Inf));

        entireRHS=ReturnMatrix+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,N_a1-1),2);
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d*n2long*N_a2,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,:,jj)=shiftdim(Vtempii,1);
        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;
        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB;
        Policy(1,:,:,jj)=d_ind;
        Policy(2,:,:,jj)=midpoint(allind);
        Policy(3,:,:,jj)=maxindexL2a2;
        Policy(4,:,:,jj)=maxindexL2a1;
        linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,jj)=2 + (inLowerStrict & (ReturnMatrix_ii(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii(linidx_upper)==-Inf));
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_alt_z       =DiscountedEV_alt      (:,:,:,:,:,:,z_c);
            DiscountedEV_z           =DiscountedEV          (:,:,:,:,:,:,z_c);
            DiscountedEVinterp_alt_z =DiscountedEVinterp_alt(:,:,:,:,:,:,z_c);
            DiscountedEVinterp_z     =DiscountedEVinterp    (:,:,:,:,:,:,z_c);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);

            entireRHS_alt_z=ReturnMatrix_z+repelem(DiscountedEV_alt_z,N_d1,1,1,1,1,1,1);
            [~,maxindex_alt]=max(entireRHS_alt_z,[],2);
            midpoint_alt=max(min(maxindex_alt,N_a1-1),2);
            a1primeindexes_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_alt_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1prime_grid(a1primeindexes_alt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprimez_alt_z=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_alt_z=reshape(ReturnMatrix_ii_alt_z+DiscountedEVinterp_alt_z(aprimez_alt_z),[N_d*n2long*N_a2,N_a]);
            [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_alt_z,[],1);
            Valt(:,z_c,jj)=Vtempii_alt(:);
            d_ind_alt        =rem(maxindexL2_alt-1,N_d)+1;
            maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d),n2long)+1;
            maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d*n2long))+1;
            allind_alt_z=d_ind_alt + N_d*(maxindexL2a2_alt-1) + N_d*N_a2*aind;
            Policyalt(1,:,z_c,jj)=d_ind_alt;
            Policyalt(2,:,z_c,jj)=midpoint_alt(allind_alt_z);
            Policyalt(3,:,z_c,jj)=maxindexL2a2_alt;
            Policyalt(4,:,z_c,jj)=maxindexL2a1_alt;
            linidx_lower=d_ind_alt                  + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind;
            linidx_upper=d_ind_alt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind;
            inLowerStrict=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
            inUpperStrict=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
            PolicyaltL2flag(1,:,z_c,jj)=2 + (inLowerStrict & (ReturnMatrix_ii_alt_z(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii_alt_z(linidx_upper)==-Inf));

            entireRHS_z=ReturnMatrix_z+repelem(DiscountedEV_z,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS_z,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprimez_z=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprimez_z),[N_d*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            Vtilde(:,z_c,jj)=Vtempii(:);
            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;
            allind_z=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
            Policy(1,:,z_c,jj)=d_ind;
            Policy(2,:,z_c,jj)=midpoint(allind_z);
            Policy(3,:,z_c,jj)=maxindexL2a2;
            Policy(4,:,z_c,jj)=maxindexL2a1;
            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,jj)=2 + (inLowerStrict & (ReturnMatrix_ii_z(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii_z(linidx_upper)==-Inf));
        end
    elseif vfoptions.lowmemory==2
        error('lowmemory=2 not supported in QH+ExpAssetz _raw (no e dim to nest); use _e_raw variant')
    end
end


%% Post-process
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);
Policy=[Policy;PolicyL2flag];

adjust_alt=(Policyalt(4,:,:,:)<1+n2short+1);
Policyalt(2,:,:,:)=Policyalt(2,:,:,:)-adjust_alt;
Policyalt(4,:,:,:)=adjust_alt.*Policyalt(4,:,:,:)+(1-adjust_alt).*(Policyalt(4,:,:,:)-n2short-1);
Policyalt=[Policyalt;PolicyaltL2flag];

end
