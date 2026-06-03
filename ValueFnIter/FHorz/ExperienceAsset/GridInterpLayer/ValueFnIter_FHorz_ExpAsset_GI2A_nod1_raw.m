function [V,Policy]=ValueFnIter_FHorz_ExpAsset_GI2A_nod1_raw(n_d2, n_a1, n_a2, n_a3, n_z, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Grid Interpolation Layer on the first standard endogenous state (n_a1, scalar),
% folded standard middle endogenous states (n_a2, may be multi-dim),
% experience asset as the last endogenous state (n_a3, scalar).
% _nod1: only d2 (no d1).
% Policy is 4-channel: 1=d2, 2=a1prime midpoint, 3=a2prime, 4=a1prime L2.
% A 5th channel PolicyL2flag is concatenated at the end.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_j,'gpuArray'); % 1=d2, 2=a1prime midpoint, 3=a2prime, 4=a1prime L2 fine
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray');

%% GI setup
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
else
    aind=gpuArray(0:1:N_a-1);                       % flat a-axis
    zindB=shiftdim(gpuArray(0:1:N_z-1),-1);         % at dim 3 of [1,N_a,N_z]
end

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % --- Coarse pass ---
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        % [N_d2, N_a1, N_a2, N_a1, N_a2, N_a3, N_z]
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,N_a1-1),2); % [N_d2,1,N_a2,N_a1,N_a2,N_a3,N_z]

        % --- Fine pass ---
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        % [N_d2,n2long,N_a2,N_a1,N_a2,N_a3,N_z]
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        % [N_d2*n2long*N_a2, N_a, N_z]
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d2)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

        allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*zindB;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,N_j)=maxindexL2a1;

        linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        aind_z=gpuArray(0:1:N_a-1);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_z,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind_z;
            Policy(1,:,z_c,N_j)=d_ind;
            Policy(2,:,z_c,N_j)=midpoint(allind);
            Policy(3,:,z_c,N_j)=maxindexL2a2;
            Policy(4,:,z_c,N_j)=maxindexL2a1;

            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind_z;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind_z;
            isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end

else
    % vfoptions.V_Jplus1 provided
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a3, d2_gridvals, a3_grid, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,3)); % (d2*a1prime*a2prime, a3, z)

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]);
    % Interpolate over a1prime (dim 2)
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
    % [N_d2,N_a1fine,N_a2,1,1,N_a3,N_z]

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        entireRHS=ReturnMatrix+DiscountedEV;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,N_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        % [N_d2,n2long,N_a2,N_a1,N_a2,N_a3,N_z]
        aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d2)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

        allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*zindB;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,N_j)=maxindexL2a1;

        linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        aind_z=gpuArray(0:1:N_a-1);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            entireRHS_z=ReturnMatrix_z+DiscountedEV_z;
            [~,maxindex]=max(entireRHS_z,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprime_z=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprime_z),[N_d2*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind_z;
            Policy(1,:,z_c,N_j)=d_ind;
            Policy(2,:,z_c,N_j)=midpoint(allind);
            Policy(3,:,z_c,N_j)=maxindexL2a2;
            Policy(4,:,z_c,N_j)=maxindexL2a1;

            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind_z;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind_z;
            isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a3, d2_gridvals, a3_grid, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,N_z);

    Vlower=reshape(V(aprimeIndex(:),:,jj+1),    [N_d2*N_a1*N_a2,N_a3,N_z]);
    Vupper=reshape(V(aprimeplus1Index(:),:,jj+1),[N_d2*N_a1*N_a2,N_a3,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,3));

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS=ReturnMatrix+DiscountedEV;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,N_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d2)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

        allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*zindB;
        Policy(1,:,:,jj)=d_ind;
        Policy(2,:,:,jj)=midpoint(allind);
        Policy(3,:,:,jj)=maxindexL2a2;
        Policy(4,:,:,jj)=maxindexL2a1;

        linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        aind_z=gpuArray(0:1:N_a-1);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            entireRHS_z=ReturnMatrix_z+DiscountedEV_z;
            [~,maxindex]=max(entireRHS_z,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprime_z=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprime_z),[N_d2*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            V(:,z_c,jj)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind_z;
            Policy(1,:,z_c,jj)=d_ind;
            Policy(2,:,z_c,jj)=midpoint(allind);
            Policy(3,:,z_c,jj)=maxindexL2a2;
            Policy(4,:,z_c,jj)=maxindexL2a1;

            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind_z;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind_z;
            isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end
end


%% Post-process: convert "midpoint + L2 offset" into "lower coarse point + L2 ratio"
% Currently Policy(2,:) is the midpoint, Policy(4,:) is the L2 index (ranges -n2short-1:1:1+n2short).
% Switch Policy(2,:) to 'lower grid point', and Policy(4,:) to a 1..(n2short+2) offset.
adjust=(Policy(4,:,:,:)<1+n2short+1); % is the L2 index below midpoint?
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

end
