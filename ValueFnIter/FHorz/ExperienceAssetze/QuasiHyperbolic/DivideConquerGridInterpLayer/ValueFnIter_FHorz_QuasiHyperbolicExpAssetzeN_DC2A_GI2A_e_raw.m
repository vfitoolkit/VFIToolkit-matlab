function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_DC2A_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive QH + ExpAssetze (z+e dep aprimeFn), DC2A_GI2A pattern (with d1).
% lowmemory=0 full vectorization; lowmemory=1 loops z (e vectorized); lowmemory=2 nested z+e.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);
N_e=prod(n_e);

Valt=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray');
Policyalt=zeros(4,N_a,N_z,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray');
PolicyaltL2flag=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray');

aind=gpuArray(0:1:N_a-1);
zindB=shiftdim(gpuArray(0:1:N_z-1),-1);
eindB=shiftdim(gpuArray(0:1:N_e-1),-2);
d2ind_vec=repelem((1:1:N_d2)',N_d1,1);

if vfoptions.lowmemory==0
    midpoint    =zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,N_z,N_e,'gpuArray');
    midpoint_alt=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,N_z,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
    midpoint_z    =zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,1,N_e,'gpuArray');
    midpoint_alt_z=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,1,N_e,'gpuArray');
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
    midpoint_ze    =zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
    midpoint_alt_ze=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
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
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_inner=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                [~,maxindex_inner]=max(ReturnMatrix_ii_inner,[],2);
                midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
            end
        end
        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Valt(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d*n2long))+1;
        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
        Policyalt(1,:,:,:,N_j)=d_ind;
        Policyalt(2,:,:,:,N_j)=midpoint(allind);
        Policyalt(3,:,:,:,N_j)=maxindexL2a2;
        Policyalt(4,:,:,:,N_j)=maxindexL2a1;
        linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyaltL2flag(1,:,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            midpoint_z(:)=0;
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex1_z]=max(ReturnMatrix_ii_z,[],2);
            midpoint_z(:,1,:,level1ii,:,:,1,:)=maxindex1_z;
            maxgap=squeeze(max(max(max(max(max( maxindex1_z(:,1,:,2:end,:,:,1,:)-maxindex1_z(:,1,:,1:end-1,:,:,1,:), [],8),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1_z(:,1,:,ii,:,:,1,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_inner_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii_inner_z,[],2);
                    midpoint_z(:,1,:,curra1inner,:,:,1,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1_z(:,1,:,ii,:,:,1,:);
                    midpoint_z(:,1,:,curra1inner,:,:,1,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint_z=max(min(midpoint_z,N_a1-1),2);
            a1primeindexesfine_z=(midpoint_z+(midpoint_z-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine_z), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            Valt(:,z_c,:,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d*n2long))+1;
            allind_z=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*eindB;
            Policyalt(1,:,z_c,:,N_j)=d_ind;
            Policyalt(2,:,z_c,:,N_j)=midpoint_z(allind_z);
            Policyalt(3,:,z_c,:,N_j)=maxindexL2a2;
            Policyalt(4,:,z_c,:,N_j)=maxindexL2a1;
            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            isInfLower=(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyaltL2flag(1,:,z_c,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                midpoint_ze(:)=0;
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1_ze]=max(ReturnMatrix_ii_ze,[],2);
                midpoint_ze(:,1,:,level1ii,:,:)=maxindex1_ze;
                maxgap=squeeze(max(max(max(max( maxindex1_ze(:,1,:,2:end,:,:)-maxindex1_ze(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1_ze(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_inner_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                        [~,maxindex_inner]=max(ReturnMatrix_ii_inner_ze,[],2);
                        midpoint_ze(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1_ze(:,1,:,ii,:,:);
                        midpoint_ze(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end
                midpoint_ze=max(min(midpoint_ze,N_a1-1),2);
                a1primeindexesfine_ze=(midpoint_ze+(midpoint_ze-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine_ze), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                Valt(:,z_c,e_c,N_j)=Vtempii(:);
                d_ind=rem(maxindexL2-1,N_d)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d*n2long))+1;
                allind_ze=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
                Policyalt(1,:,z_c,e_c,N_j)=d_ind;
                Policyalt(2,:,z_c,e_c,N_j)=midpoint_ze(allind_ze);
                Policyalt(3,:,z_c,e_c,N_j)=maxindexL2a2;
                Policyalt(4,:,z_c,e_c,N_j)=maxindexL2a1;
                linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                PolicyaltL2flag(1,:,z_c,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end
    end
    Vtilde(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy(:,:,:,:,N_j)=Policyalt(:,:,:,:,N_j);
    PolicyL2flag(:,:,:,:,N_j)=PolicyaltL2flag(:,:,:,:,N_j);

else
    if vfoptions.lowmemory~=0
        error('lowmem=1/2 for QH+ExpAssetze N DC2A_GI2A V_Jplus1 init not yet implemented (with d1); use lowmem=0 for terminal')
    end
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a3, n_z, n_e, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*reshape(pi_z_J(:,:,N_j),[1,1,N_z,1,N_z]);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,5));

    EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z,N_e]);
    DiscountedEV_alt=beta*EVbase;
    DiscountedEV    =beta0beta*EVbase;
    DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);
    DiscountedEVinterp    =permute(interp1(a1_grid,permute(DiscountedEV,    [2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

    [Valt(:,:,:,N_j),Policyalt(:,:,:,:,N_j),PolicyaltL2flag(:,:,:,:,N_j),Vtilde(:,:,:,N_j),Policy(:,:,:,:,N_j),PolicyL2flag(:,:,:,:,N_j)]=...
        DC_GI_step_lm0_Nd1(n_d1,n_d2,n_a1,n_a2,n_a3,n_z,n_e,N_a1,N_a2,N_a3,N_a,N_d,N_d1,N_d2,N_z,N_e,N_a1fine,n2short,n2long,level1ii,level1iidiff, ...
            ReturnFn,d_gridvals,a1_grid,a2_gridvals,a3_grid,a1prime_grid,z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec, ...
            DiscountedEV_alt,DiscountedEV,DiscountedEVinterp_alt,DiscountedEVinterp,midpoint,midpoint_alt,aind,zindB,eindB,d2ind_vec);
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

    EVpre=squeeze(sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3));

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a3, n_z, n_e, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*reshape(pi_z_J(:,:,jj),[1,1,N_z,1,N_z]);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,5));

    EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z,N_e]);
    DiscountedEV_alt=beta*EVbase;
    DiscountedEV    =beta0beta*EVbase;
    DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);
    DiscountedEVinterp    =permute(interp1(a1_grid,permute(DiscountedEV,    [2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

    if vfoptions.lowmemory==0
        [Valt(:,:,:,jj),Policyalt(:,:,:,:,jj),PolicyaltL2flag(:,:,:,:,jj),Vtilde(:,:,:,jj),Policy(:,:,:,:,jj),PolicyL2flag(:,:,:,:,jj)]=...
            DC_GI_step_lm0_Nd1(n_d1,n_d2,n_a1,n_a2,n_a3,n_z,n_e,N_a1,N_a2,N_a3,N_a,N_d,N_d1,N_d2,N_z,N_e,N_a1fine,n2short,n2long,level1ii,level1iidiff, ...
                ReturnFn,d_gridvals,a1_grid,a2_gridvals,a3_grid,a1prime_grid,z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),ReturnFnParamsVec, ...
                DiscountedEV_alt,DiscountedEV,DiscountedEVinterp_alt,DiscountedEVinterp,midpoint,midpoint_alt,aind,zindB,eindB,d2ind_vec);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_alt_z       =DiscountedEV_alt      (:,:,:,:,:,:,z_c,:);
            DiscountedEV_z           =DiscountedEV          (:,:,:,:,:,:,z_c,:);
            DiscountedEVinterp_alt_z =DiscountedEVinterp_alt(:,:,:,:,:,:,z_c,:);
            DiscountedEVinterp_z     =DiscountedEVinterp    (:,:,:,:,:,:,z_c,:);
            [Valt(:,z_c,:,jj),Policyalt(:,:,z_c,:,jj),PolicyaltL2flag(:,:,z_c,:,jj),Vtilde(:,z_c,:,jj),Policy(:,:,z_c,:,jj),PolicyL2flag(:,:,z_c,:,jj)]=...
                DC_GI_step_lm1_Nd1(n_d1,n_d2,n_a1,n_a2,n_a3,special_n_z,n_e,N_a1,N_a2,N_a3,N_a,N_d,N_d1,N_d2,N_e,N_a1fine,n2short,n2long,level1ii,level1iidiff, ...
                    ReturnFn,d_gridvals,a1_grid,a2_gridvals,a3_grid,a1prime_grid,z_val,e_gridvals_J(:,:,jj),ReturnFnParamsVec, ...
                    DiscountedEV_alt_z,DiscountedEV_z,DiscountedEVinterp_alt_z,DiscountedEVinterp_z,midpoint_z,midpoint_alt_z,aind,eindB,d2ind_vec);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_alt_ze       =DiscountedEV_alt      (:,:,:,:,:,:,z_c,e_c);
                DiscountedEV_ze           =DiscountedEV          (:,:,:,:,:,:,z_c,e_c);
                DiscountedEVinterp_alt_ze =DiscountedEVinterp_alt(:,:,:,:,:,:,z_c,e_c);
                DiscountedEVinterp_ze     =DiscountedEVinterp    (:,:,:,:,:,:,z_c,e_c);
                [Valt(:,z_c,e_c,jj),Policyalt(:,:,z_c,e_c,jj),PolicyaltL2flag(:,:,z_c,e_c,jj),Vtilde(:,z_c,e_c,jj),Policy(:,:,z_c,e_c,jj),PolicyL2flag(:,:,z_c,e_c,jj)]=...
                    DC_GI_step_lm2_Nd1(n_d1,n_d2,n_a1,n_a2,n_a3,special_n_z,special_n_e,N_a1,N_a2,N_a3,N_a,N_d,N_d1,N_d2,N_a1fine,n2short,n2long,level1ii,level1iidiff, ...
                        ReturnFn,d_gridvals,a1_grid,a2_gridvals,a3_grid,a1prime_grid,z_val,e_val,ReturnFnParamsVec, ...
                        DiscountedEV_alt_ze,DiscountedEV_ze,DiscountedEVinterp_alt_ze,DiscountedEVinterp_ze,midpoint_ze,midpoint_alt_ze,aind,d2ind_vec);
            end
        end
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


%% ============================================================
%% Inner helpers (with d1): N DC coarse + GI fine (one backward step)
%% ============================================================

function [Valt_jj,Policyalt_jj,PolicyaltL2flag_jj,Vtilde_jj,Policy_jj,PolicyL2flag_jj]=DC_GI_step_lm0_Nd1(n_d1,n_d2,n_a1,n_a2,n_a3,n_z,n_e,N_a1,N_a2,N_a3,N_a,N_d,N_d1,N_d2,N_z,N_e,N_a1fine,n2short,n2long,level1ii,level1iidiff,ReturnFn,d_gridvals,a1_grid,a2_gridvals,a3_grid,a1prime_grid,z_gridvals_jj,e_gridvals_jj,ReturnFnParamsVec,DiscountedEV_alt,DiscountedEV,DiscountedEVinterp_alt,DiscountedEVinterp,midpoint,midpoint_alt,aind,zindB,eindB,d2ind_vec)
ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_jj, e_gridvals_jj, ReturnFnParamsVec, 1);
entireRHS_alt=ReturnMatrix_ii+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
[~,maxindex1_alt]=max(entireRHS_alt,[],2);
midpoint_alt(:,1,:,level1ii,:,:,:,:)=maxindex1_alt;
maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
for ii=1:length(level1iidiff)
    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
    if maxgap_alt(ii)>0
        loweredge=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
        a1primeindexes=loweredge+(0:1:maxgap_alt(ii));
        ReturnMatrix_inner=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_jj, e_gridvals_jj, ReturnFnParamsVec, 3);
        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
        entireRHS_inner=ReturnMatrix_inner+DiscountedEV_alt(d2aprimez);
        [~,maxindex_inner]=max(entireRHS_inner,[],2);
        midpoint_alt(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
    else
        loweredge=maxindex1_alt(:,1,:,ii,:,:,:,:);
        midpoint_alt(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
    end
end
entireRHS=ReturnMatrix_ii+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
[~,maxindex1]=max(entireRHS,[],2);
midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
for ii=1:length(level1iidiff)
    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
        a1primeindexes=loweredge+(0:1:maxgap(ii));
        ReturnMatrix_inner=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_jj, e_gridvals_jj, ReturnFnParamsVec, 3);
        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
        entireRHS_inner=ReturnMatrix_inner+DiscountedEV(d2aprimez);
        [~,maxindex_inner]=max(entireRHS_inner,[],2);
        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
    else
        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
    end
end
midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_jj, e_gridvals_jj, ReturnFnParamsVec, 3);
aprimez_alt=d2ind_vec + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+DiscountedEVinterp_alt(aprimez_alt),[N_d*n2long*N_a2,N_a,N_z,N_e]);
[Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_alt,[],1);
Valt_jj=shiftdim(Vtempii_alt,1);
d_ind_alt=rem(maxindexL2_alt-1,N_d)+1;
maxindexL2a1_alt=rem(floor((maxindexL2_alt-1)/N_d),n2long)+1;
maxindexL2a2_alt=floor((maxindexL2_alt-1)/(N_d*n2long))+1;
allind_alt=d_ind_alt + N_d*(maxindexL2a2_alt-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
Policyalt_jj=zeros(4,N_a,N_z,N_e,'gpuArray');
Policyalt_jj(1,:,:,:)=d_ind_alt;
Policyalt_jj(2,:,:,:)=midpoint_alt(allind_alt);
Policyalt_jj(3,:,:,:)=maxindexL2a2_alt;
Policyalt_jj(4,:,:,:)=maxindexL2a1_alt;
RM_alt_flat=reshape(ReturnMatrix_ii_alt,[N_d*n2long*N_a2,N_a,N_z,N_e]);
linidx_lower=d_ind_alt                  + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
linidx_upper=d_ind_alt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
inLowerStrict=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
inUpperStrict=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
PolicyaltL2flag_jj=2 + (inLowerStrict & (RM_alt_flat(linidx_lower)==-Inf)) - (inUpperStrict & (RM_alt_flat(linidx_upper)==-Inf));

midpoint=max(min(midpoint,N_a1-1),2);
a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_jj, e_gridvals_jj, ReturnFnParamsVec, 3);
aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d*n2long*N_a2,N_a,N_z,N_e]);
[Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
Vtilde_jj=shiftdim(Vtempii,1);
d_ind=rem(maxindexL2-1,N_d)+1;
maxindexL2a1=rem(floor((maxindexL2-1)/N_d),n2long)+1;
maxindexL2a2=floor((maxindexL2-1)/(N_d*n2long))+1;
allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
Policy_jj=zeros(4,N_a,N_z,N_e,'gpuArray');
Policy_jj(1,:,:,:)=d_ind;
Policy_jj(2,:,:,:)=midpoint(allind);
Policy_jj(3,:,:,:)=maxindexL2a2;
Policy_jj(4,:,:,:)=maxindexL2a1;
RM_flat=reshape(ReturnMatrix_ii,[N_d*n2long*N_a2,N_a,N_z,N_e]);
linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
PolicyL2flag_jj=2 + (inLowerStrict & (RM_flat(linidx_lower)==-Inf)) - (inUpperStrict & (RM_flat(linidx_upper)==-Inf));
end


function [Valt_jj,Policyalt_jj,PolicyaltL2flag_jj,Vtilde_jj,Policy_jj,PolicyL2flag_jj]=DC_GI_step_lm1_Nd1(n_d1,n_d2,n_a1,n_a2,n_a3,special_n_z,n_e,N_a1,N_a2,N_a3,N_a,N_d,N_d1,N_d2,N_e,N_a1fine,n2short,n2long,level1ii,level1iidiff,ReturnFn,d_gridvals,a1_grid,a2_gridvals,a3_grid,a1prime_grid,z_val,e_gridvals_jj,ReturnFnParamsVec,DiscountedEV_alt_z,DiscountedEV_z,DiscountedEVinterp_alt_z,DiscountedEVinterp_z,midpoint_z,midpoint_alt_z,aind,eindB,d2ind_vec)
ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_gridvals_jj, ReturnFnParamsVec, 1);
entireRHS_alt_z=ReturnMatrix_ii_z+repelem(DiscountedEV_alt_z,N_d1,1,1,1,1,1,1);
[~,maxindex1_alt_z]=max(entireRHS_alt_z,[],2);
midpoint_alt_z(:,1,:,level1ii,:,:,1,:)=maxindex1_alt_z;
maxgap_alt=squeeze(max(max(max(max(max( maxindex1_alt_z(:,1,:,2:end,:,:,1,:)-maxindex1_alt_z(:,1,:,1:end-1,:,:,1,:), [],8),[],6),[],5),[],3),[],1));
for ii=1:length(level1iidiff)
    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
    if maxgap_alt(ii)>0
        loweredge=min(maxindex1_alt_z(:,1,:,ii,:,:,1,:),N_a1-maxgap_alt(ii));
        a1primeindexes=loweredge+(0:1:maxgap_alt(ii));
        ReturnMatrix_inner_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_gridvals_jj, ReturnFnParamsVec, 3);
        d2aprime_z=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
        entireRHS_inner_z=ReturnMatrix_inner_z+DiscountedEV_alt_z(d2aprime_z);
        [~,maxindex_inner]=max(entireRHS_inner_z,[],2);
        midpoint_alt_z(:,1,:,curra1inner,:,:,1,:)=maxindex_inner+(loweredge-1);
    else
        loweredge=maxindex1_alt_z(:,1,:,ii,:,:,1,:);
        midpoint_alt_z(:,1,:,curra1inner,:,:,1,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
    end
end
entireRHS_z=ReturnMatrix_ii_z+repelem(DiscountedEV_z,N_d1,1,1,1,1,1,1);
[~,maxindex1_z]=max(entireRHS_z,[],2);
midpoint_z(:,1,:,level1ii,:,:,1,:)=maxindex1_z;
maxgap=squeeze(max(max(max(max(max( maxindex1_z(:,1,:,2:end,:,:,1,:)-maxindex1_z(:,1,:,1:end-1,:,:,1,:), [],8),[],6),[],5),[],3),[],1));
for ii=1:length(level1iidiff)
    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
    if maxgap(ii)>0
        loweredge=min(maxindex1_z(:,1,:,ii,:,:,1,:),N_a1-maxgap(ii));
        a1primeindexes=loweredge+(0:1:maxgap(ii));
        ReturnMatrix_inner_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_gridvals_jj, ReturnFnParamsVec, 3);
        d2aprime_z=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
        entireRHS_inner_z=ReturnMatrix_inner_z+DiscountedEV_z(d2aprime_z);
        [~,maxindex_inner]=max(entireRHS_inner_z,[],2);
        midpoint_z(:,1,:,curra1inner,:,:,1,:)=maxindex_inner+(loweredge-1);
    else
        loweredge=maxindex1_z(:,1,:,ii,:,:,1,:);
        midpoint_z(:,1,:,curra1inner,:,:,1,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
    end
end
midpoint_alt_z=max(min(midpoint_alt_z,N_a1-1),2);
a1primeindexesfine_alt=(midpoint_alt_z+(midpoint_alt_z-1)*n2short)+(-n2short-1:1:1+n2short);
ReturnMatrix_ii_alt_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_jj, ReturnFnParamsVec, 3);
aprimez_alt_z=d2ind_vec + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
entireRHS_ii_alt_z=reshape(ReturnMatrix_ii_alt_z+DiscountedEVinterp_alt_z(aprimez_alt_z),[N_d*n2long*N_a2,N_a,1,N_e]);
[Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_alt_z,[],1);
Valt_jj=shiftdim(Vtempii_alt,1);
d_ind_alt=rem(maxindexL2_alt-1,N_d)+1;
maxindexL2a1_alt=rem(floor((maxindexL2_alt-1)/N_d),n2long)+1;
maxindexL2a2_alt=floor((maxindexL2_alt-1)/(N_d*n2long))+1;
allind_alt_z=d_ind_alt + N_d*(maxindexL2a2_alt-1) + N_d*N_a2*aind + N_d*N_a2*N_a*eindB;
Policyalt_jj=zeros(4,N_a,1,N_e,'gpuArray');
Policyalt_jj(1,:,1,:)=d_ind_alt;
Policyalt_jj(2,:,1,:)=midpoint_alt_z(allind_alt_z);
Policyalt_jj(3,:,1,:)=maxindexL2a2_alt;
Policyalt_jj(4,:,1,:)=maxindexL2a1_alt;
RM_alt_flat=reshape(ReturnMatrix_ii_alt_z,[N_d*n2long*N_a2,N_a,1,N_e]);
linidx_lower=d_ind_alt                  + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
linidx_upper=d_ind_alt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
inLowerStrict=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
inUpperStrict=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
PolicyaltL2flag_jj=2 + (inLowerStrict & (RM_alt_flat(linidx_lower)==-Inf)) - (inUpperStrict & (RM_alt_flat(linidx_upper)==-Inf));

midpoint_z=max(min(midpoint_z,N_a1-1),2);
a1primeindexesfine=(midpoint_z+(midpoint_z-1)*n2short)+(-n2short-1:1:1+n2short);
ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_jj, ReturnFnParamsVec, 3);
aprimez_z=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprimez_z),[N_d*n2long*N_a2,N_a,1,N_e]);
[Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
Vtilde_jj=shiftdim(Vtempii,1);
d_ind=rem(maxindexL2-1,N_d)+1;
maxindexL2a1=rem(floor((maxindexL2-1)/N_d),n2long)+1;
maxindexL2a2=floor((maxindexL2-1)/(N_d*n2long))+1;
allind_z=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*eindB;
Policy_jj=zeros(4,N_a,1,N_e,'gpuArray');
Policy_jj(1,:,1,:)=d_ind;
Policy_jj(2,:,1,:)=midpoint_z(allind_z);
Policy_jj(3,:,1,:)=maxindexL2a2;
Policy_jj(4,:,1,:)=maxindexL2a1;
RM_flat=reshape(ReturnMatrix_ii_z,[N_d*n2long*N_a2,N_a,1,N_e]);
linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
PolicyL2flag_jj=2 + (inLowerStrict & (RM_flat(linidx_lower)==-Inf)) - (inUpperStrict & (RM_flat(linidx_upper)==-Inf));
end


function [Valt_jj,Policyalt_jj,PolicyaltL2flag_jj,Vtilde_jj,Policy_jj,PolicyL2flag_jj]=DC_GI_step_lm2_Nd1(n_d1,n_d2,n_a1,n_a2,n_a3,special_n_z,special_n_e,N_a1,N_a2,N_a3,N_a,N_d,N_d1,N_d2,N_a1fine,n2short,n2long,level1ii,level1iidiff,ReturnFn,d_gridvals,a1_grid,a2_gridvals,a3_grid,a1prime_grid,z_val,e_val,ReturnFnParamsVec,DiscountedEV_alt_ze,DiscountedEV_ze,DiscountedEVinterp_alt_ze,DiscountedEVinterp_ze,midpoint_ze,midpoint_alt_ze,aind,d2ind_vec)
ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
entireRHS_alt_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_alt_ze,N_d1,1,1,1,1,1);
[~,maxindex1_alt_ze]=max(entireRHS_alt_ze,[],2);
midpoint_alt_ze(:,1,:,level1ii,:,:)=maxindex1_alt_ze;
maxgap_alt=squeeze(max(max(max(max( maxindex1_alt_ze(:,1,:,2:end,:,:)-maxindex1_alt_ze(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
for ii=1:length(level1iidiff)
    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
    if maxgap_alt(ii)>0
        loweredge=min(maxindex1_alt_ze(:,1,:,ii,:,:),N_a1-maxgap_alt(ii));
        a1primeindexes=loweredge+(0:1:maxgap_alt(ii));
        ReturnMatrix_inner_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
        d2aprime_ze=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
        entireRHS_inner_ze=ReturnMatrix_inner_ze+DiscountedEV_alt_ze(d2aprime_ze);
        [~,maxindex_inner]=max(entireRHS_inner_ze,[],2);
        midpoint_alt_ze(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
    else
        loweredge=maxindex1_alt_ze(:,1,:,ii,:,:);
        midpoint_alt_ze(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
    end
end
entireRHS_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_ze,N_d1,1,1,1,1,1);
[~,maxindex1_ze]=max(entireRHS_ze,[],2);
midpoint_ze(:,1,:,level1ii,:,:)=maxindex1_ze;
maxgap=squeeze(max(max(max(max( maxindex1_ze(:,1,:,2:end,:,:)-maxindex1_ze(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
for ii=1:length(level1iidiff)
    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
    if maxgap(ii)>0
        loweredge=min(maxindex1_ze(:,1,:,ii,:,:),N_a1-maxgap(ii));
        a1primeindexes=loweredge+(0:1:maxgap(ii));
        ReturnMatrix_inner_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
        d2aprime_ze=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
        entireRHS_inner_ze=ReturnMatrix_inner_ze+DiscountedEV_ze(d2aprime_ze);
        [~,maxindex_inner]=max(entireRHS_inner_ze,[],2);
        midpoint_ze(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
    else
        loweredge=maxindex1_ze(:,1,:,ii,:,:);
        midpoint_ze(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
    end
end
midpoint_alt_ze=max(min(midpoint_alt_ze,N_a1-1),2);
a1primeindexesfine_alt=(midpoint_alt_ze+(midpoint_alt_ze-1)*n2short)+(-n2short-1:1:1+n2short);
ReturnMatrix_ii_alt_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
aprimez_alt_ze=d2ind_vec + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
entireRHS_ii_alt_ze=reshape(ReturnMatrix_ii_alt_ze+DiscountedEVinterp_alt_ze(aprimez_alt_ze),[N_d*n2long*N_a2,N_a]);
[Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_alt_ze,[],1);
Valt_jj=Vtempii_alt(:);
d_ind_alt=rem(maxindexL2_alt-1,N_d)+1;
maxindexL2a1_alt=rem(floor((maxindexL2_alt-1)/N_d),n2long)+1;
maxindexL2a2_alt=floor((maxindexL2_alt-1)/(N_d*n2long))+1;
allind_alt_ze=d_ind_alt + N_d*(maxindexL2a2_alt-1) + N_d*N_a2*aind;
Policyalt_jj=zeros(4,N_a,'gpuArray');
Policyalt_jj(1,:)=d_ind_alt;
Policyalt_jj(2,:)=midpoint_alt_ze(allind_alt_ze);
Policyalt_jj(3,:)=maxindexL2a2_alt;
Policyalt_jj(4,:)=maxindexL2a1_alt;
RM_alt_flat=reshape(ReturnMatrix_ii_alt_ze,[N_d*n2long*N_a2,N_a]);
linidx_lower=d_ind_alt                  + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind;
linidx_upper=d_ind_alt + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2_alt-1) + N_d*n2long*N_a2*aind;
inLowerStrict=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
inUpperStrict=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
PolicyaltL2flag_jj=2 + (inLowerStrict & (RM_alt_flat(linidx_lower)==-Inf)) - (inUpperStrict & (RM_alt_flat(linidx_upper)==-Inf));

midpoint_ze=max(min(midpoint_ze,N_a1-1),2);
a1primeindexesfine=(midpoint_ze+(midpoint_ze-1)*n2short)+(-n2short-1:1:1+n2short);
ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, special_n_z, special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
aprimez_ze=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEVinterp_ze(aprimez_ze),[N_d*n2long*N_a2,N_a]);
[Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
Vtilde_jj=Vtempii(:);
d_ind=rem(maxindexL2-1,N_d)+1;
maxindexL2a1=rem(floor((maxindexL2-1)/N_d),n2long)+1;
maxindexL2a2=floor((maxindexL2-1)/(N_d*n2long))+1;
allind_ze=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
Policy_jj=zeros(4,N_a,'gpuArray');
Policy_jj(1,:)=d_ind;
Policy_jj(2,:)=midpoint_ze(allind_ze);
Policy_jj(3,:)=maxindexL2a2;
Policy_jj(4,:)=maxindexL2a1;
RM_flat=reshape(ReturnMatrix_ii_ze,[N_d*n2long*N_a2,N_a]);
linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
PolicyL2flag_jj=2 + (inLowerStrict & (RM_flat(linidx_lower)==-Inf)) - (inUpperStrict & (RM_flat(linidx_upper)==-Inf));
end
