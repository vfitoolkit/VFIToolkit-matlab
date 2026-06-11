function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_DC2A_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated QH + ExpAssetz, DC2A_GI2A pattern (with d1).
% Single argmax with beta0*beta -> Vhat, Policy; Vunderbar by lookup at same policy under beta.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);
N_e=prod(n_e);

Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray');

aind=gpuArray(0:1:N_a-1);
zindB=shiftdim(gpuArray(0:1:N_z-1),-1);
eindB=shiftdim(gpuArray(0:1:N_e-1),-2);
d2ind_vec=repelem((1:1:N_d2)',N_d1,1);

if vfoptions.lowmemory==0
    midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,N_z,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
    midpoint_z=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,1,N_e,'gpuArray');
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
    midpoint_ze=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
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
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;

    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_inner=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            [~,maxindex_inner]=max(ReturnMatrix_inner,[],2);
            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
        end
    end

    midpoint=max(min(midpoint,N_a1-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    Vhat(:,:,:,N_j)=shiftdim(Vtempii,1);

    d_ind        =rem(maxindexL2-1,N_d)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

    allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
    Policy(1,:,:,:,N_j)=d_ind;
    Policy(2,:,:,:,N_j)=midpoint(allind);
    Policy(3,:,:,:,N_j)=maxindexL2a2;
    Policy(4,:,:,:,N_j)=maxindexL2a1;

    linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
    linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    PolicyL2flag(1,:,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex1_z]=max(ReturnMatrix_ii_z,[],2);
            midpoint_z(:,1,:,level1ii,:,:,1,:)=maxindex1_z;
            maxgap=squeeze(max(max(max(max(max( maxindex1_z(:,1,:,2:end,:,:,1,:)-maxindex1_z(:,1,:,1:end-1,:,:,1,:), [],8),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1_z(:,1,:,ii,:,:,1,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_inner_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_inner_z,[],2);
                    midpoint_z(:,1,:,curra1inner,:,:,1,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1_z(:,1,:,ii,:,:,1,:);
                    midpoint_z(:,1,:,curra1inner,:,:,1,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint_z=max(min(midpoint_z,N_a1-1),2);
            a1primeindexesfine_z=(midpoint_z+(midpoint_z-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine_z), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            Vhat(:,z_c,:,N_j)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;
            allind_z=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*eindB;
            Policy(1,:,z_c,:,N_j)=d_ind;
            Policy(2,:,z_c,:,N_j)=midpoint_z(allind_z);
            Policy(3,:,z_c,:,N_j)=maxindexL2a2;
            Policy(4,:,z_c,:,N_j)=maxindexL2a1;
            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,:,N_j)=2 + (inLowerStrict & (ReturnMatrix_ii_z(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii_z(linidx_upper)==-Inf));
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1_ze]=max(ReturnMatrix_ii_ze,[],2);
                midpoint_ze(:,1,:,level1ii,:,:)=maxindex1_ze;
                maxgap=squeeze(max(max(max(max( maxindex1_ze(:,1,:,2:end,:,:)-maxindex1_ze(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1_ze(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_inner_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                        [~,maxindex_inner]=max(ReturnMatrix_inner_ze,[],2);
                        midpoint_ze(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1_ze(:,1,:,ii,:,:);
                        midpoint_ze(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end
                midpoint_ze=max(min(midpoint_ze,N_a1-1),2);
                a1primeindexesfine_ze=(midpoint_ze+(midpoint_ze-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine_ze), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                Vhat(:,z_c,e_c,N_j)=Vtempii(:);
                d_ind        =rem(maxindexL2-1,N_d)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;
                allind_ze=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
                Policy(1,:,z_c,e_c,N_j)=d_ind;
                Policy(2,:,z_c,e_c,N_j)=midpoint_ze(allind_ze);
                Policy(3,:,z_c,e_c,N_j)=maxindexL2a2;
                Policy(4,:,z_c,e_c,N_j)=maxindexL2a1;
                linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j)=2 + (inLowerStrict & (ReturnMatrix_ii_ze(linidx_lower)==-Inf)) - (inUpperStrict & (ReturnMatrix_ii_ze(linidx_upper)==-Inf));
            end
        end
    end

    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3));

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
    DiscountedEV_under=beta*EVbase;
    DiscountedEV_hat  =beta0beta*EVbase;
    DiscountedEVinterp_under=permute(interp1(a1_grid,permute(DiscountedEV_under,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
    DiscountedEVinterp_hat  =permute(interp1(a1_grid,permute(DiscountedEV_hat,  [2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    if vfoptions.lowmemory==0
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
    entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_hat,N_d1,1,1,1,1,1,1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;

    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_inner=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
            entireRHS_inner=ReturnMatrix_inner+DiscountedEV_hat(d2aprimez);
            [~,maxindex_inner]=max(entireRHS_inner,[],2);
            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
        end
    end

    midpoint=max(min(midpoint,N_a1-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
    aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
    entireRHS_ii_hat  =reshape(ReturnMatrix_ii+DiscountedEVinterp_hat(aprimez),  [N_d*n2long*N_a2,N_a,N_z,N_e]);
    entireRHS_ii_under=reshape(ReturnMatrix_ii+DiscountedEVinterp_under(aprimez),[N_d*n2long*N_a2,N_a,N_z,N_e]);
    [Vtempii_hat,maxindexL2]=max(entireRHS_ii_hat,[],1);
    firstdim=N_d*n2long*N_a2;
    maxindexfull=maxindexL2 + firstdim*aind + firstdim*N_a*zindB + firstdim*N_a*N_z*eindB;
    Vtempii_under=entireRHS_ii_under(maxindexfull);
    Vhat(:,:,:,N_j)      =shiftdim(Vtempii_hat,1);
    Vunderbar(:,:,:,N_j) =shiftdim(Vtempii_under,1);

    d_ind        =rem(maxindexL2-1,N_d)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

    allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
    Policy(1,:,:,:,N_j)=d_ind;
    Policy(2,:,:,:,N_j)=midpoint(allind);
    Policy(3,:,:,:,N_j)=maxindexL2a2;
    Policy(4,:,:,:,N_j)=maxindexL2a1;

    RM_flat=reshape(ReturnMatrix_ii,[N_d*n2long*N_a2,N_a,N_z,N_e]);
    linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
    linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
    isInfLower=(RM_flat(linidx_lower)==-Inf);
    isInfUpper=(RM_flat(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    PolicyL2flag(1,:,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    elseif vfoptions.lowmemory==1
        error('lowmem=1 for QH+ExpAssetz S DC2A_GI2A V_Jplus1 init not yet implemented; use lowmem=0 for terminal')
    elseif vfoptions.lowmemory==2
        error('lowmem=2 for QH+ExpAssetz S DC2A_GI2A V_Jplus1 init not yet implemented; use lowmem=0 for terminal')
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

    EVpre=squeeze(sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3));

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
    DiscountedEV_under=beta*EVbase;
    DiscountedEV_hat  =beta0beta*EVbase;
    DiscountedEVinterp_under=permute(interp1(a1_grid,permute(DiscountedEV_under,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
    DiscountedEVinterp_hat  =permute(interp1(a1_grid,permute(DiscountedEV_hat,  [2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    if vfoptions.lowmemory==0
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
    entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_hat,N_d1,1,1,1,1,1,1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;

    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_inner=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
            entireRHS_inner=ReturnMatrix_inner+DiscountedEV_hat(d2aprimez);
            [~,maxindex_inner]=max(entireRHS_inner,[],2);
            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
        end
    end

    midpoint=max(min(midpoint,N_a1-1),2);
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
    aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
    entireRHS_ii_hat  =reshape(ReturnMatrix_ii+DiscountedEVinterp_hat(aprimez),  [N_d*n2long*N_a2,N_a,N_z,N_e]);
    entireRHS_ii_under=reshape(ReturnMatrix_ii+DiscountedEVinterp_under(aprimez),[N_d*n2long*N_a2,N_a,N_z,N_e]);
    [Vtempii_hat,maxindexL2]=max(entireRHS_ii_hat,[],1);
    firstdim=N_d*n2long*N_a2;
    maxindexfull=maxindexL2 + firstdim*aind + firstdim*N_a*zindB + firstdim*N_a*N_z*eindB;
    Vtempii_under=entireRHS_ii_under(maxindexfull);
    Vhat(:,:,:,jj)      =shiftdim(Vtempii_hat,1);
    Vunderbar(:,:,:,jj) =shiftdim(Vtempii_under,1);

    d_ind        =rem(maxindexL2-1,N_d)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

    allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB + N_d*N_a2*N_a*N_z*eindB;
    Policy(1,:,:,:,jj)=d_ind;
    Policy(2,:,:,:,jj)=midpoint(allind);
    Policy(3,:,:,:,jj)=maxindexL2a2;
    Policy(4,:,:,:,jj)=maxindexL2a1;

    RM_flat=reshape(ReturnMatrix_ii,[N_d*n2long*N_a2,N_a,N_z,N_e]);
    linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
    linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB + N_d*n2long*N_a2*N_a*N_z*eindB;
    isInfLower=(RM_flat(linidx_lower)==-Inf);
    isInfUpper=(RM_flat(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    PolicyL2flag(1,:,:,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_hat_z         =DiscountedEV_hat        (:,:,:,:,:,:,z_c);
            DiscountedEV_under_z       =DiscountedEV_under      (:,:,:,:,:,:,z_c);
            DiscountedEVinterp_hat_z   =DiscountedEVinterp_hat  (:,:,:,:,:,:,z_c);
            DiscountedEVinterp_under_z =DiscountedEVinterp_under(:,:,:,:,:,:,z_c);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);

            entireRHS_hat_z=ReturnMatrix_ii_z+repelem(DiscountedEV_hat_z,N_d1,1,1,1,1,1);
            [~,maxindex1_z]=max(entireRHS_hat_z,[],2);
            midpoint_z(:,1,:,level1ii,:,:,1,:)=maxindex1_z;
            maxgap=squeeze(max(max(max(max(max( maxindex1_z(:,1,:,2:end,:,:,1,:)-maxindex1_z(:,1,:,1:end-1,:,:,1,:), [],8),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1_z(:,1,:,ii,:,:,1,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_inner_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprime_z=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_inner_z=ReturnMatrix_inner_z+DiscountedEV_hat_z(d2aprime_z);
                    [~,maxindex_inner]=max(entireRHS_inner_z,[],2);
                    midpoint_z(:,1,:,curra1inner,:,:,1,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1_z(:,1,:,ii,:,:,1,:);
                    midpoint_z(:,1,:,curra1inner,:,:,1,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end

            midpoint_z=max(min(midpoint_z,N_a1-1),2);
            a1primeindexesfine=(midpoint_z+(midpoint_z-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez_z=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            entireRHS_ii_hat_z  =reshape(ReturnMatrix_ii_z+DiscountedEVinterp_hat_z  (aprimez_z),[N_d*n2long*N_a2,N_a,1,N_e]);
            entireRHS_ii_under_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_under_z(aprimez_z),[N_d*n2long*N_a2,N_a,1,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_hat_z,[],1);
            Vhat(:,z_c,:,jj)=shiftdim(Vtempii,1);
            firstdim=N_d*n2long*N_a2;
            maxindexfull_z=maxindexL2 + firstdim*aind + firstdim*N_a*eindB;
            Vunderbar(:,z_c,:,jj)=shiftdim(entireRHS_ii_under_z(maxindexfull_z),1);
            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;
            allind_z=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*eindB;
            Policy(1,:,z_c,:,jj)=d_ind;
            Policy(2,:,z_c,:,jj)=midpoint_z(allind_z);
            Policy(3,:,z_c,:,jj)=maxindexL2a2;
            Policy(4,:,z_c,:,jj)=maxindexL2a1;
            RM_flat=reshape(ReturnMatrix_ii_z,[N_d*n2long*N_a2,N_a,1,N_e]);
            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*eindB;
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,:,jj)=2 + (inLowerStrict & (RM_flat(linidx_lower)==-Inf)) - (inUpperStrict & (RM_flat(linidx_upper)==-Inf));
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_hat_ze         =DiscountedEV_hat        (:,:,:,:,:,:,z_c);
            DiscountedEV_under_ze       =DiscountedEV_under      (:,:,:,:,:,:,z_c);
            DiscountedEVinterp_hat_ze   =DiscountedEVinterp_hat  (:,:,:,:,:,:,z_c);
            DiscountedEVinterp_under_ze =DiscountedEVinterp_under(:,:,:,:,:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);

                entireRHS_hat_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_hat_ze,N_d1,1,1,1,1,1);
                [~,maxindex1_ze]=max(entireRHS_hat_ze,[],2);
                midpoint_ze(:,1,:,level1ii,:,:)=maxindex1_ze;
                maxgap=squeeze(max(max(max(max( maxindex1_ze(:,1,:,2:end,:,:)-maxindex1_ze(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1_ze(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_inner_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                        d2aprime_ze=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_inner_ze=ReturnMatrix_inner_ze+DiscountedEV_hat_ze(d2aprime_ze);
                        [~,maxindex_inner]=max(entireRHS_inner_ze,[],2);
                        midpoint_ze(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1_ze(:,1,:,ii,:,:);
                        midpoint_ze(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end

                midpoint_ze=max(min(midpoint_ze,N_a1-1),2);
                a1primeindexesfine=(midpoint_ze+(midpoint_ze-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                aprimez_ze=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_hat_ze  =reshape(ReturnMatrix_ii_ze+DiscountedEVinterp_hat_ze  (aprimez_ze),[N_d*n2long*N_a2,N_a]);
                entireRHS_ii_under_ze=reshape(ReturnMatrix_ii_ze+DiscountedEVinterp_under_ze(aprimez_ze),[N_d*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_hat_ze,[],1);
                Vhat(:,z_c,e_c,jj)=Vtempii(:);
                firstdim=N_d*n2long*N_a2;
                maxindexfull_ze=maxindexL2 + firstdim*aind;
                Vunderbar(:,z_c,e_c,jj)=entireRHS_ii_under_ze(maxindexfull_ze);
                d_ind        =rem(maxindexL2-1,N_d)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;
                allind_ze=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
                Policy(1,:,z_c,e_c,jj)=d_ind;
                Policy(2,:,z_c,e_c,jj)=midpoint_ze(allind_ze);
                Policy(3,:,z_c,e_c,jj)=maxindexL2a2;
                Policy(4,:,z_c,e_c,jj)=maxindexL2a1;
                RM_flat=reshape(ReturnMatrix_ii_ze,[N_d*n2long*N_a2,N_a]);
                linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                PolicyL2flag(1,:,z_c,e_c,jj)=2 + (inLowerStrict & (RM_flat(linidx_lower)==-Inf)) - (inUpperStrict & (RM_flat(linidx_upper)==-Inf));
            end
        end
    end
end


%% Post-process
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

end
