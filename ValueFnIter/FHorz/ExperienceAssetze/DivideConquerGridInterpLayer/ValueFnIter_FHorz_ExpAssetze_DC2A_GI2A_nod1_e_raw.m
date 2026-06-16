function [V,Policy]=ValueFnIter_FHorz_ExpAssetze_DC2A_GI2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% _e analog of ValueFnIter_FHorz_ExpAssetze_DC2A_GI2A_nod1_raw.
% lowmemory=0 full; lowmemory=1 loop z (e vectorized); lowmemory=2 nested z+e.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);
N_e=prod(n_e);

% Per-dim factored a3 grid for the ReturnFn builder (l_a3==1: 1 column, l_a3==2: 2 columns)
a3_gridvals=CreateGridvals(n_a3,a3_grid,1);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray');

aind=gpuArray(0:1:N_a-1);
if vfoptions.lowmemory==0
    zindB=shiftdim(gpuArray(0:1:N_z-1),-1);
    eindB=shiftdim(gpuArray(0:1:N_e-1),-2);
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_z,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
    eindB=shiftdim(gpuArray(0:1:N_e-1),-2);
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,1,N_e,'gpuArray');
else
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
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
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
            end
        end
        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind        =rem(maxindexL2-1,N_d2)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
        allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*zindB + N_d2*N_a2*N_a*N_z*eindB;
        Policy(1,:,:,:,N_j)=d_ind;
        Policy(2,:,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,:,N_j)=maxindexL2a1;
        linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB + N_d2*n2long*N_a2*N_a*N_z*eindB;
        linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB + N_d2*n2long*N_a2*N_a*N_z*eindB;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii_z,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            V(:,z_c,:,N_j)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*eindB;
            Policy(1,:,z_c,:,N_j)=d_ind;
            Policy(2,:,z_c,:,N_j)=midpoint(allind);
            Policy(3,:,z_c,:,N_j)=maxindexL2a2;
            Policy(4,:,z_c,:,N_j)=maxindexL2a1;
            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
            isInfLower=(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                        [~,maxindex_inner]=max(ReturnMatrix_ii_ze,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy(1,:,z_c,e_c,N_j)=d_ind;
                Policy(2,:,z_c,e_c,N_j)=midpoint(allind);
                Policy(3,:,z_c,e_c,N_j)=maxindexL2a2;
                Policy(4,:,z_c,e_c,N_j)=maxindexL2a1;
                linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end
    end

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3));

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a3, n_z, n_e, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % l_a3==1: a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z,N_e] (legacy lower-corner)
    % l_a3==2: a3primeIndex/a3primeProbs are [l_a3,N_d2,N_a3,N_z,N_e] (per-dim factored)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);

    if length(n_a3)==1
        a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1,1);
        aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
        aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
        aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,1,N_z);

        Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    else
        % l_a3==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a3_1=n_a3(1);
        loIdx_1_repd=repmat(reshape(a3primeIndex(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        loIdx_2_repd=repmat(reshape(a3primeIndex(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        prob_1_exp=repmat(reshape(a3primeProbs(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1,N_z);
        prob_2_exp=repmat(reshape(a3primeProbs(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1,N_z);

        a3_kron_ll= loIdx_1_repd   +n_a3_1*(loIdx_2_repd-1);
        a3_kron_hl=(loIdx_1_repd+1)+n_a3_1*(loIdx_2_repd-1);
        a3_kron_lh= loIdx_1_repd   +n_a3_1* loIdx_2_repd;
        a3_kron_hh=(loIdx_1_repd+1)+n_a3_1* loIdx_2_repd;

        aprime_ll=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_ll-1);
        aprime_hl=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hl-1);
        aprime_lh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_lh-1);
        aprime_hh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hh-1);

        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    EV=EV.*reshape(pi_z_J(:,:,N_j),[1,1,N_z,1,N_z]);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,5));

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z,N_e]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d2aprimez);
                [~,maxindex_inner]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
            end
        end
        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        aprimez=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind        =rem(maxindexL2-1,N_d2)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
        allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*zindB + N_d2*N_a2*N_a*N_z*eindB;
        Policy(1,:,:,:,N_j)=d_ind;
        Policy(2,:,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,:,N_j)=maxindexL2a1;
        ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d2*n2long*N_a2,N_a,N_z,N_e]);
        linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB + N_d2*n2long*N_a2*N_a*N_z*eindB;
        linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB + N_d2*n2long*N_a2*N_a*N_z*eindB;
        isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c,:);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c,:);

            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z;
            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z(d2aprime);
                    [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
            entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprime),[N_d2*n2long*N_a2,N_a,1,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            V(:,z_c,:,N_j)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*eindB;
            Policy(1,:,z_c,:,N_j)=d_ind;
            Policy(2,:,z_c,:,N_j)=midpoint(allind);
            Policy(3,:,z_c,:,N_j)=maxindexL2a2;
            Policy(4,:,z_c,:,N_j)=maxindexL2a1;
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_z,[N_d2*n2long*N_a2,N_a,1,N_e]);
            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,z_c,e_c);
                DiscountedEVinterp_ze=DiscountedEVinterp(:,:,:,:,:,:,z_c,e_c);

                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_ze;
                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_ze(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_ze,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEVinterp_ze(aprime),[N_d2*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy(1,:,z_c,e_c,N_j)=d_ind;
                Policy(2,:,z_c,e_c,N_j)=midpoint(allind);
                Policy(3,:,z_c,e_c,N_j)=maxindexL2a2;
                Policy(4,:,z_c,e_c,N_j)=maxindexL2a1;
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_ze,[N_d2*n2long*N_a2,N_a]);
                linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3));

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a3, n_z, n_e, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % l_a3==1: a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z,N_e] (legacy lower-corner)
    % l_a3==2: a3primeIndex/a3primeProbs are [l_a3,N_d2,N_a3,N_z,N_e] (per-dim factored)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);

    if length(n_a3)==1
        a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1,1);
        aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
        aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
        aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,1,N_z);

        Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    else
        % l_a3==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a3_1=n_a3(1);
        loIdx_1_repd=repmat(reshape(a3primeIndex(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        loIdx_2_repd=repmat(reshape(a3primeIndex(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        prob_1_exp=repmat(reshape(a3primeProbs(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1,N_z);
        prob_2_exp=repmat(reshape(a3primeProbs(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1,N_z);

        a3_kron_ll= loIdx_1_repd   +n_a3_1*(loIdx_2_repd-1);
        a3_kron_hl=(loIdx_1_repd+1)+n_a3_1*(loIdx_2_repd-1);
        a3_kron_lh= loIdx_1_repd   +n_a3_1* loIdx_2_repd;
        a3_kron_hh=(loIdx_1_repd+1)+n_a3_1* loIdx_2_repd;

        aprime_ll=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_ll-1);
        aprime_hl=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hl-1);
        aprime_lh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_lh-1);
        aprime_hh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hh-1);

        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    EV=EV.*reshape(pi_z_J(:,:,jj),[1,1,N_z,1,N_z]);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,5));

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z,N_e]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d2aprimez);
                [~,maxindex_inner]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
            end
        end
        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimez=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,jj)=shiftdim(Vtempii,1);
        d_ind        =rem(maxindexL2-1,N_d2)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
        allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*zindB + N_d2*N_a2*N_a*N_z*eindB;
        Policy(1,:,:,:,jj)=d_ind;
        Policy(2,:,:,:,jj)=midpoint(allind);
        Policy(3,:,:,:,jj)=maxindexL2a2;
        Policy(4,:,:,:,jj)=maxindexL2a1;
        ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d2*n2long*N_a2,N_a,N_z,N_e]);
        linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB + N_d2*n2long*N_a2*N_a*N_z*eindB;
        linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*zindB + N_d2*n2long*N_a2*N_a*N_z*eindB;
        isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c,:);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c,:);

            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z;
            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z(d2aprime);
                    [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
            entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprime),[N_d2*n2long*N_a2,N_a,1,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            V(:,z_c,:,jj)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*eindB;
            Policy(1,:,z_c,:,jj)=d_ind;
            Policy(2,:,z_c,:,jj)=midpoint(allind);
            Policy(3,:,z_c,:,jj)=maxindexL2a2;
            Policy(4,:,z_c,:,jj)=maxindexL2a1;
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_z,[N_d2*n2long*N_a2,N_a,1,N_e]);
            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*eindB;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,z_c,e_c);
                DiscountedEVinterp_ze=DiscountedEVinterp(:,:,:,:,:,:,z_c,e_c);

                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_ze;
                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_ze(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_ze,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEVinterp_ze(aprime),[N_d2*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                V(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy(1,:,z_c,e_c,jj)=d_ind;
                Policy(2,:,z_c,e_c,jj)=midpoint(allind);
                Policy(3,:,z_c,e_c,jj)=maxindexL2a2;
                Policy(4,:,z_c,e_c,jj)=maxindexL2a1;
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_ze,[N_d2*n2long*N_a2,N_a]);
                linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                PolicyL2flag(1,:,z_c,e_c,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end
    end
end


%% Post-process
adjust=(Policy(4,:,:,:,:)<1+n2short+1);
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust;
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

end
