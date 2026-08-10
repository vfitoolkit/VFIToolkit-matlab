function [V,Policy]=ValueFnIter_FHorz_ExpAssetzSemiExo_DC2A_GI2A_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% semiz analog of ValueFnIter_FHorz_ExpAssetz_DC2A_GI2A_e_raw: d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is first standard endo state (DC+GI), a2 is folded remaining standard endo states, a3 is experience asset
% aprimeFn = aprimeFn(d2, a3, z, ...)   (depends on current markov z only, never e)
% lowmemory=0 full; lowmemory=1 loop e; lowmemory=2 loop z (markov) + e, vectorize semiz; lowmemory=3 loop bothz + e

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
d2ind=repelem(gpuArray(1:1:N_d2)',N_d1,1);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

% a3 gridvals column for the ReturnFn builder (experience asset is single-dim in the z family)
a3_gridvals=CreateGridvals(n_a3,a3_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray'); % (d12, d3, midpoint, a2prime, L2ind)
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

if vfoptions.lowmemory==0
    midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_bothz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_bothz,1,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==3
    midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
end

V_ford3_jj=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_jj=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray'); % (d12, midpoint, a2prime, L2ind)
flag_ford3_jj=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray'); % L2 flag per d3, aggregated after d3 max

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
eBind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_jj(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_jj(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_jj(4,:,:,:,d3_c)=maxindexL2a1;
            linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(1,:,:,:,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        [~,maxindex_inner]=max(ReturnMatrix_ii_e,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_jj(4,:,:,e_c,d3_c)=maxindexL2a1;
                linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_e(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(1,:,:,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                    midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            [~,maxindex_inner]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_jj(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,semizblock,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,semizblock,e_c,d3_c)=maxindexL2a1;
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            [~,maxindex_inner]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_jj(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_jj(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_jj(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_jj(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_jj(flat_idx),[1,N_a,N_bothz,N_e]);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z] (lower-corner index and its interpolation probability)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);

    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1); % [N_d2*N_a1*N_a2,N_a3,N_z]
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz); % [N_d2*N_a1*N_a2,N_a3,N_bothz] (semiz fastest within bothz)
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(repmat(a3primeProbs,N_a1*N_a2,1,1),1,1,N_semiz);
    bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_jj(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_jj(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_jj(4,:,:,:,d3_c)=maxindexL2a1;
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_d3,[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(1,:,:,:,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_e,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_e,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEVinterp(aprime),[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_jj(4,:,:,e_c,d3_c)=maxindexL2a1;
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_e,[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(1,:,:,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                DiscountedEV_zb=DiscountedEV(:,:,:,:,:,:,semizblock);
                DiscountedEVinterp_zb=DiscountedEVinterp(:,:,:,:,:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_zb,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                    midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_zb(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii_ze,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEVinterp_zb(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_jj(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,semizblock,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,semizblock,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_ze,[N_d12*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_z,N_d1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii_ze,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEVinterp_z(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_ze,[N_d12*n2long*N_a2,N_a]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_jj(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_jj(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_jj(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_jj(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_jj(flat_idx),[1,N_a,N_bothz,N_e]);
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

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z] (lower-corner index and its interpolation probability)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);

    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1); % [N_d2*N_a1*N_a2,N_a3,N_z]
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz); % [N_d2*N_a1*N_a2,N_a3,N_bothz] (semiz fastest within bothz)
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(repmat(a3primeProbs,N_a1*N_a2,1,1),1,1,N_semiz);
    bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_jj(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_jj(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_jj(4,:,:,:,d3_c)=maxindexL2a1;
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_d3,[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(1,:,:,:,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_e,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_e,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEVinterp(aprime),[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_jj(4,:,:,e_c,d3_c)=maxindexL2a1;
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_e,[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(1,:,:,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                DiscountedEV_zb=DiscountedEV(:,:,:,:,:,:,semizblock);
                DiscountedEVinterp_zb=DiscountedEVinterp(:,:,:,:,:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_zb,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                    midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_zb(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii_ze,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEVinterp_zb(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_jj(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,semizblock,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,semizblock,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_ze,[N_d12*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_z,N_d1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii_ze,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEVinterp_z(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_ze,[N_d12*n2long*N_a2,N_a]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_jj(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,jj)=reshape(Policy4_ford3_jj(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_jj(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_jj(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_jj(flat_idx),[1,N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];


end
