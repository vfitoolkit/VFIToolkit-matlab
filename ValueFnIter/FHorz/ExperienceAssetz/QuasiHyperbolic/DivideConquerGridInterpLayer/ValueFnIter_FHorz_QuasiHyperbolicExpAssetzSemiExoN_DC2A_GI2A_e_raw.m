function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoN_DC2A_GI2A_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
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

Valt=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policyalt=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray'); % (d12, d3, midpoint, a2prime, L2ind)
PolicyL2flagalt=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

if vfoptions.lowmemory==0
    midpoint_alt=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_bothz,N_e,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_bothz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint_alt=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_bothz,1,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_bothz,1,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint_alt=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==3
    midpoint_alt=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
end

V_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_alt=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_tilde=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray'); % (d12, midpoint, a2prime, L2ind)
flag_ford3_alt=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray');
flag_ford3_tilde=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray'); % L2 flag per d3, aggregated after d3 max

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
            midpoint_alt(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                    midpoint_alt(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint_alt(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
            a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_alt(2,:,:,:,d3_c)=midpoint_alt(allind);
            Policy4_ford3_alt(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1;
            linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
                midpoint_alt(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        [~,maxindex_inner]=max(ReturnMatrix_ii_e,[],2);
                        midpoint_alt(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint_alt(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
                a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=midpoint_alt(allind);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1;
                linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_e(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
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
                    midpoint_alt(:,1,:,level1ii,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            [~,maxindex_inner]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint_alt(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint_alt(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
                    a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=midpoint_alt(allind);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=maxindexL2a1;
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
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
                    midpoint_alt(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            [~,maxindex_inner]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint_alt(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint_alt(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
                    a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=midpoint_alt(allind);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policyalt(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(3,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flagalt(1,:,:,:,N_j)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz,N_e]);
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy(:,:,:,:,N_j)=Policyalt(:,:,:,:,N_j);
    PolicyL2flag(:,:,:,:,N_j)=PolicyL2flagalt(:,:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

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

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);


            % --- alt pass ---
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
            [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
            midpoint_alt(:,1,:,level1ii,:,:,:,:)=maxindex1_alt;
            maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_alt(ii)>0
                    loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                    a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                    ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez_alt=d2ind + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt);
                    [~,maxindex_inner_alt]=max(entireRHS_ii_d3_alt,[],2);
                    midpoint_alt(:,1,:,curra1inner,:,:,:,:)=maxindex_inner_alt+(loweredge_alt-1);
                else
                    loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                    midpoint_alt(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge_alt,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
            a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_d3_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEVinterp_alt(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii_alt,1);
            d_ind_alt        =rem(maxindexL2_alt-1,N_d12)+1;
            maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d12),n2long)+1;
            maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d12*n2long))+1;
            allind_alt=d_ind_alt + N_d12*(maxindexL2a2_alt-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=d_ind_alt;
            Policy4_ford3_alt(2,:,:,:,d3_c)=midpoint_alt(allind_alt);
            Policy4_ford3_alt(3,:,:,:,d3_c)=maxindexL2a2_alt;
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1_alt;
            ReturnMatrix_ii_flat_alt=reshape(ReturnMatrix_ii_d3_alt,[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower_alt=d_ind_alt                + N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper_alt=d_ind_alt + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower_alt=(ReturnMatrix_ii_flat_alt(linidx_lower_alt)==-Inf);
            isInfUpper_alt=(ReturnMatrix_ii_flat_alt(linidx_upper_alt)==-Inf);
            inLowerStrict_alt=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
            inUpperStrict_alt=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c)=2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt);

            % --- tilde pass ---
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1,1);
            [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
            midpoint_tilde(:,1,:,level1ii,:,:,:,:)=maxindex1_tilde;
            maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_tilde(ii)>0
                    loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                    a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                    ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez_tilde=d2ind + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                    [~,maxindex_inner_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                    midpoint_tilde(:,1,:,curra1inner,:,:,:,:)=maxindex_inner_tilde+(loweredge_tilde-1);
                else
                    loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                    midpoint_tilde(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge_tilde,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint_tilde=max(min(midpoint_tilde,N_a1-1),2);
            a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind + N_d2*(a1primeindexesfine_tilde-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_d3_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEVinterp_tilde(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii_tilde,1);
            d_ind_tilde        =rem(maxindexL2_tilde-1,N_d12)+1;
            maxindexL2a1_tilde =rem(floor((maxindexL2_tilde-1)/N_d12),n2long)+1;
            maxindexL2a2_tilde =floor((maxindexL2_tilde-1)/(N_d12*n2long))+1;
            allind_tilde=d_ind_tilde + N_d12*(maxindexL2a2_tilde-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=d_ind_tilde;
            Policy4_ford3_tilde(2,:,:,:,d3_c)=midpoint_tilde(allind_tilde);
            Policy4_ford3_tilde(3,:,:,:,d3_c)=maxindexL2a2_tilde;
            Policy4_ford3_tilde(4,:,:,:,d3_c)=maxindexL2a1_tilde;
            ReturnMatrix_ii_flat_tilde=reshape(ReturnMatrix_ii_d3_tilde,[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower_tilde=d_ind_tilde                + N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper_tilde=d_ind_tilde + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower_tilde=(ReturnMatrix_ii_flat_tilde(linidx_lower_tilde)==-Inf);
            isInfUpper_tilde=(ReturnMatrix_ii_flat_tilde(linidx_upper_tilde)==-Inf);
            inLowerStrict_tilde=(maxindexL2a1_tilde>=2)         & (maxindexL2a1_tilde<=n2short+1);
            inUpperStrict_tilde=(maxindexL2a1_tilde>=n2short+3) & (maxindexL2a1_tilde<=n2long-1);
            flag_ford3_tilde(1,:,:,:,d3_c)=2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde);
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

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

            % --- alt pass ---
                ReturnMatrix_ii_e_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_e_alt=ReturnMatrix_ii_e_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
                [~,maxindex1_alt]=max(entireRHS_ii_e_alt,[],2);
                midpoint_alt(:,1,:,level1ii,:,:,:,:)=maxindex1_alt;
                maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_alt(ii)>0
                        loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                        a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                        ReturnMatrix_ii_e_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprime_alt=d2ind + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_e_alt=ReturnMatrix_ii_e_alt+DiscountedEV_alt(d2aprime_alt);
                        [~,maxindex_inner_alt]=max(entireRHS_ii_e_alt,[],2);
                        midpoint_alt(:,1,:,curra1inner,:,:,:,:)=maxindex_inner_alt+(loweredge_alt-1);
                    else
                        loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                        midpoint_alt(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge_alt,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_e_alt=reshape(ReturnMatrix_ii_e_alt+DiscountedEVinterp_alt(aprime),[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_e_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt        =rem(maxindexL2_alt-1,N_d12)+1;
                maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d12),n2long)+1;
                maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d12*n2long))+1;
                allind_alt=d_ind_alt + N_d12*(maxindexL2a2_alt-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=d_ind_alt;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=midpoint_alt(allind_alt);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=maxindexL2a2_alt;
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1_alt;
                ReturnMatrix_ii_flat_alt=reshape(ReturnMatrix_ii_e_alt,[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                linidx_lower_alt=d_ind_alt                + N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper_alt=d_ind_alt + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower_alt=(ReturnMatrix_ii_flat_alt(linidx_lower_alt)==-Inf);
                isInfUpper_alt=(ReturnMatrix_ii_flat_alt(linidx_upper_alt)==-Inf);
                inLowerStrict_alt=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
                inUpperStrict_alt=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c)=2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt);

            % --- tilde pass ---
                ReturnMatrix_ii_e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_e_tilde=ReturnMatrix_ii_e_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1,1);
                [~,maxindex1_tilde]=max(entireRHS_ii_e_tilde,[],2);
                midpoint_tilde(:,1,:,level1ii,:,:,:,:)=maxindex1_tilde;
                maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_tilde(ii)>0
                        loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                        a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                        ReturnMatrix_ii_e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprime_tilde=d2ind + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_e_tilde=ReturnMatrix_ii_e_tilde+DiscountedEV_tilde(d2aprime_tilde);
                        [~,maxindex_inner_tilde]=max(entireRHS_ii_e_tilde,[],2);
                        midpoint_tilde(:,1,:,curra1inner,:,:,:,:)=maxindex_inner_tilde+(loweredge_tilde-1);
                    else
                        loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                        midpoint_tilde(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge_tilde,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint_tilde=max(min(midpoint_tilde,N_a1-1),2);
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind + N_d2*(a1primeindexesfine_tilde-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_e_tilde=reshape(ReturnMatrix_ii_e_tilde+DiscountedEVinterp_tilde(aprime),[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_e_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde        =rem(maxindexL2_tilde-1,N_d12)+1;
                maxindexL2a1_tilde =rem(floor((maxindexL2_tilde-1)/N_d12),n2long)+1;
                maxindexL2a2_tilde =floor((maxindexL2_tilde-1)/(N_d12*n2long))+1;
                allind_tilde=d_ind_tilde + N_d12*(maxindexL2a2_tilde-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=d_ind_tilde;
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=midpoint_tilde(allind_tilde);
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=maxindexL2a2_tilde;
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=maxindexL2a1_tilde;
                ReturnMatrix_ii_flat_tilde=reshape(ReturnMatrix_ii_e_tilde,[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                linidx_lower_tilde=d_ind_tilde                + N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper_tilde=d_ind_tilde + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower_tilde=(ReturnMatrix_ii_flat_tilde(linidx_lower_tilde)==-Inf);
                isInfUpper_tilde=(ReturnMatrix_ii_flat_tilde(linidx_upper_tilde)==-Inf);
                inLowerStrict_tilde=(maxindexL2a1_tilde>=2)         & (maxindexL2a1_tilde<=n2short+1);
                inUpperStrict_tilde=(maxindexL2a1_tilde>=n2short+3) & (maxindexL2a1_tilde<=n2long-1);
                flag_ford3_tilde(1,:,:,e_c,d3_c)=2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde);
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

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

            % --- alt pass ---
                DiscountedEV_zb_alt=DiscountedEV_alt(:,:,:,:,:,:,semizblock);
                DiscountedEVinterp_zb_alt=DiscountedEVinterp_alt(:,:,:,:,:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze_alt=ReturnMatrix_ii_ze_alt+repelem(DiscountedEV_zb_alt,N_d1,1,1,1,1,1,1);
                    [~,maxindex1_alt]=max(entireRHS_ii_ze_alt,[],2);
                    midpoint_alt(:,1,:,level1ii,:,:,:)=maxindex1_alt;
                    maxgap_alt=squeeze(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d2aprime_alt=d2ind + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_ze_alt=ReturnMatrix_ii_ze_alt+DiscountedEV_zb_alt(d2aprime_alt);
                            [~,maxindex_inner_alt]=max(entireRHS_ii_ze_alt,[],2);
                            midpoint_alt(:,1,:,curra1inner,:,:,:)=maxindex_inner_alt+(loweredge_alt-1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:);
                            midpoint_alt(:,1,:,curra1inner,:,:,:)=repelem(loweredge_alt,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
                    a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_ze_alt=reshape(ReturnMatrix_ii_ze_alt+DiscountedEVinterp_zb_alt(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_ze_alt,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    d_ind_alt        =rem(maxindexL2_alt-1,N_d12)+1;
                    maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d12),n2long)+1;
                    maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d12*n2long))+1;
                    allind_alt=d_ind_alt + N_d12*(maxindexL2a2_alt-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=d_ind_alt;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=midpoint_alt(allind_alt);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=maxindexL2a2_alt;
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=maxindexL2a1_alt;
                    ReturnMatrix_ii_flat_alt=reshape(ReturnMatrix_ii_ze_alt,[N_d12*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower_alt=d_ind_alt                + N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper_alt=d_ind_alt + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower_alt=(ReturnMatrix_ii_flat_alt(linidx_lower_alt)==-Inf);
                    isInfUpper_alt=(ReturnMatrix_ii_flat_alt(linidx_upper_alt)==-Inf);
                    inLowerStrict_alt=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
                    inUpperStrict_alt=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
                    flag_ford3_alt(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt);
                end

            % --- tilde pass ---
                DiscountedEV_zb_tilde=DiscountedEV_tilde(:,:,:,:,:,:,semizblock);
                DiscountedEVinterp_zb_tilde=DiscountedEVinterp_tilde(:,:,:,:,:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze_tilde=ReturnMatrix_ii_ze_tilde+repelem(DiscountedEV_zb_tilde,N_d1,1,1,1,1,1,1);
                    [~,maxindex1_tilde]=max(entireRHS_ii_ze_tilde,[],2);
                    midpoint_tilde(:,1,:,level1ii,:,:,:)=maxindex1_tilde;
                    maxgap_tilde=squeeze(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d2aprime_tilde=d2ind + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_ze_tilde=ReturnMatrix_ii_ze_tilde+DiscountedEV_zb_tilde(d2aprime_tilde);
                            [~,maxindex_inner_tilde]=max(entireRHS_ii_ze_tilde,[],2);
                            midpoint_tilde(:,1,:,curra1inner,:,:,:)=maxindex_inner_tilde+(loweredge_tilde-1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:);
                            midpoint_tilde(:,1,:,curra1inner,:,:,:)=repelem(loweredge_tilde,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint_tilde=max(min(midpoint_tilde,N_a1-1),2);
                    a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine_tilde-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_ze_tilde=reshape(ReturnMatrix_ii_ze_tilde+DiscountedEVinterp_zb_tilde(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_ze_tilde,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    d_ind_tilde        =rem(maxindexL2_tilde-1,N_d12)+1;
                    maxindexL2a1_tilde =rem(floor((maxindexL2_tilde-1)/N_d12),n2long)+1;
                    maxindexL2a2_tilde =floor((maxindexL2_tilde-1)/(N_d12*n2long))+1;
                    allind_tilde=d_ind_tilde + N_d12*(maxindexL2a2_tilde-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,semizblock,e_c,d3_c)=d_ind_tilde;
                    Policy4_ford3_tilde(2,:,semizblock,e_c,d3_c)=midpoint_tilde(allind_tilde);
                    Policy4_ford3_tilde(3,:,semizblock,e_c,d3_c)=maxindexL2a2_tilde;
                    Policy4_ford3_tilde(4,:,semizblock,e_c,d3_c)=maxindexL2a1_tilde;
                    ReturnMatrix_ii_flat_tilde=reshape(ReturnMatrix_ii_ze_tilde,[N_d12*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower_tilde=d_ind_tilde                + N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper_tilde=d_ind_tilde + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower_tilde=(ReturnMatrix_ii_flat_tilde(linidx_lower_tilde)==-Inf);
                    isInfUpper_tilde=(ReturnMatrix_ii_flat_tilde(linidx_upper_tilde)==-Inf);
                    inLowerStrict_tilde=(maxindexL2a1_tilde>=2)         & (maxindexL2a1_tilde<=n2short+1);
                    inUpperStrict_tilde=(maxindexL2a1_tilde>=n2short+3) & (maxindexL2a1_tilde<=n2long-1);
                    flag_ford3_tilde(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde);
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

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

            % --- alt pass ---
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z_alt=DiscountedEVinterp_alt(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze_alt=ReturnMatrix_ii_ze_alt+repelem(DiscountedEV_z_alt,N_d1,1,1,1,1,1);
                    [~,maxindex1_alt]=max(entireRHS_ii_ze_alt,[],2);
                    midpoint_alt(:,1,:,level1ii,:,:)=maxindex1_alt;
                    maxgap_alt=squeeze(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_alt=d2ind + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_ze_alt=ReturnMatrix_ii_ze_alt+DiscountedEV_z_alt(d2aprime_alt);
                            [~,maxindex_inner_alt]=max(entireRHS_ii_ze_alt,[],2);
                            midpoint_alt(:,1,:,curra1inner,:,:)=maxindex_inner_alt+(loweredge_alt-1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,:,ii,:,:);
                            midpoint_alt(:,1,:,curra1inner,:,:)=repelem(loweredge_alt,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
                    a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_ze_alt=reshape(ReturnMatrix_ii_ze_alt+DiscountedEVinterp_z_alt(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_ze_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    d_ind_alt        =rem(maxindexL2_alt-1,N_d12)+1;
                    maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d12),n2long)+1;
                    maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d12*n2long))+1;
                    allind_alt=d_ind_alt + N_d12*(maxindexL2a2_alt-1) + N_d12*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=d_ind_alt;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=midpoint_alt(allind_alt);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=maxindexL2a2_alt;
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1_alt;
                    ReturnMatrix_ii_flat_alt=reshape(ReturnMatrix_ii_ze_alt,[N_d12*n2long*N_a2,N_a]);
                    linidx_lower_alt=d_ind_alt                + N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper_alt=d_ind_alt + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind;
                    isInfLower_alt=(ReturnMatrix_ii_flat_alt(linidx_lower_alt)==-Inf);
                    isInfUpper_alt=(ReturnMatrix_ii_flat_alt(linidx_upper_alt)==-Inf);
                    inLowerStrict_alt=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
                    inUpperStrict_alt=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt);
                end

            % --- tilde pass ---
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z_tilde=DiscountedEVinterp_tilde(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze_tilde=ReturnMatrix_ii_ze_tilde+repelem(DiscountedEV_z_tilde,N_d1,1,1,1,1,1);
                    [~,maxindex1_tilde]=max(entireRHS_ii_ze_tilde,[],2);
                    midpoint_tilde(:,1,:,level1ii,:,:)=maxindex1_tilde;
                    maxgap_tilde=squeeze(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_tilde=d2ind + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_ze_tilde=ReturnMatrix_ii_ze_tilde+DiscountedEV_z_tilde(d2aprime_tilde);
                            [~,maxindex_inner_tilde]=max(entireRHS_ii_ze_tilde,[],2);
                            midpoint_tilde(:,1,:,curra1inner,:,:)=maxindex_inner_tilde+(loweredge_tilde-1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:);
                            midpoint_tilde(:,1,:,curra1inner,:,:)=repelem(loweredge_tilde,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint_tilde=max(min(midpoint_tilde,N_a1-1),2);
                    a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine_tilde-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_ze_tilde=reshape(ReturnMatrix_ii_ze_tilde+DiscountedEVinterp_z_tilde(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_ze_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    d_ind_tilde        =rem(maxindexL2_tilde-1,N_d12)+1;
                    maxindexL2a1_tilde =rem(floor((maxindexL2_tilde-1)/N_d12),n2long)+1;
                    maxindexL2a2_tilde =floor((maxindexL2_tilde-1)/(N_d12*n2long))+1;
                    allind_tilde=d_ind_tilde + N_d12*(maxindexL2a2_tilde-1) + N_d12*N_a2*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=d_ind_tilde;
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=midpoint_tilde(allind_tilde);
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=maxindexL2a2_tilde;
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=maxindexL2a1_tilde;
                    ReturnMatrix_ii_flat_tilde=reshape(ReturnMatrix_ii_ze_tilde,[N_d12*n2long*N_a2,N_a]);
                    linidx_lower_tilde=d_ind_tilde                + N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper_tilde=d_ind_tilde + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind;
                    isInfLower_tilde=(ReturnMatrix_ii_flat_tilde(linidx_lower_tilde)==-Inf);
                    isInfUpper_tilde=(ReturnMatrix_ii_flat_tilde(linidx_upper_tilde)==-Inf);
                    inLowerStrict_tilde=(maxindexL2a1_tilde>=2)         & (maxindexL2a1_tilde<=n2short+1);
                    inUpperStrict_tilde=(maxindexL2a1_tilde>=n2short+3) & (maxindexL2a1_tilde<=n2long-1);
                    flag_ford3_tilde(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde);
                end
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policyalt(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(3,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flagalt(1,:,:,:,N_j)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz,N_e]);

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    Vtilde(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_bothz,N_e]);

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

    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

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

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);


            % --- alt pass ---
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
            [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
            midpoint_alt(:,1,:,level1ii,:,:,:,:)=maxindex1_alt;
            maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_alt(ii)>0
                    loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                    a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                    ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez_alt=d2ind + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt);
                    [~,maxindex_inner_alt]=max(entireRHS_ii_d3_alt,[],2);
                    midpoint_alt(:,1,:,curra1inner,:,:,:,:)=maxindex_inner_alt+(loweredge_alt-1);
                else
                    loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                    midpoint_alt(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge_alt,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
            a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_d3_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEVinterp_alt(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii_alt,1);
            d_ind_alt        =rem(maxindexL2_alt-1,N_d12)+1;
            maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d12),n2long)+1;
            maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d12*n2long))+1;
            allind_alt=d_ind_alt + N_d12*(maxindexL2a2_alt-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=d_ind_alt;
            Policy4_ford3_alt(2,:,:,:,d3_c)=midpoint_alt(allind_alt);
            Policy4_ford3_alt(3,:,:,:,d3_c)=maxindexL2a2_alt;
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1_alt;
            ReturnMatrix_ii_flat_alt=reshape(ReturnMatrix_ii_d3_alt,[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower_alt=d_ind_alt                + N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper_alt=d_ind_alt + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower_alt=(ReturnMatrix_ii_flat_alt(linidx_lower_alt)==-Inf);
            isInfUpper_alt=(ReturnMatrix_ii_flat_alt(linidx_upper_alt)==-Inf);
            inLowerStrict_alt=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
            inUpperStrict_alt=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c)=2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt);

            % --- tilde pass ---
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1,1);
            [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
            midpoint_tilde(:,1,:,level1ii,:,:,:,:)=maxindex1_tilde;
            maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_tilde(ii)>0
                    loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                    a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                    ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez_tilde=d2ind + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                    [~,maxindex_inner_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                    midpoint_tilde(:,1,:,curra1inner,:,:,:,:)=maxindex_inner_tilde+(loweredge_tilde-1);
                else
                    loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                    midpoint_tilde(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge_tilde,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint_tilde=max(min(midpoint_tilde,N_a1-1),2);
            a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind + N_d2*(a1primeindexesfine_tilde-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_d3_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEVinterp_tilde(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii_tilde,1);
            d_ind_tilde        =rem(maxindexL2_tilde-1,N_d12)+1;
            maxindexL2a1_tilde =rem(floor((maxindexL2_tilde-1)/N_d12),n2long)+1;
            maxindexL2a2_tilde =floor((maxindexL2_tilde-1)/(N_d12*n2long))+1;
            allind_tilde=d_ind_tilde + N_d12*(maxindexL2a2_tilde-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=d_ind_tilde;
            Policy4_ford3_tilde(2,:,:,:,d3_c)=midpoint_tilde(allind_tilde);
            Policy4_ford3_tilde(3,:,:,:,d3_c)=maxindexL2a2_tilde;
            Policy4_ford3_tilde(4,:,:,:,d3_c)=maxindexL2a1_tilde;
            ReturnMatrix_ii_flat_tilde=reshape(ReturnMatrix_ii_d3_tilde,[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower_tilde=d_ind_tilde                + N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper_tilde=d_ind_tilde + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower_tilde=(ReturnMatrix_ii_flat_tilde(linidx_lower_tilde)==-Inf);
            isInfUpper_tilde=(ReturnMatrix_ii_flat_tilde(linidx_upper_tilde)==-Inf);
            inLowerStrict_tilde=(maxindexL2a1_tilde>=2)         & (maxindexL2a1_tilde<=n2short+1);
            inUpperStrict_tilde=(maxindexL2a1_tilde>=n2short+3) & (maxindexL2a1_tilde<=n2long-1);
            flag_ford3_tilde(1,:,:,:,d3_c)=2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde);
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

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

            % --- alt pass ---
                ReturnMatrix_ii_e_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_e_alt=ReturnMatrix_ii_e_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
                [~,maxindex1_alt]=max(entireRHS_ii_e_alt,[],2);
                midpoint_alt(:,1,:,level1ii,:,:,:,:)=maxindex1_alt;
                maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_alt(ii)>0
                        loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                        a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                        ReturnMatrix_ii_e_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprime_alt=d2ind + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_e_alt=ReturnMatrix_ii_e_alt+DiscountedEV_alt(d2aprime_alt);
                        [~,maxindex_inner_alt]=max(entireRHS_ii_e_alt,[],2);
                        midpoint_alt(:,1,:,curra1inner,:,:,:,:)=maxindex_inner_alt+(loweredge_alt-1);
                    else
                        loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                        midpoint_alt(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge_alt,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_e_alt=reshape(ReturnMatrix_ii_e_alt+DiscountedEVinterp_alt(aprime),[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_e_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt        =rem(maxindexL2_alt-1,N_d12)+1;
                maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d12),n2long)+1;
                maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d12*n2long))+1;
                allind_alt=d_ind_alt + N_d12*(maxindexL2a2_alt-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=d_ind_alt;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=midpoint_alt(allind_alt);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=maxindexL2a2_alt;
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1_alt;
                ReturnMatrix_ii_flat_alt=reshape(ReturnMatrix_ii_e_alt,[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                linidx_lower_alt=d_ind_alt                + N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper_alt=d_ind_alt + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower_alt=(ReturnMatrix_ii_flat_alt(linidx_lower_alt)==-Inf);
                isInfUpper_alt=(ReturnMatrix_ii_flat_alt(linidx_upper_alt)==-Inf);
                inLowerStrict_alt=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
                inUpperStrict_alt=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c)=2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt);

            % --- tilde pass ---
                ReturnMatrix_ii_e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_e_tilde=ReturnMatrix_ii_e_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1,1);
                [~,maxindex1_tilde]=max(entireRHS_ii_e_tilde,[],2);
                midpoint_tilde(:,1,:,level1ii,:,:,:,:)=maxindex1_tilde;
                maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_tilde(ii)>0
                        loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                        a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                        ReturnMatrix_ii_e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprime_tilde=d2ind + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_e_tilde=ReturnMatrix_ii_e_tilde+DiscountedEV_tilde(d2aprime_tilde);
                        [~,maxindex_inner_tilde]=max(entireRHS_ii_e_tilde,[],2);
                        midpoint_tilde(:,1,:,curra1inner,:,:,:,:)=maxindex_inner_tilde+(loweredge_tilde-1);
                    else
                        loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                        midpoint_tilde(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge_tilde,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint_tilde=max(min(midpoint_tilde,N_a1-1),2);
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind + N_d2*(a1primeindexesfine_tilde-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_e_tilde=reshape(ReturnMatrix_ii_e_tilde+DiscountedEVinterp_tilde(aprime),[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_e_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde        =rem(maxindexL2_tilde-1,N_d12)+1;
                maxindexL2a1_tilde =rem(floor((maxindexL2_tilde-1)/N_d12),n2long)+1;
                maxindexL2a2_tilde =floor((maxindexL2_tilde-1)/(N_d12*n2long))+1;
                allind_tilde=d_ind_tilde + N_d12*(maxindexL2a2_tilde-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=d_ind_tilde;
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=midpoint_tilde(allind_tilde);
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=maxindexL2a2_tilde;
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=maxindexL2a1_tilde;
                ReturnMatrix_ii_flat_tilde=reshape(ReturnMatrix_ii_e_tilde,[N_d12*n2long*N_a2,N_a,N_bothz,1]);
                linidx_lower_tilde=d_ind_tilde                + N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper_tilde=d_ind_tilde + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower_tilde=(ReturnMatrix_ii_flat_tilde(linidx_lower_tilde)==-Inf);
                isInfUpper_tilde=(ReturnMatrix_ii_flat_tilde(linidx_upper_tilde)==-Inf);
                inLowerStrict_tilde=(maxindexL2a1_tilde>=2)         & (maxindexL2a1_tilde<=n2short+1);
                inUpperStrict_tilde=(maxindexL2a1_tilde>=n2short+3) & (maxindexL2a1_tilde<=n2long-1);
                flag_ford3_tilde(1,:,:,e_c,d3_c)=2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde);
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

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

            % --- alt pass ---
                DiscountedEV_zb_alt=DiscountedEV_alt(:,:,:,:,:,:,semizblock);
                DiscountedEVinterp_zb_alt=DiscountedEVinterp_alt(:,:,:,:,:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze_alt=ReturnMatrix_ii_ze_alt+repelem(DiscountedEV_zb_alt,N_d1,1,1,1,1,1,1);
                    [~,maxindex1_alt]=max(entireRHS_ii_ze_alt,[],2);
                    midpoint_alt(:,1,:,level1ii,:,:,:)=maxindex1_alt;
                    maxgap_alt=squeeze(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d2aprime_alt=d2ind + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_ze_alt=ReturnMatrix_ii_ze_alt+DiscountedEV_zb_alt(d2aprime_alt);
                            [~,maxindex_inner_alt]=max(entireRHS_ii_ze_alt,[],2);
                            midpoint_alt(:,1,:,curra1inner,:,:,:)=maxindex_inner_alt+(loweredge_alt-1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:);
                            midpoint_alt(:,1,:,curra1inner,:,:,:)=repelem(loweredge_alt,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
                    a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_ze_alt=reshape(ReturnMatrix_ii_ze_alt+DiscountedEVinterp_zb_alt(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_ze_alt,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    d_ind_alt        =rem(maxindexL2_alt-1,N_d12)+1;
                    maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d12),n2long)+1;
                    maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d12*n2long))+1;
                    allind_alt=d_ind_alt + N_d12*(maxindexL2a2_alt-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=d_ind_alt;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=midpoint_alt(allind_alt);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=maxindexL2a2_alt;
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=maxindexL2a1_alt;
                    ReturnMatrix_ii_flat_alt=reshape(ReturnMatrix_ii_ze_alt,[N_d12*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower_alt=d_ind_alt                + N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper_alt=d_ind_alt + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower_alt=(ReturnMatrix_ii_flat_alt(linidx_lower_alt)==-Inf);
                    isInfUpper_alt=(ReturnMatrix_ii_flat_alt(linidx_upper_alt)==-Inf);
                    inLowerStrict_alt=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
                    inUpperStrict_alt=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
                    flag_ford3_alt(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt);
                end

            % --- tilde pass ---
                DiscountedEV_zb_tilde=DiscountedEV_tilde(:,:,:,:,:,:,semizblock);
                DiscountedEVinterp_zb_tilde=DiscountedEVinterp_tilde(:,:,:,:,:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze_tilde=ReturnMatrix_ii_ze_tilde+repelem(DiscountedEV_zb_tilde,N_d1,1,1,1,1,1,1);
                    [~,maxindex1_tilde]=max(entireRHS_ii_ze_tilde,[],2);
                    midpoint_tilde(:,1,:,level1ii,:,:,:)=maxindex1_tilde;
                    maxgap_tilde=squeeze(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d2aprime_tilde=d2ind + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_ze_tilde=ReturnMatrix_ii_ze_tilde+DiscountedEV_zb_tilde(d2aprime_tilde);
                            [~,maxindex_inner_tilde]=max(entireRHS_ii_ze_tilde,[],2);
                            midpoint_tilde(:,1,:,curra1inner,:,:,:)=maxindex_inner_tilde+(loweredge_tilde-1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:);
                            midpoint_tilde(:,1,:,curra1inner,:,:,:)=repelem(loweredge_tilde,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint_tilde=max(min(midpoint_tilde,N_a1-1),2);
                    a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine_tilde-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_ze_tilde=reshape(ReturnMatrix_ii_ze_tilde+DiscountedEVinterp_zb_tilde(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_ze_tilde,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    d_ind_tilde        =rem(maxindexL2_tilde-1,N_d12)+1;
                    maxindexL2a1_tilde =rem(floor((maxindexL2_tilde-1)/N_d12),n2long)+1;
                    maxindexL2a2_tilde =floor((maxindexL2_tilde-1)/(N_d12*n2long))+1;
                    allind_tilde=d_ind_tilde + N_d12*(maxindexL2a2_tilde-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,semizblock,e_c,d3_c)=d_ind_tilde;
                    Policy4_ford3_tilde(2,:,semizblock,e_c,d3_c)=midpoint_tilde(allind_tilde);
                    Policy4_ford3_tilde(3,:,semizblock,e_c,d3_c)=maxindexL2a2_tilde;
                    Policy4_ford3_tilde(4,:,semizblock,e_c,d3_c)=maxindexL2a1_tilde;
                    ReturnMatrix_ii_flat_tilde=reshape(ReturnMatrix_ii_ze_tilde,[N_d12*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower_tilde=d_ind_tilde                + N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper_tilde=d_ind_tilde + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower_tilde=(ReturnMatrix_ii_flat_tilde(linidx_lower_tilde)==-Inf);
                    isInfUpper_tilde=(ReturnMatrix_ii_flat_tilde(linidx_upper_tilde)==-Inf);
                    inLowerStrict_tilde=(maxindexL2a1_tilde>=2)         & (maxindexL2a1_tilde<=n2short+1);
                    inUpperStrict_tilde=(maxindexL2a1_tilde>=n2short+3) & (maxindexL2a1_tilde<=n2long-1);
                    flag_ford3_tilde(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde);
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

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

            % --- alt pass ---
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z_alt=DiscountedEVinterp_alt(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze_alt=ReturnMatrix_ii_ze_alt+repelem(DiscountedEV_z_alt,N_d1,1,1,1,1,1);
                    [~,maxindex1_alt]=max(entireRHS_ii_ze_alt,[],2);
                    midpoint_alt(:,1,:,level1ii,:,:)=maxindex1_alt;
                    maxgap_alt=squeeze(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_alt=d2ind + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_ze_alt=ReturnMatrix_ii_ze_alt+DiscountedEV_z_alt(d2aprime_alt);
                            [~,maxindex_inner_alt]=max(entireRHS_ii_ze_alt,[],2);
                            midpoint_alt(:,1,:,curra1inner,:,:)=maxindex_inner_alt+(loweredge_alt-1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,:,ii,:,:);
                            midpoint_alt(:,1,:,curra1inner,:,:)=repelem(loweredge_alt,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint_alt=max(min(midpoint_alt,N_a1-1),2);
                    a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine_alt-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_ze_alt=reshape(ReturnMatrix_ii_ze_alt+DiscountedEVinterp_z_alt(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_ze_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    d_ind_alt        =rem(maxindexL2_alt-1,N_d12)+1;
                    maxindexL2a1_alt =rem(floor((maxindexL2_alt-1)/N_d12),n2long)+1;
                    maxindexL2a2_alt =floor((maxindexL2_alt-1)/(N_d12*n2long))+1;
                    allind_alt=d_ind_alt + N_d12*(maxindexL2a2_alt-1) + N_d12*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=d_ind_alt;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=midpoint_alt(allind_alt);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=maxindexL2a2_alt;
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1_alt;
                    ReturnMatrix_ii_flat_alt=reshape(ReturnMatrix_ii_ze_alt,[N_d12*n2long*N_a2,N_a]);
                    linidx_lower_alt=d_ind_alt                + N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper_alt=d_ind_alt + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_alt-1) + N_d12*n2long*N_a2*aind;
                    isInfLower_alt=(ReturnMatrix_ii_flat_alt(linidx_lower_alt)==-Inf);
                    isInfUpper_alt=(ReturnMatrix_ii_flat_alt(linidx_upper_alt)==-Inf);
                    inLowerStrict_alt=(maxindexL2a1_alt>=2)         & (maxindexL2a1_alt<=n2short+1);
                    inUpperStrict_alt=(maxindexL2a1_alt>=n2short+3) & (maxindexL2a1_alt<=n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt);
                end

            % --- tilde pass ---
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z_tilde=DiscountedEVinterp_tilde(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_ze_tilde=ReturnMatrix_ii_ze_tilde+repelem(DiscountedEV_z_tilde,N_d1,1,1,1,1,1);
                    [~,maxindex1_tilde]=max(entireRHS_ii_ze_tilde,[],2);
                    midpoint_tilde(:,1,:,level1ii,:,:)=maxindex1_tilde;
                    maxgap_tilde=squeeze(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_tilde=d2ind + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_ze_tilde=ReturnMatrix_ii_ze_tilde+DiscountedEV_z_tilde(d2aprime_tilde);
                            [~,maxindex_inner_tilde]=max(entireRHS_ii_ze_tilde,[],2);
                            midpoint_tilde(:,1,:,curra1inner,:,:)=maxindex_inner_tilde+(loweredge_tilde-1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:);
                            midpoint_tilde(:,1,:,curra1inner,:,:)=repelem(loweredge_tilde,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint_tilde=max(min(midpoint_tilde,N_a1-1),2);
                    a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind + N_d2*(a1primeindexesfine_tilde-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_ze_tilde=reshape(ReturnMatrix_ii_ze_tilde+DiscountedEVinterp_z_tilde(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_ze_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    d_ind_tilde        =rem(maxindexL2_tilde-1,N_d12)+1;
                    maxindexL2a1_tilde =rem(floor((maxindexL2_tilde-1)/N_d12),n2long)+1;
                    maxindexL2a2_tilde =floor((maxindexL2_tilde-1)/(N_d12*n2long))+1;
                    allind_tilde=d_ind_tilde + N_d12*(maxindexL2a2_tilde-1) + N_d12*N_a2*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=d_ind_tilde;
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=midpoint_tilde(allind_tilde);
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=maxindexL2a2_tilde;
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=maxindexL2a1_tilde;
                    ReturnMatrix_ii_flat_tilde=reshape(ReturnMatrix_ii_ze_tilde,[N_d12*n2long*N_a2,N_a]);
                    linidx_lower_tilde=d_ind_tilde                + N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper_tilde=d_ind_tilde + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2_tilde-1) + N_d12*n2long*N_a2*aind;
                    isInfLower_tilde=(ReturnMatrix_ii_flat_tilde(linidx_lower_tilde)==-Inf);
                    isInfUpper_tilde=(ReturnMatrix_ii_flat_tilde(linidx_upper_tilde)==-Inf);
                    inLowerStrict_tilde=(maxindexL2a1_tilde>=2)         & (maxindexL2a1_tilde<=n2short+1);
                    inUpperStrict_tilde=(maxindexL2a1_tilde>=n2short+3) & (maxindexL2a1_tilde<=n2long-1);
                    flag_ford3_tilde(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde);
                end
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,jj)=V_jj;
    Policyalt(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policyalt(1,:,:,:,jj)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(3,:,:,:,jj)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,jj)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,jj)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flagalt(1,:,:,:,jj)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz,N_e]);

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    Vtilde(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,jj)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_bothz,N_e]);

end


%% Switch from midpoint to lower grid index
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

adjustalt=(Policyalt(5,:,:,:,:)<1+n2short+1);
Policyalt(3,:,:,:,:)=Policyalt(3,:,:,:,:)-adjustalt;
Policyalt(5,:,:,:,:)=adjustalt.*Policyalt(5,:,:,:,:)+(1-adjustalt).*(Policyalt(5,:,:,:,:)-n2short-1);

Policyalt=[Policyalt; PolicyL2flagalt];


end
