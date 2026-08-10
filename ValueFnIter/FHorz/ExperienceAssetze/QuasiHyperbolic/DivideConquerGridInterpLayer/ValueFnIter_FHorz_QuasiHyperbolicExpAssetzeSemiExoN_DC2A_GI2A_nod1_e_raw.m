function [V,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExoN_DC2A_GI2A_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic + ExperienceAssetze + SemiExo + DC2A + GI2A (no d1).
% d2 determines experience asset, d3 determines semi-exog state (no d1)
% a1 is first standard endo state (DC+GI), a2 is folded remaining standard endo states, a3 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a3, z, e, ...)   (depends on BOTH current z and current e)
%
% Naive QH dual pass. Both value functions are computed with the FULL divide-conquer + grid-interp
% machinery (each with its own midpoint localization over the fine grid):
%   Valt/Policyalt maximise  F + beta*EV        (the exponential value)
%   V/Policy       maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Valt (the exponential continuation value).
% Policy rows: (d2, d3, midpoint, a2prime, L2ind) + L2flag appended; Policyalt likewise.
%
% lowmemory levels {0,1,2,3} implemented (shocks: z markov + semiz + e iid).

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

% Per-dim factored a3 grid for the ReturnFn builder (l_a3==1: 1 column, l_a3==2: 2 columns)
a3_gridvals=CreateGridvals(n_a3,a3_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray'); % (d2, d3, midpoint, a2prime, L2ind)
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper
Valt=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policyalt=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray');
PolicyL2flagalt=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray');

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

if vfoptions.lowmemory==0
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_bothz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_bothz,1,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==3
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
end

% Per-d3 arrays (alt=exponential [F+beta*EV], tilde=QH-perceived [F+beta0*beta*EV])
V_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_alt=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray'); % (d2, midpoint, a2prime, L2ind)
flag_ford3_alt=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_tilde=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray');
flag_ford3_tilde=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray');

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
    % Terminal period has no continuation, so QH-perceived and exponential coincide: single pass
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_alt(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_alt(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1;
            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        [~,maxindex_inner]=max(ReturnMatrix_ii_e,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1;
                linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_e(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                    midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            [~,maxindex_inner]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=maxindexL2a1;
                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
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
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            [~,maxindex_inner]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end
    end

    % Max over d3; terminal period has no continuation, so QH-perceived and exponential coincide
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz,N_e]);
    Valt(:,:,:,N_j)=V(:,:,:,N_j);
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j);
    PolicyL2flagalt(1,:,:,:,N_j)=PolicyL2flag(1,:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

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
        aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz,1); % [N_d2*N_a1*N_a2,N_a3,N_bothz,N_e]
        aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz,1);
        aprimeProbs_full=repelem(repmat(a3primeProbs,N_a1*N_a2,1,1,1),1,1,N_semiz,1);
    else
        % l_a3==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a3_1=n_a3(1);
        loIdx_1_repd=repmat(reshape(a3primeIndex(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        loIdx_2_repd=repmat(reshape(a3primeIndex(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        prob_1_full=repelem(repmat(reshape(a3primeProbs(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1);
        prob_2_full=repelem(repmat(reshape(a3primeProbs(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1);

        a3_kron_ll= loIdx_1_repd   +n_a3_1*(loIdx_2_repd-1);
        a3_kron_hl=(loIdx_1_repd+1)+n_a3_1*(loIdx_2_repd-1);
        a3_kron_lh= loIdx_1_repd   +n_a3_1* loIdx_2_repd;
        a3_kron_hh=(loIdx_1_repd+1)+n_a3_1* loIdx_2_repd;

        aprime_ll_full=repelem(a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_ll-1),1,1,N_semiz,1);
        aprime_hl_full=repelem(a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hl-1),1,1,N_semiz,1);
        aprime_lh_full=repelem(a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_lh-1),1,1,N_semiz,1);
        aprime_hh_full=repelem(a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hh-1),1,1,N_semiz,1);
    end
    bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a3)==1
                EV1=EV_2D(aprimeIndex_full+bothz_offset);
                EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;
                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            else
                V_ll=EV_2D(aprime_ll_full+bothz_offset);
                V_hl=EV_2D(aprime_hl_full+bothz_offset);
                V_lh=EV_2D(aprime_lh_full+bothz_offset);
                V_hh=EV_2D(aprime_hh_full+bothz_offset);
                p1_loy=prob_1_full; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_full; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_full; p2(EV_loy==EV_hiy)=0;
                c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                entireEV=c_loy+c_hiy;
            end

            % Undiscounted EV; beta vs beta0beta applied at use (Naive QH dual pass)
            EVreshape=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_grid,permute(EVreshape,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

            %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
            entireRHS_ii=ReturnMatrix_ii_d3+beta*EVreshape;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=ReturnMatrix_ii+beta*EVreshape(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
            entireRHS_L2=reshape(ReturnMatrix_L2+beta*EVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_alt(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_alt(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1;
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            %% Vtilde (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
            entireRHS_ii=ReturnMatrix_ii_d3+beta0beta*EVreshape;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
            entireRHS_L2=reshape(ReturnMatrix_L2+beta0beta*EVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_tilde(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_tilde(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_tilde(4,:,:,:,d3_c)=maxindexL2a1;
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_tilde(1,:,:,:,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a3)==1
                EV1=EV_2D(aprimeIndex_full+bothz_offset);
                EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;
                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            else
                V_ll=EV_2D(aprime_ll_full+bothz_offset);
                V_hl=EV_2D(aprime_hl_full+bothz_offset);
                V_lh=EV_2D(aprime_lh_full+bothz_offset);
                V_hh=EV_2D(aprime_hh_full+bothz_offset);
                p1_loy=prob_1_full; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_full; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_full; p2(EV_loy==EV_hiy)=0;
                c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                entireEV=c_loy+c_hiy;
            end

            EVreshape=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_grid,permute(EVreshape,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                EVreshape_e=EVreshape(:,:,:,:,:,:,:,e_c);
                EVinterp_e=EVinterp(:,:,:,:,:,:,:,e_c);

                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);

                %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
                entireRHS_ii=ReturnMatrix_ii_d3+beta*EVreshape_e;
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii=ReturnMatrix_ii+beta*EVreshape_e(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_L2=reshape(ReturnMatrix_L2+beta*EVinterp_e(aprime),[N_d2*n2long*N_a2,N_a,N_bothz,1]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1;
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_bothz,1]);
                linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                %% Vtilde (QH-perceived): F + beta0beta*EV, full divide-conquer + grid-interp
                entireRHS_ii=ReturnMatrix_ii_d3+beta0beta*EVreshape_e;
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_e(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_L2=reshape(ReturnMatrix_L2+beta0beta*EVinterp_e(aprime),[N_d2*n2long*N_a2,N_a,N_bothz,1]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=maxindexL2a1;
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_bothz,1]);
                linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_tilde(1,:,:,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a3)==1
                EV1=EV_2D(aprimeIndex_full+bothz_offset);
                EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;
                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            else
                V_ll=EV_2D(aprime_ll_full+bothz_offset);
                V_hl=EV_2D(aprime_hl_full+bothz_offset);
                V_lh=EV_2D(aprime_lh_full+bothz_offset);
                V_hh=EV_2D(aprime_hh_full+bothz_offset);
                p1_loy=prob_1_full; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_full; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_full; p2(EV_loy==EV_hiy)=0;
                c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                entireEV=c_loy+c_hiy;
            end

            EVreshape=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_grid,permute(EVreshape,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                EVreshape_zb=EVreshape(:,:,:,:,:,:,semizblock,:);
                EVinterp_zb=EVinterp(:,:,:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    EVreshape_zbe=EVreshape_zb(:,:,:,:,:,:,:,e_c);
                    EVinterp_zbe=EVinterp_zb(:,:,:,:,:,:,:,e_c);

                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 1);

                    %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+beta*EVreshape_zbe;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii=ReturnMatrix_ii+beta*EVreshape_zbe(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_L2=reshape(ReturnMatrix_L2+beta*EVinterp_zbe(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    %% Vtilde (QH-perceived): F + beta0beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+beta0beta*EVreshape_zbe;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_zbe(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_L2=reshape(ReturnMatrix_L2+beta0beta*EVinterp_zbe(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_tilde(2,:,semizblock,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_tilde(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_tilde(4,:,semizblock,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a3)==1
                EV1=EV_2D(aprimeIndex_full+bothz_offset);
                EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;
                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            else
                V_ll=EV_2D(aprime_ll_full+bothz_offset);
                V_hl=EV_2D(aprime_hl_full+bothz_offset);
                V_lh=EV_2D(aprime_lh_full+bothz_offset);
                V_hh=EV_2D(aprime_hh_full+bothz_offset);
                p1_loy=prob_1_full; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_full; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_full; p2(EV_loy==EV_hiy)=0;
                c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                entireEV=c_loy+c_hiy;
            end

            EVreshape=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_grid,permute(EVreshape,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    EVreshape_ze=EVreshape(:,:,:,:,:,:,z_c,e_c);
                    EVinterp_ze=EVinterp(:,:,:,:,:,:,z_c,e_c);

                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);

                    %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+beta*EVreshape_ze;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=ReturnMatrix_ii+beta*EVreshape_ze(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_L2=reshape(ReturnMatrix_L2+beta*EVinterp_ze(aprime),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a]);
                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    %% Vtilde (QH-perceived): F + beta0beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+beta0beta*EVreshape_ze;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_ze(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_L2=reshape(ReturnMatrix_L2+beta0beta*EVinterp_ze(aprime),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a]);
                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end
    end

    % Max over d3 for tilde (QH-perceived) -> V/Policy
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_bothz,N_e]);

    % Max over d3 for alt (exponential) -> Valt/Policyalt
    [Valt_jj,maxindexalt]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=Valt_jj;
    Policyalt(2,:,:,:,N_j)=shiftdim(maxindexalt,-1);
    maxindexalt=reshape(maxindexalt,[N_a*N_bothz*N_e,1]);
    tempalt=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1)-1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+tempalt),[1,N_a,N_bothz,N_e]);
    Policyalt(3,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+tempalt),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+tempalt),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+tempalt),[1,N_a,N_bothz,N_e]);
    flat_idxalt=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1);
    PolicyL2flagalt(1,:,:,:,N_j)=reshape(flag_ford3_alt(flat_idxalt),[1,N_a,N_bothz,N_e]);
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

    % Continuation value is the exponential value (Valt), integrated over e'
    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

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
        aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz,1); % [N_d2*N_a1*N_a2,N_a3,N_bothz,N_e]
        aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz,1);
        aprimeProbs_full=repelem(repmat(a3primeProbs,N_a1*N_a2,1,1,1),1,1,N_semiz,1);
    else
        % l_a3==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a3_1=n_a3(1);
        loIdx_1_repd=repmat(reshape(a3primeIndex(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        loIdx_2_repd=repmat(reshape(a3primeIndex(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        prob_1_full=repelem(repmat(reshape(a3primeProbs(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1);
        prob_2_full=repelem(repmat(reshape(a3primeProbs(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1);

        a3_kron_ll= loIdx_1_repd   +n_a3_1*(loIdx_2_repd-1);
        a3_kron_hl=(loIdx_1_repd+1)+n_a3_1*(loIdx_2_repd-1);
        a3_kron_lh= loIdx_1_repd   +n_a3_1* loIdx_2_repd;
        a3_kron_hh=(loIdx_1_repd+1)+n_a3_1* loIdx_2_repd;

        aprime_ll_full=repelem(a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_ll-1),1,1,N_semiz,1);
        aprime_hl_full=repelem(a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hl-1),1,1,N_semiz,1);
        aprime_lh_full=repelem(a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_lh-1),1,1,N_semiz,1);
        aprime_hh_full=repelem(a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hh-1),1,1,N_semiz,1);
    end
    bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a3)==1
                EV1=EV_2D(aprimeIndex_full+bothz_offset);
                EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;
                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            else
                V_ll=EV_2D(aprime_ll_full+bothz_offset);
                V_hl=EV_2D(aprime_hl_full+bothz_offset);
                V_lh=EV_2D(aprime_lh_full+bothz_offset);
                V_hh=EV_2D(aprime_hh_full+bothz_offset);
                p1_loy=prob_1_full; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_full; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_full; p2(EV_loy==EV_hiy)=0;
                c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                entireEV=c_loy+c_hiy;
            end

            % Undiscounted EV; beta vs beta0beta applied at use (Naive QH dual pass)
            EVreshape=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_grid,permute(EVreshape,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);

            %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
            entireRHS_ii=ReturnMatrix_ii_d3+beta*EVreshape;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=ReturnMatrix_ii+beta*EVreshape(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
            entireRHS_L2=reshape(ReturnMatrix_L2+beta*EVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_alt(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_alt(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1;
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            %% Vtilde (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
            entireRHS_ii=ReturnMatrix_ii_d3+beta0beta*EVreshape;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
            entireRHS_L2=reshape(ReturnMatrix_L2+beta0beta*EVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_tilde(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_tilde(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_tilde(4,:,:,:,d3_c)=maxindexL2a1;
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_tilde(1,:,:,:,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a3)==1
                EV1=EV_2D(aprimeIndex_full+bothz_offset);
                EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;
                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            else
                V_ll=EV_2D(aprime_ll_full+bothz_offset);
                V_hl=EV_2D(aprime_hl_full+bothz_offset);
                V_lh=EV_2D(aprime_lh_full+bothz_offset);
                V_hh=EV_2D(aprime_hh_full+bothz_offset);
                p1_loy=prob_1_full; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_full; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_full; p2(EV_loy==EV_hiy)=0;
                c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                entireEV=c_loy+c_hiy;
            end

            EVreshape=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_grid,permute(EVreshape,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                EVreshape_e=EVreshape(:,:,:,:,:,:,:,e_c);
                EVinterp_e=EVinterp(:,:,:,:,:,:,:,e_c);

                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);

                %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
                entireRHS_ii=ReturnMatrix_ii_d3+beta*EVreshape_e;
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii=ReturnMatrix_ii+beta*EVreshape_e(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_L2=reshape(ReturnMatrix_L2+beta*EVinterp_e(aprime),[N_d2*n2long*N_a2,N_a,N_bothz,1]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1;
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_bothz,1]);
                linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                %% Vtilde (QH-perceived): F + beta0beta*EV, full divide-conquer + grid-interp
                entireRHS_ii=ReturnMatrix_ii_d3+beta0beta*EVreshape_e;
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_e(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_L2=reshape(ReturnMatrix_L2+beta0beta*EVinterp_e(aprime),[N_d2*n2long*N_a2,N_a,N_bothz,1]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=maxindexL2a1;
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_bothz,1]);
                linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_tilde(1,:,:,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a3)==1
                EV1=EV_2D(aprimeIndex_full+bothz_offset);
                EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;
                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            else
                V_ll=EV_2D(aprime_ll_full+bothz_offset);
                V_hl=EV_2D(aprime_hl_full+bothz_offset);
                V_lh=EV_2D(aprime_lh_full+bothz_offset);
                V_hh=EV_2D(aprime_hh_full+bothz_offset);
                p1_loy=prob_1_full; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_full; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_full; p2(EV_loy==EV_hiy)=0;
                c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                entireEV=c_loy+c_hiy;
            end

            EVreshape=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_grid,permute(EVreshape,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                EVreshape_zb=EVreshape(:,:,:,:,:,:,semizblock,:);
                EVinterp_zb=EVinterp(:,:,:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    EVreshape_zbe=EVreshape_zb(:,:,:,:,:,:,:,e_c);
                    EVinterp_zbe=EVinterp_zb(:,:,:,:,:,:,:,e_c);

                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 1);

                    %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+beta*EVreshape_zbe;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii=ReturnMatrix_ii+beta*EVreshape_zbe(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_L2=reshape(ReturnMatrix_L2+beta*EVinterp_zbe(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    %% Vtilde (QH-perceived): F + beta0beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+beta0beta*EVreshape_zbe;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_zbe(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_L2=reshape(ReturnMatrix_L2+beta0beta*EVinterp_zbe(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_tilde(2,:,semizblock,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_tilde(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_tilde(4,:,semizblock,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(1,:,semizblock,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a3)==1
                EV1=EV_2D(aprimeIndex_full+bothz_offset);
                EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;
                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            else
                V_ll=EV_2D(aprime_ll_full+bothz_offset);
                V_hl=EV_2D(aprime_hl_full+bothz_offset);
                V_lh=EV_2D(aprime_lh_full+bothz_offset);
                V_hh=EV_2D(aprime_hh_full+bothz_offset);
                p1_loy=prob_1_full; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_full; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_full; p2(EV_loy==EV_hiy)=0;
                c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                entireEV=c_loy+c_hiy;
            end

            EVreshape=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_grid,permute(EVreshape,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    EVreshape_ze=EVreshape(:,:,:,:,:,:,z_c,e_c);
                    EVinterp_ze=EVinterp(:,:,:,:,:,:,z_c,e_c);

                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);

                    %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+beta*EVreshape_ze;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=ReturnMatrix_ii+beta*EVreshape_ze(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_L2=reshape(ReturnMatrix_L2+beta*EVinterp_ze(aprime),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a]);
                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    %% Vtilde (QH-perceived): F + beta0beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+beta0beta*EVreshape_ze;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_ze(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_L2=reshape(ReturnMatrix_L2+beta0beta*EVinterp_ze(aprime),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a]);
                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(1,:,z_c,e_c,d3_c)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
                end
            end
        end
    end

    % Max over d3 for tilde (QH-perceived) -> V/Policy
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    V(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,jj)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_bothz,N_e]);

    % Max over d3 for alt (exponential) -> Valt/Policyalt
    [Valt_jj,maxindexalt]=max(V_ford3_alt,[],4);
    Valt(:,:,:,jj)=Valt_jj;
    Policyalt(2,:,:,:,jj)=shiftdim(maxindexalt,-1);
    maxindexalt=reshape(maxindexalt,[N_a*N_bothz*N_e,1]);
    tempalt=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1)-1);
    Policyalt(1,:,:,:,jj)=reshape(Policy4_ford3_alt(1+tempalt),[1,N_a,N_bothz,N_e]);
    Policyalt(3,:,:,:,jj)=reshape(Policy4_ford3_alt(2+tempalt),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,jj)=reshape(Policy4_ford3_alt(3+tempalt),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,jj)=reshape(Policy4_ford3_alt(4+tempalt),[1,N_a,N_bothz,N_e]);
    flat_idxalt=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1);
    PolicyL2flagalt(1,:,:,:,jj)=reshape(flag_ford3_alt(flat_idxalt),[1,N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index (QH-perceived Policy)
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

%% Switch from midpoint to lower grid index (exponential Policyalt)
adjustalt=(Policyalt(5,:,:,:,:)<1+n2short+1);
Policyalt(3,:,:,:,:)=Policyalt(3,:,:,:,:)-adjustalt;
Policyalt(5,:,:,:,:)=adjustalt.*Policyalt(5,:,:,:,:)+(1-adjustalt).*(Policyalt(5,:,:,:,:)-n2short-1);

Policyalt=[Policyalt; PolicyL2flagalt];


end
