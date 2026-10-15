function [V,Policy]=ValueFnIter_FHorz_ExpAssetsemiz_DC1_GI1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state (no d1)
% a1 is standard endogenous state, a2 is experience asset (z-dependent aprimeFn)
% z is exogenous markov state (required), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, semiz, ...)
% DC + GI splice (no L2flag scaffold).

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray');
Policy(5,:,:,:,:)=2; % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
if vfoptions.lowmemory==0
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_bothz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_bothz,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==3
    midpoint=zeros(N_d2,1,N_a1,N_a2,'gpuArray');
end

V_ford3_jj=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy3_ford3_jj=zeros(3,N_a,N_bothz,N_e,N_d3,'gpuArray');
flag_ford3_jj=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray'); % L2 flag per d3, aggregated after d3 max

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-3); % already includes -1
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3); % already includes -1 (for lowmemory==2 split: outer z, inner e, vectorize semiz)
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1

bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_jj(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford3_jj(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_jj(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end

    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        [~,maxindex]=max(ReturnMatrix_ii_e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_jj(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3_jj(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d2);
                linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_jj(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer loop over z (markov), inner loop over e, vectorize over semiz

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);

                    midpoint(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            [~,maxindex]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind+N_d2*N_a*semizBind;
                    Policy3_ford3_jj(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy3_ford3_jj(2,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_jj(3,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_jj(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end

    elseif vfoptions.lowmemory==3

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);

                    midpoint(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            [~,maxindex]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_jj(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_jj(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                  + N_d2*n2long*aind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_jj(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=3*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    Policy(5,:,:,:,N_j)=reshape(flag_ford3_jj(flat_idx),[1,N_a,N_bothz,N_e]);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    if length(n_a2)==1
        aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
        aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
        aprimeProbs_d2a1a2semiz=repmat(a2primeProbs,N_a1,1,1);
        aprimeIndex_full=repmat(aprimeIndex,1,1,N_z);
        aprimeplus1Index_full=repmat(aprimeplus1Index,1,1,N_z);
        aprimeProbs_full=repmat(aprimeProbs_d2a1a2semiz,1,1,N_z);
    else
        % l_a2==2: the builder returns [l_a2,N_d2,N_a2,N_semiz] per-dim factored. Fold the two
        % per-dim lower indexes into the four corners of the a2 lattice, then push each corner
        % through THIS FILE'S OWN construction so the a1-Kron and any tiling over z are applied
        % exactly as the l_a2==1 arm applies them.
        n_a2_1=n_a2(1);
        lo_1=reshape(a2primeIndex(1,:,:,:),[N_d2,N_a2,N_semiz]);
        lo_2=reshape(a2primeIndex(2,:,:,:),[N_d2,N_a2,N_semiz]);
        p_1=reshape(a2primeProbs(1,:,:,:),[N_d2,N_a2,N_semiz]);
        p_2=reshape(a2primeProbs(2,:,:,:),[N_d2,N_a2,N_semiz]);
        a2k_ll=lo_1+n_a2_1*(lo_2-1);
        a2k_hl=(lo_1+1)+n_a2_1*(lo_2-1);
        a2k_lh=lo_1+n_a2_1*lo_2;
        a2k_hh=(lo_1+1)+n_a2_1*lo_2;
        aIdxK_ll=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_ll-1,N_a1,1,1);
        aIdxK_hl=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_hl-1,N_a1,1,1);
        aIdxK_lh=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_lh-1,N_a1,1,1);
        aIdxK_hh=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_hh-1,N_a1,1,1);
        prK_1=repmat(p_1,N_a1,1,1);
        prK_2=repmat(p_2,N_a1,1,1);
        aIdx_ll=repmat(aIdxK_ll,1,1,N_z);
        aIdx_hl=repmat(aIdxK_hl,1,1,N_z);
        aIdx_lh=repmat(aIdxK_lh,1,1,N_z);
        aIdx_hh=repmat(aIdxK_hh,1,1,N_z);
        pr1_full=repmat(prK_1,1,1,N_z);
        pr2_full=repmat(prK_2,1,1,N_z);
    end

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                k2_Vll=EV_2D(aIdx_ll+bothz_offset);
                k2_Vhl=EV_2D(aIdx_hl+bothz_offset);
                k2_Vlh=EV_2D(aIdx_lh+bothz_offset);
                k2_Vhh=EV_2D(aIdx_hh+bothz_offset);
                k2_p1=pr1_full; k2_p1(k2_Vll==k2_Vhl)=0;
                k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                k2_lo=k2_cll+k2_chl;
                k2_p1b=pr1_full; k2_p1b(k2_Vlh==k2_Vhh)=0;
                k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                k2_hi=k2_clh+k2_chh;
                k2_p2=pr2_full; k2_p2(k2_lo==k2_hi)=0;
                k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                entireEV=k2_clo+k2_chi;
            end

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV; % broadcasts over e

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]); % midpoint (hence gather) is e-dependent at lowmemory=0
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_jj(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford3_jj(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_ii_d3(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_d3(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_jj(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end

    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                k2_Vll=EV_2D(aIdx_ll+bothz_offset);
                k2_Vhl=EV_2D(aIdx_hl+bothz_offset);
                k2_Vlh=EV_2D(aIdx_lh+bothz_offset);
                k2_Vhh=EV_2D(aIdx_hh+bothz_offset);
                k2_p1=pr1_full; k2_p1(k2_Vll==k2_Vhl)=0;
                k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                k2_lo=k2_cll+k2_chl;
                k2_p1b=pr1_full; k2_p1b(k2_Vlh==k2_Vhh)=0;
                k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                k2_hi=k2_clh+k2_chh;
                k2_p2=pr2_full; k2_p2(k2_lo==k2_hi)=0;
                k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                entireEV=k2_clo+k2_chi;
            end

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+DiscountedEV;

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e=ReturnMatrix_ii_d3e+DiscountedEV(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii_d3e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+reshape(DiscountedEVinterp(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_jj(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3_jj(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d2);
                linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_ii_d3e(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_d3e(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_jj(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer loop over z (markov), inner loop over e, vectorize over semiz

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz_next, N_semiz]
                EV(isnan(EV))=0;
                EV=sum(EV,2); % [N_a, 1, N_semiz]
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                if length(n_a2)==1
                    EV1=EV_2D(aprimeIndex+semizblock_offset);
                    EV2=EV_2D(aprimeplus1Index+semizblock_offset);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=aprimeProbs_d2a1a2semiz;
                    aprimeProbs_z(skipinterp)=0;
                    entireEV=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                    % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                    % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                    k2_Vll=EV_2D(aIdxK_ll+semizblock_offset);
                    k2_Vhl=EV_2D(aIdxK_hl+semizblock_offset);
                    k2_Vlh=EV_2D(aIdxK_lh+semizblock_offset);
                    k2_Vhh=EV_2D(aIdxK_hh+semizblock_offset);
                    k2_p1=prK_1; k2_p1(k2_Vll==k2_Vhl)=0;
                    k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                    k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                    k2_lo=k2_cll+k2_chl;
                    k2_p1b=prK_1; k2_p1b(k2_Vlh==k2_Vhh)=0;
                    k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                    k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                    k2_hi=k2_clh+k2_chh;
                    k2_p2=prK_2; k2_p2(k2_lo==k2_hi)=0;
                    k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                    k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                    entireEV=k2_clo+k2_chi;
                end

                DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz]);
                DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV;

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV(d2aprimez);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind+N_d2*N_a*semizBind;
                    Policy3_ford3_jj(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy3_ford3_jj(2,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_jj(3,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    isInfLower = (ReturnMatrix_ii_d3ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_d3ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_jj(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end

    elseif vfoptions.lowmemory==3

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                k2_Vll=EV_2D(aIdx_ll+bothz_offset);
                k2_Vhl=EV_2D(aIdx_hl+bothz_offset);
                k2_Vlh=EV_2D(aIdx_lh+bothz_offset);
                k2_Vhh=EV_2D(aIdx_hh+bothz_offset);
                k2_p1=pr1_full; k2_p1(k2_Vll==k2_Vhl)=0;
                k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                k2_lo=k2_cll+k2_chl;
                k2_p1b=pr1_full; k2_p1b(k2_Vlh==k2_Vhh)=0;
                k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                k2_hi=k2_clh+k2_chh;
                k2_p2=pr2_full; k2_p2(k2_lo==k2_hi)=0;
                k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                entireEV=k2_clo+k2_chi;
            end

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_z;

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_z(d2aprime);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp_z(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_jj(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_jj(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                  + N_d2*n2long*aind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLower = (ReturnMatrix_ii_d3ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_d3ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_jj(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=3*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    Policy(5,:,:,:,N_j)=reshape(flag_ford3_jj(flat_idx),[1,N_a,N_bothz,N_e]);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    if length(n_a2)==1
        aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
        aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
        aprimeProbs_d2a1a2semiz=repmat(a2primeProbs,N_a1,1,1);
        aprimeIndex_full=repmat(aprimeIndex,1,1,N_z);
        aprimeplus1Index_full=repmat(aprimeplus1Index,1,1,N_z);
        aprimeProbs_full=repmat(aprimeProbs_d2a1a2semiz,1,1,N_z);
    else
        % l_a2==2: the builder returns [l_a2,N_d2,N_a2,N_semiz] per-dim factored. Fold the two
        % per-dim lower indexes into the four corners of the a2 lattice, then push each corner
        % through THIS FILE'S OWN construction so the a1-Kron and any tiling over z are applied
        % exactly as the l_a2==1 arm applies them.
        n_a2_1=n_a2(1);
        lo_1=reshape(a2primeIndex(1,:,:,:),[N_d2,N_a2,N_semiz]);
        lo_2=reshape(a2primeIndex(2,:,:,:),[N_d2,N_a2,N_semiz]);
        p_1=reshape(a2primeProbs(1,:,:,:),[N_d2,N_a2,N_semiz]);
        p_2=reshape(a2primeProbs(2,:,:,:),[N_d2,N_a2,N_semiz]);
        a2k_ll=lo_1+n_a2_1*(lo_2-1);
        a2k_hl=(lo_1+1)+n_a2_1*(lo_2-1);
        a2k_lh=lo_1+n_a2_1*lo_2;
        a2k_hh=(lo_1+1)+n_a2_1*lo_2;
        aIdxK_ll=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_ll-1,N_a1,1,1);
        aIdxK_hl=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_hl-1,N_a1,1,1);
        aIdxK_lh=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_lh-1,N_a1,1,1);
        aIdxK_hh=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_hh-1,N_a1,1,1);
        prK_1=repmat(p_1,N_a1,1,1);
        prK_2=repmat(p_2,N_a1,1,1);
        aIdx_ll=repmat(aIdxK_ll,1,1,N_z);
        aIdx_hl=repmat(aIdxK_hl,1,1,N_z);
        aIdx_lh=repmat(aIdxK_lh,1,1,N_z);
        aIdx_hh=repmat(aIdxK_hh,1,1,N_z);
        pr1_full=repmat(prK_1,1,1,N_z);
        pr2_full=repmat(prK_2,1,1,N_z);
    end

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                k2_Vll=EV_2D(aIdx_ll+bothz_offset);
                k2_Vhl=EV_2D(aIdx_hl+bothz_offset);
                k2_Vlh=EV_2D(aIdx_lh+bothz_offset);
                k2_Vhh=EV_2D(aIdx_hh+bothz_offset);
                k2_p1=pr1_full; k2_p1(k2_Vll==k2_Vhl)=0;
                k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                k2_lo=k2_cll+k2_chl;
                k2_p1b=pr1_full; k2_p1b(k2_Vlh==k2_Vhh)=0;
                k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                k2_hi=k2_clh+k2_chh;
                k2_p2=pr2_full; k2_p2(k2_lo==k2_hi)=0;
                k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                entireEV=k2_clo+k2_chi;
            end

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            % First layer
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV;

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            % Second layer of DC
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Grid interpolation layer
            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]); % midpoint (hence gather) is e-dependent at lowmemory=0
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);

            % Store each d3 (later max over d3)
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_jj(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford3_jj(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_ii_d3(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_d3(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_jj(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end

    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                k2_Vll=EV_2D(aIdx_ll+bothz_offset);
                k2_Vhl=EV_2D(aIdx_hl+bothz_offset);
                k2_Vlh=EV_2D(aIdx_lh+bothz_offset);
                k2_Vhh=EV_2D(aIdx_hh+bothz_offset);
                k2_p1=pr1_full; k2_p1(k2_Vll==k2_Vhl)=0;
                k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                k2_lo=k2_cll+k2_chl;
                k2_p1b=pr1_full; k2_p1b(k2_Vlh==k2_Vhh)=0;
                k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                k2_hi=k2_clh+k2_chh;
                k2_p2=pr2_full; k2_p2(k2_lo==k2_hi)=0;
                k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                entireEV=k2_clo+k2_chi;
            end

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+DiscountedEV;

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e=ReturnMatrix_ii_d3e+DiscountedEV(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii_d3e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+reshape(DiscountedEVinterp(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_jj(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3_jj(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d2);
                linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_ii_d3e(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_d3e(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_jj(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer loop over z (markov), inner loop over e, vectorize over semiz

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz_next, N_semiz]
                EV(isnan(EV))=0;
                EV=sum(EV,2); % [N_a, 1, N_semiz]
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                if length(n_a2)==1
                    EV1=EV_2D(aprimeIndex+semizblock_offset);
                    EV2=EV_2D(aprimeplus1Index+semizblock_offset);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=aprimeProbs_d2a1a2semiz;
                    aprimeProbs_z(skipinterp)=0;
                    entireEV=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                    % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                    % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                    k2_Vll=EV_2D(aIdxK_ll+semizblock_offset);
                    k2_Vhl=EV_2D(aIdxK_hl+semizblock_offset);
                    k2_Vlh=EV_2D(aIdxK_lh+semizblock_offset);
                    k2_Vhh=EV_2D(aIdxK_hh+semizblock_offset);
                    k2_p1=prK_1; k2_p1(k2_Vll==k2_Vhl)=0;
                    k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                    k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                    k2_lo=k2_cll+k2_chl;
                    k2_p1b=prK_1; k2_p1b(k2_Vlh==k2_Vhh)=0;
                    k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                    k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                    k2_hi=k2_clh+k2_chh;
                    k2_p2=prK_2; k2_p2(k2_lo==k2_hi)=0;
                    k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                    k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                    entireEV=k2_clo+k2_chi;
                end

                DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz]);
                DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV;

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV(d2aprimez);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind+N_d2*N_a*semizBind;
                    Policy3_ford3_jj(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy3_ford3_jj(2,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_jj(3,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    isInfLower = (ReturnMatrix_ii_d3ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_d3ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_jj(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end

    elseif vfoptions.lowmemory==3

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                k2_Vll=EV_2D(aIdx_ll+bothz_offset);
                k2_Vhl=EV_2D(aIdx_hl+bothz_offset);
                k2_Vlh=EV_2D(aIdx_lh+bothz_offset);
                k2_Vhh=EV_2D(aIdx_hh+bothz_offset);
                k2_p1=pr1_full; k2_p1(k2_Vll==k2_Vhl)=0;
                k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                k2_lo=k2_cll+k2_chl;
                k2_p1b=pr1_full; k2_p1b(k2_Vlh==k2_Vhh)=0;
                k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                k2_hi=k2_clh+k2_chh;
                k2_p2=pr2_full; k2_p2(k2_lo==k2_hi)=0;
                k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                entireEV=k2_clo+k2_chi;
            end

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_z;

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_z(d2aprime);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp_z(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_jj(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_jj(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                  + N_d2*n2long*aind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLower = (ReturnMatrix_ii_d3ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_d3ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_jj(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=3*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,jj)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,jj)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    Policy(5,:,:,:,jj)=reshape(flag_ford3_jj(flat_idx),[1,N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(4,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-(n2short+1)*(~adjust);


end
