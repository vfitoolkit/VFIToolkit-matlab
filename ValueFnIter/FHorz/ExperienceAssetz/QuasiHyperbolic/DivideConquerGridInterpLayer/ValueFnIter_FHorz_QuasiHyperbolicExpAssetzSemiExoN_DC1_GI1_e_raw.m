function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoN_DC1_GI1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset (z-dependent aprimeFn)
% z is exogenous markov state (required), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, z, ...)
% DC + GI splice (no L2flag scaffold).

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
d2ind=repelem(gpuArray(1:1:N_d2)',N_d1,1); % [N_d12,1]
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

Valt=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policyalt=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray');
PolicyL2flagalt=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
if vfoptions.lowmemory==0
    midpoint_alt=zeros(N_d12,1,N_a1,N_a2,N_bothz,N_e,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a1,N_a2,N_bothz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint_alt=zeros(N_d12,1,N_a1,N_a2,N_bothz,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a1,N_a2,N_bothz,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint_alt=zeros(N_d12,1,N_a1,N_a2,N_semiz,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a1,N_a2,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==3
    midpoint_alt=zeros(N_d12,1,N_a1,N_a2,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a1,N_a2,'gpuArray');
end

V_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_alt=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_tilde=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray');
flag_ford3_alt=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray');
flag_ford3_tilde=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray'); % L2 flag per d3, aggregated after d3 max

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

bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            midpoint_alt(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoint_alt(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint_alt(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_alt(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint_alt(allind)),-1);
            Policy4_ford3_alt(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d12);
            linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end

    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

                midpoint_alt(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        [~,maxindex]=max(ReturnMatrix_ii_e,[],2);
                        midpoint_alt(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint_alt(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind)),-1);
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end

    elseif vfoptions.lowmemory==2 % split: vectorize semiz, outer loop z, inner loop e
        semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % [1,1,N_semiz]
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);

                    midpoint_alt(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            [~,maxindex]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint_alt(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint_alt(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind)),-1);
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_alt(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint loop over bothz (outer), inner loop e

        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);

                    midpoint_alt(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            [~,maxindex]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint_alt(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint_alt(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind)),-1);
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    % Max over d3 (dim 4 of V_ford3_alt since e is dim 3)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policyalt(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(2,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flagalt(1,:,:,:,N_j)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz,N_e]);
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy(:,:,:,:,N_j)=Policyalt(:,:,:,:,N_j);
    PolicyL2flag(:,:,:,:,N_j)=PolicyL2flagalt(:,:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);

    % Integrate out e first
    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3); % [N_a,N_bothz]

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            % n-Monotonicity
            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            % n-Monotonicity

            % --- alt pass ---
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1); % broadcasts over e

            [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
            midpoint_alt(:,1,level1ii,:,:,:)=maxindex1_alt;

            maxgap_alt=squeeze(max(max(max(max(maxindex1_alt(:,1,2:end,:,:,:)-maxindex1_alt(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_alt(ii)>0
                    loweredge_alt=min(maxindex1_alt(:,1,ii,:,:,:),N_a1-maxgap_alt(ii));
                    a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                    ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind; % broadcasts over e
                    entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt);
                    [~,maxindex_alt]=max(entireRHS_ii_d3_alt,[],2);
                    midpoint_alt(:,1,curraindex_alt,:,:,:)=maxindex_alt+(loweredge_alt-1);
                else
                    loweredge_alt=maxindex1_alt(:,1,ii,:,:,:);
                    midpoint_alt(:,1,curraindex_alt,:,:,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                end
            end

            midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
            a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothz_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,1]); % broadcasts over e
            [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii_alt,1);
            d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
            allind_alt=d_ind_alt+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind_alt-1,N_d1)+1;
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind_alt/N_d1);
            Policy4_ford3_alt(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
            Policy4_ford3_alt(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset_alt = ceil(maxindexL2_alt/N_d12);
            linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            isInfLower_alt = (ReturnMatrix_ii_d3_alt(linidx_lower_alt) == -Inf);
            isInfUpper_alt = (ReturnMatrix_ii_d3_alt(linidx_upper_alt) == -Inf);
            inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
            inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

            % --- tilde pass ---
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1); % broadcasts over e

            [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
            midpoint_tilde(:,1,level1ii,:,:,:)=maxindex1_tilde;

            maxgap_tilde=squeeze(max(max(max(max(maxindex1_tilde(:,1,2:end,:,:,:)-maxindex1_tilde(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_tilde(ii)>0
                    loweredge_tilde=min(maxindex1_tilde(:,1,ii,:,:,:),N_a1-maxgap_tilde(ii));
                    a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                    ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind; % broadcasts over e
                    entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                    [~,maxindex_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                    midpoint_tilde(:,1,curraindex_tilde,:,:,:)=maxindex_tilde+(loweredge_tilde-1);
                else
                    loweredge_tilde=maxindex1_tilde(:,1,ii,:,:,:);
                    midpoint_tilde(:,1,curraindex_tilde,:,:,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                end
            end

            midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
            a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothz_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,1]); % broadcasts over e
            [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii_tilde,1);
            d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
            allind_tilde=d_ind_tilde+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind_tilde/N_d1);
            Policy4_ford3_tilde(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
            Policy4_ford3_tilde(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
            linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            isInfLower_tilde = (ReturnMatrix_ii_d3_tilde(linidx_lower_tilde) == -Inf);
            isInfUpper_tilde = (ReturnMatrix_ii_d3_tilde(linidx_upper_tilde) == -Inf);
            inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
            inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
            flag_ford3_tilde(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
        end

    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);


            % --- alt pass ---
                ReturnMatrix_ii_d3e_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e_alt=ReturnMatrix_ii_d3e_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

                [~,maxindex1_alt]=max(entireRHS_ii_d3e_alt,[],2);
                midpoint_alt(:,1,level1ii,:,:)=maxindex1_alt;

                maxgap_alt=squeeze(max(max(max(maxindex1_alt(:,1,2:end,:,:)-maxindex1_alt(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_alt(ii)>0
                        loweredge_alt=min(maxindex1_alt(:,1,ii,:,:),N_a1-maxgap_alt(ii));
                        a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                        ReturnMatrix_ii_d3e_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e_alt=ReturnMatrix_ii_d3e_alt+DiscountedEV_alt(d2aprimez_alt);
                        [~,maxindex_alt]=max(entireRHS_ii_d3e_alt,[],2);
                        midpoint_alt(:,1,curraindex_alt,:,:)=maxindex_alt+(loweredge_alt-1);
                    else
                        loweredge_alt=maxindex1_alt(:,1,ii,:,:);
                        midpoint_alt(:,1,curraindex_alt,:,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e_alt=ReturnMatrix_ii_d3e_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3e_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
                allind_alt=d_ind_alt+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind_alt-1,N_d1)+1;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind_alt/N_d1);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_alt = ceil(maxindexL2_alt/N_d12);
                linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                isInfLower_alt = (ReturnMatrix_ii_d3e_alt(linidx_lower_alt) == -Inf);
                isInfUpper_alt = (ReturnMatrix_ii_d3e_alt(linidx_upper_alt) == -Inf);
                inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
                inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

            % --- tilde pass ---
                ReturnMatrix_ii_d3e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e_tilde=ReturnMatrix_ii_d3e_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

                [~,maxindex1_tilde]=max(entireRHS_ii_d3e_tilde,[],2);
                midpoint_tilde(:,1,level1ii,:,:)=maxindex1_tilde;

                maxgap_tilde=squeeze(max(max(max(maxindex1_tilde(:,1,2:end,:,:)-maxindex1_tilde(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_tilde(ii)>0
                        loweredge_tilde=min(maxindex1_tilde(:,1,ii,:,:),N_a1-maxgap_tilde(ii));
                        a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                        ReturnMatrix_ii_d3e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e_tilde=ReturnMatrix_ii_d3e_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                        [~,maxindex_tilde]=max(entireRHS_ii_d3e_tilde,[],2);
                        midpoint_tilde(:,1,curraindex_tilde,:,:)=maxindex_tilde+(loweredge_tilde-1);
                    else
                        loweredge_tilde=maxindex1_tilde(:,1,ii,:,:);
                        midpoint_tilde(:,1,curraindex_tilde,:,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e_tilde=ReturnMatrix_ii_d3e_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3e_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
                allind_tilde=d_ind_tilde+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind_tilde/N_d1);
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
                linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                isInfLower_tilde = (ReturnMatrix_ii_d3e_tilde(linidx_lower_tilde) == -Inf);
                isInfUpper_tilde = (ReturnMatrix_ii_d3e_tilde(linidx_upper_tilde) == -Inf);
                inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
                inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
                flag_ford3_tilde(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
            end
        end

    elseif vfoptions.lowmemory==2 % split: vectorize semiz, outer loop z, inner loop e
        semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3); % [1,1,1,1,N_semiz]
        semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % [1,1,N_semiz]
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV(isnan(EV))=0;
                EV=sum(EV,2);
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_full(:,:,semizblock);
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                EVbase_qh=reshape(entireEV_z,[N_d2,N_a1,1,N_a2,N_semiz]);
                DiscountedEV_alt=beta*EVbase_qh;
                DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                DiscountedEV_tilde=beta0beta*EVbase_qh;
                DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);


                % --- alt pass ---
                    ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

                    [~,maxindex1_alt]=max(entireRHS_ii_d3ze_alt,[],2);
                    midpoint_alt(:,1,level1ii,:,:)=maxindex1_alt;

                    maxgap_alt=squeeze(max(max(max(maxindex1_alt(:,1,2:end,:,:)-maxindex1_alt(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,ii,:,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                            entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+DiscountedEV_alt(d2aprimez_alt);
                            [~,maxindex_alt]=max(entireRHS_ii_d3ze_alt,[],2);
                            midpoint_alt(:,1,curraindex_alt,:,:)=maxindex_alt+(loweredge_alt-1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,ii,:,:);
                            midpoint_alt(:,1,curraindex_alt,:,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                    a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2bothz_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                    entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3ze_alt,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
                    allind_alt=d_ind_alt+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=rem(d_ind_alt-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=ceil(d_ind_alt/N_d1);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset_alt = ceil(maxindexL2_alt/N_d12);
                    linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    isInfLower_alt = (ReturnMatrix_ii_d3ze_alt(linidx_lower_alt) == -Inf);
                    isInfUpper_alt = (ReturnMatrix_ii_d3ze_alt(linidx_upper_alt) == -Inf);
                    inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
                    inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
                    flag_ford3_alt(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

                % --- tilde pass ---
                    ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

                    [~,maxindex1_tilde]=max(entireRHS_ii_d3ze_tilde,[],2);
                    midpoint_tilde(:,1,level1ii,:,:)=maxindex1_tilde;

                    maxgap_tilde=squeeze(max(max(max(maxindex1_tilde(:,1,2:end,:,:)-maxindex1_tilde(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,ii,:,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                            entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                            [~,maxindex_tilde]=max(entireRHS_ii_d3ze_tilde,[],2);
                            midpoint_tilde(:,1,curraindex_tilde,:,:)=maxindex_tilde+(loweredge_tilde-1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,ii,:,:);
                            midpoint_tilde(:,1,curraindex_tilde,:,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
                    a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2bothz_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                    entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3ze_tilde,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
                    allind_tilde=d_ind_tilde+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,semizblock,e_c,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
                    Policy4_ford3_tilde(2,:,semizblock,e_c,d3_c)=ceil(d_ind_tilde/N_d1);
                    Policy4_ford3_tilde(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                    Policy4_ford3_tilde(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
                    linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    isInfLower_tilde = (ReturnMatrix_ii_d3ze_tilde(linidx_lower_tilde) == -Inf);
                    isInfUpper_tilde = (ReturnMatrix_ii_d3ze_tilde(linidx_upper_tilde) == -Inf);
                    inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
                    inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
                    flag_ford3_tilde(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint loop over bothz (outer), inner loop e

        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

            % --- alt pass ---
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,z_c);
                DiscountedEVinterp_z_alt=DiscountedEVinterp_alt(:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+repelem(DiscountedEV_z_alt,N_d1,1,1,1);

                    [~,maxindex1_alt]=max(entireRHS_ii_d3ze_alt,[],2);
                    midpoint_alt(:,1,level1ii,:)=maxindex1_alt;

                    maxgap_alt=squeeze(max(max(maxindex1_alt(:,1,2:end,:)-maxindex1_alt(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,ii,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+DiscountedEV_z_alt(d2aprime_alt);
                            [~,maxindex_alt]=max(entireRHS_ii_d3ze_alt,[],2);
                            midpoint_alt(:,1,curraindex_alt,:)=maxindex_alt+(loweredge_alt-1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,ii,:);
                            midpoint_alt(:,1,curraindex_alt,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                    a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+reshape(DiscountedEVinterp_z_alt(d2a1primea2_alt(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3ze_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
                    allind_alt=d_ind_alt+N_d12*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind_alt-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind_alt/N_d1);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset_alt = ceil(maxindexL2_alt/N_d12);
                    linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind;
                    linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower_alt = (ReturnMatrix_ii_d3ze_alt(linidx_lower_alt) == -Inf);
                    isInfUpper_alt = (ReturnMatrix_ii_d3ze_alt(linidx_upper_alt) == -Inf);
                    inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
                    inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);
                end

            % --- tilde pass ---
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,z_c);
                DiscountedEVinterp_z_tilde=DiscountedEVinterp_tilde(:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+repelem(DiscountedEV_z_tilde,N_d1,1,1,1);

                    [~,maxindex1_tilde]=max(entireRHS_ii_d3ze_tilde,[],2);
                    midpoint_tilde(:,1,level1ii,:)=maxindex1_tilde;

                    maxgap_tilde=squeeze(max(max(maxindex1_tilde(:,1,2:end,:)-maxindex1_tilde(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,ii,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+DiscountedEV_z_tilde(d2aprime_tilde);
                            [~,maxindex_tilde]=max(entireRHS_ii_d3ze_tilde,[],2);
                            midpoint_tilde(:,1,curraindex_tilde,:)=maxindex_tilde+(loweredge_tilde-1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,ii,:);
                            midpoint_tilde(:,1,curraindex_tilde,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
                    a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+reshape(DiscountedEVinterp_z_tilde(d2a1primea2_tilde(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3ze_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
                    allind_tilde=d_ind_tilde+N_d12*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind_tilde/N_d1);
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
                    linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind;
                    linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower_tilde = (ReturnMatrix_ii_d3ze_tilde(linidx_lower_tilde) == -Inf);
                    isInfUpper_tilde = (ReturnMatrix_ii_d3ze_tilde(linidx_upper_tilde) == -Inf);
                    inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
                    inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
                    flag_ford3_tilde(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
                end
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policyalt(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(2,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flagalt(1,:,:,:,N_j)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz,N_e]);

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    Vtilde(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]);
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

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);

    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);


            % --- alt pass ---
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

            [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
            midpoint_alt(:,1,level1ii,:,:,:)=maxindex1_alt;

            maxgap_alt=squeeze(max(max(max(max(maxindex1_alt(:,1,2:end,:,:,:)-maxindex1_alt(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_alt(ii)>0
                    loweredge_alt=min(maxindex1_alt(:,1,ii,:,:,:),N_a1-maxgap_alt(ii));
                    a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                    ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt);
                    [~,maxindex_alt]=max(entireRHS_ii_d3_alt,[],2);
                    midpoint_alt(:,1,curraindex_alt,:,:,:)=maxindex_alt+(loweredge_alt-1);
                else
                    loweredge_alt=maxindex1_alt(:,1,ii,:,:,:);
                    midpoint_alt(:,1,curraindex_alt,:,:,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                end
            end

            midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
            a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothz_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii_alt,1);
            d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
            allind_alt=d_ind_alt+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind_alt-1,N_d1)+1;
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind_alt/N_d1);
            Policy4_ford3_alt(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
            Policy4_ford3_alt(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset_alt = ceil(maxindexL2_alt/N_d12);
            linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            isInfLower_alt = (ReturnMatrix_ii_d3_alt(linidx_lower_alt) == -Inf);
            isInfUpper_alt = (ReturnMatrix_ii_d3_alt(linidx_upper_alt) == -Inf);
            inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
            inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

            % --- tilde pass ---
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

            [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
            midpoint_tilde(:,1,level1ii,:,:,:)=maxindex1_tilde;

            maxgap_tilde=squeeze(max(max(max(max(maxindex1_tilde(:,1,2:end,:,:,:)-maxindex1_tilde(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_tilde(ii)>0
                    loweredge_tilde=min(maxindex1_tilde(:,1,ii,:,:,:),N_a1-maxgap_tilde(ii));
                    a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                    ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                    [~,maxindex_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                    midpoint_tilde(:,1,curraindex_tilde,:,:,:)=maxindex_tilde+(loweredge_tilde-1);
                else
                    loweredge_tilde=maxindex1_tilde(:,1,ii,:,:,:);
                    midpoint_tilde(:,1,curraindex_tilde,:,:,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                end
            end

            midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
            a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothz_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii_tilde,1);
            d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
            allind_tilde=d_ind_tilde+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind_tilde/N_d1);
            Policy4_ford3_tilde(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
            Policy4_ford3_tilde(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
            linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            isInfLower_tilde = (ReturnMatrix_ii_d3_tilde(linidx_lower_tilde) == -Inf);
            isInfUpper_tilde = (ReturnMatrix_ii_d3_tilde(linidx_upper_tilde) == -Inf);
            inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
            inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
            flag_ford3_tilde(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
        end

    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);


            % --- alt pass ---
                ReturnMatrix_ii_d3e_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e_alt=ReturnMatrix_ii_d3e_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

                [~,maxindex1_alt]=max(entireRHS_ii_d3e_alt,[],2);
                midpoint_alt(:,1,level1ii,:,:)=maxindex1_alt;

                maxgap_alt=squeeze(max(max(max(maxindex1_alt(:,1,2:end,:,:)-maxindex1_alt(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_alt(ii)>0
                        loweredge_alt=min(maxindex1_alt(:,1,ii,:,:),N_a1-maxgap_alt(ii));
                        a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                        ReturnMatrix_ii_d3e_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e_alt=ReturnMatrix_ii_d3e_alt+DiscountedEV_alt(d2aprimez_alt);
                        [~,maxindex_alt]=max(entireRHS_ii_d3e_alt,[],2);
                        midpoint_alt(:,1,curraindex_alt,:,:)=maxindex_alt+(loweredge_alt-1);
                    else
                        loweredge_alt=maxindex1_alt(:,1,ii,:,:);
                        midpoint_alt(:,1,curraindex_alt,:,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e_alt=ReturnMatrix_ii_d3e_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3e_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
                allind_alt=d_ind_alt+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind_alt-1,N_d1)+1;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind_alt/N_d1);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_alt = ceil(maxindexL2_alt/N_d12);
                linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                isInfLower_alt = (ReturnMatrix_ii_d3e_alt(linidx_lower_alt) == -Inf);
                isInfUpper_alt = (ReturnMatrix_ii_d3e_alt(linidx_upper_alt) == -Inf);
                inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
                inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

            % --- tilde pass ---
                ReturnMatrix_ii_d3e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e_tilde=ReturnMatrix_ii_d3e_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

                [~,maxindex1_tilde]=max(entireRHS_ii_d3e_tilde,[],2);
                midpoint_tilde(:,1,level1ii,:,:)=maxindex1_tilde;

                maxgap_tilde=squeeze(max(max(max(maxindex1_tilde(:,1,2:end,:,:)-maxindex1_tilde(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_tilde(ii)>0
                        loweredge_tilde=min(maxindex1_tilde(:,1,ii,:,:),N_a1-maxgap_tilde(ii));
                        a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                        ReturnMatrix_ii_d3e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e_tilde=ReturnMatrix_ii_d3e_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                        [~,maxindex_tilde]=max(entireRHS_ii_d3e_tilde,[],2);
                        midpoint_tilde(:,1,curraindex_tilde,:,:)=maxindex_tilde+(loweredge_tilde-1);
                    else
                        loweredge_tilde=maxindex1_tilde(:,1,ii,:,:);
                        midpoint_tilde(:,1,curraindex_tilde,:,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e_tilde=ReturnMatrix_ii_d3e_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3e_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
                allind_tilde=d_ind_tilde+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind_tilde/N_d1);
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
                linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                isInfLower_tilde = (ReturnMatrix_ii_d3e_tilde(linidx_lower_tilde) == -Inf);
                isInfUpper_tilde = (ReturnMatrix_ii_d3e_tilde(linidx_upper_tilde) == -Inf);
                inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
                inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
                flag_ford3_tilde(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
            end
        end

    elseif vfoptions.lowmemory==2 % split: vectorize semiz, outer loop z, inner loop e
        semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3); % [1,1,1,1,N_semiz]
        semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % [1,1,N_semiz]
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV(isnan(EV))=0;
                EV=sum(EV,2);
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_full(:,:,semizblock);
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                EVbase_qh=reshape(entireEV_z,[N_d2,N_a1,1,N_a2,N_semiz]);
                DiscountedEV_alt=beta*EVbase_qh;
                DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                DiscountedEV_tilde=beta0beta*EVbase_qh;
                DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);


                % --- alt pass ---
                    ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

                    [~,maxindex1_alt]=max(entireRHS_ii_d3ze_alt,[],2);
                    midpoint_alt(:,1,level1ii,:,:)=maxindex1_alt;

                    maxgap_alt=squeeze(max(max(max(maxindex1_alt(:,1,2:end,:,:)-maxindex1_alt(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,ii,:,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                            entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+DiscountedEV_alt(d2aprimez_alt);
                            [~,maxindex_alt]=max(entireRHS_ii_d3ze_alt,[],2);
                            midpoint_alt(:,1,curraindex_alt,:,:)=maxindex_alt+(loweredge_alt-1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,ii,:,:);
                            midpoint_alt(:,1,curraindex_alt,:,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                    a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2bothz_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                    entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3ze_alt,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
                    allind_alt=d_ind_alt+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=rem(d_ind_alt-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=ceil(d_ind_alt/N_d1);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset_alt = ceil(maxindexL2_alt/N_d12);
                    linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    isInfLower_alt = (ReturnMatrix_ii_d3ze_alt(linidx_lower_alt) == -Inf);
                    isInfUpper_alt = (ReturnMatrix_ii_d3ze_alt(linidx_upper_alt) == -Inf);
                    inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
                    inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
                    flag_ford3_alt(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

                % --- tilde pass ---
                    ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

                    [~,maxindex1_tilde]=max(entireRHS_ii_d3ze_tilde,[],2);
                    midpoint_tilde(:,1,level1ii,:,:)=maxindex1_tilde;

                    maxgap_tilde=squeeze(max(max(max(maxindex1_tilde(:,1,2:end,:,:)-maxindex1_tilde(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,ii,:,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                            entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                            [~,maxindex_tilde]=max(entireRHS_ii_d3ze_tilde,[],2);
                            midpoint_tilde(:,1,curraindex_tilde,:,:)=maxindex_tilde+(loweredge_tilde-1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,ii,:,:);
                            midpoint_tilde(:,1,curraindex_tilde,:,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
                    a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2bothz_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                    entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3ze_tilde,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
                    allind_tilde=d_ind_tilde+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,semizblock,e_c,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
                    Policy4_ford3_tilde(2,:,semizblock,e_c,d3_c)=ceil(d_ind_tilde/N_d1);
                    Policy4_ford3_tilde(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                    Policy4_ford3_tilde(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
                    linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    isInfLower_tilde = (ReturnMatrix_ii_d3ze_tilde(linidx_lower_tilde) == -Inf);
                    isInfUpper_tilde = (ReturnMatrix_ii_d3ze_tilde(linidx_upper_tilde) == -Inf);
                    inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
                    inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
                    flag_ford3_tilde(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint loop over bothz (outer), inner loop e

        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

            % --- alt pass ---
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,z_c);
                DiscountedEVinterp_z_alt=DiscountedEVinterp_alt(:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+repelem(DiscountedEV_z_alt,N_d1,1,1,1);

                    [~,maxindex1_alt]=max(entireRHS_ii_d3ze_alt,[],2);
                    midpoint_alt(:,1,level1ii,:)=maxindex1_alt;

                    maxgap_alt=squeeze(max(max(maxindex1_alt(:,1,2:end,:)-maxindex1_alt(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,ii,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+DiscountedEV_z_alt(d2aprime_alt);
                            [~,maxindex_alt]=max(entireRHS_ii_d3ze_alt,[],2);
                            midpoint_alt(:,1,curraindex_alt,:)=maxindex_alt+(loweredge_alt-1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,ii,:);
                            midpoint_alt(:,1,curraindex_alt,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                    a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze_alt=ReturnMatrix_ii_d3ze_alt+reshape(DiscountedEVinterp_z_alt(d2a1primea2_alt(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3ze_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
                    allind_alt=d_ind_alt+N_d12*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind_alt-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind_alt/N_d1);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset_alt = ceil(maxindexL2_alt/N_d12);
                    linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind;
                    linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower_alt = (ReturnMatrix_ii_d3ze_alt(linidx_lower_alt) == -Inf);
                    isInfUpper_alt = (ReturnMatrix_ii_d3ze_alt(linidx_upper_alt) == -Inf);
                    inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
                    inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);
                end

            % --- tilde pass ---
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,z_c);
                DiscountedEVinterp_z_tilde=DiscountedEVinterp_tilde(:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+repelem(DiscountedEV_z_tilde,N_d1,1,1,1);

                    [~,maxindex1_tilde]=max(entireRHS_ii_d3ze_tilde,[],2);
                    midpoint_tilde(:,1,level1ii,:)=maxindex1_tilde;

                    maxgap_tilde=squeeze(max(max(maxindex1_tilde(:,1,2:end,:)-maxindex1_tilde(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,ii,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+DiscountedEV_z_tilde(d2aprime_tilde);
                            [~,maxindex_tilde]=max(entireRHS_ii_d3ze_tilde,[],2);
                            midpoint_tilde(:,1,curraindex_tilde,:)=maxindex_tilde+(loweredge_tilde-1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,ii,:);
                            midpoint_tilde(:,1,curraindex_tilde,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
                    a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze_tilde=ReturnMatrix_ii_d3ze_tilde+reshape(DiscountedEVinterp_z_tilde(d2a1primea2_tilde(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3ze_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
                    allind_tilde=d_ind_tilde+N_d12*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind_tilde/N_d1);
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
                    linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind;
                    linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower_tilde = (ReturnMatrix_ii_d3ze_tilde(linidx_lower_tilde) == -Inf);
                    isInfUpper_tilde = (ReturnMatrix_ii_d3ze_tilde(linidx_upper_tilde) == -Inf);
                    inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
                    inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
                    flag_ford3_tilde(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
                end
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,jj)=V_jj;
    Policyalt(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policyalt(1,:,:,:,jj)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(2,:,:,:,jj)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,jj)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,jj)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flagalt(1,:,:,:,jj)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz,N_e]);

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    Vtilde(:,:,:,jj)=V_jj;
    Policy(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(2,:,:,:,jj)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_bothz,N_e]);

end


%% Switch from midpoint to lower grid index
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

adjustalt=(Policyalt(5,:,:,:,:)<1+n2short+1);
Policyalt(4,:,:,:,:)=Policyalt(4,:,:,:,:)-adjustalt;
Policyalt(5,:,:,:,:)=adjustalt.*Policyalt(5,:,:,:,:)+(1-adjustalt).*(Policyalt(5,:,:,:,:)-n2short-1);

Policyalt=[Policyalt; PolicyL2flagalt];


end
