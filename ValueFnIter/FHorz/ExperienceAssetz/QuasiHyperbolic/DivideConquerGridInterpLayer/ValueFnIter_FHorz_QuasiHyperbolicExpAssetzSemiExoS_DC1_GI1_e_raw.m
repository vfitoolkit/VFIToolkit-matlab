function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoS_DC1_GI1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
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

Vhat=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray');
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
    midpoint_hat=zeros(N_d12,1,N_a1,N_a2,N_bothz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint_hat=zeros(N_d12,1,N_a1,N_a2,N_bothz,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint_hat=zeros(N_d12,1,N_a1,N_a2,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==3
    midpoint_hat=zeros(N_d12,1,N_a1,N_a2,'gpuArray');
end

V_ford3_hat=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_hat=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray');
flag_ford3_hat=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray'); % L2 flag per d3, aggregated after d3 max

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

            midpoint_hat(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoint_hat(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint_hat(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eBind;
            Policy4_ford3_hat(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_hat(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_hat(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
            Policy4_ford3_hat(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d12);
            linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_hat(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end

    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

                midpoint_hat(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        [~,maxindex]=max(ReturnMatrix_ii_e,[],2);
                        midpoint_hat(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint_hat(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_hat(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_hat(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_hat(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                Policy4_ford3_hat(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_hat(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
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

                    midpoint_hat(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            [~,maxindex]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint_hat(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint_hat(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_hat(1,:,semizblock,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_hat(2,:,semizblock,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_hat(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                    Policy4_ford3_hat(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_hat(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
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

                    midpoint_hat(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            [~,maxindex]=max(ReturnMatrix_ii_ze,[],2);
                            midpoint_hat(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint_hat(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_hat(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_hat(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_hat(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                    Policy4_ford3_hat(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_hat(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    % Max over d3 (dim 4 of V_ford3_hat since e is dim 3)
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_hat(flat_idx),[1,N_a,N_bothz,N_e]);
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            % n-Monotonicity
            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            % n-Monotonicity
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_hat,N_d1,1,1,1,1); % broadcasts over e

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint_hat(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind; % broadcasts over e
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_d3,[],2);
                    midpoint_hat(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint_hat(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothz=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp_hat(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,1]); % broadcasts over e
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp_under(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,1]);
            maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(0:1:(N_bothz)-1),-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(N_bothz)*(0:1:(1)-1),-2);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eBind;
            Policy4_ford3_hat(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_hat(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_hat(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
            Policy4_ford3_hat(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d12);
            linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_ii_d3(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_d3(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_hat(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+repelem(DiscountedEV_hat,N_d1,1,1,1,1);

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                midpoint_hat(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e=ReturnMatrix_ii_d3e+DiscountedEV_hat(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii_d3e,[],2);
                        midpoint_hat(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint_hat(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+reshape(DiscountedEVinterp_hat(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_ii_d3e+reshape(DiscountedEVinterp_under(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
            maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(0:1:(N_bothz)-1),-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(N_bothz)*(0:1:(N_e)-1),-2);
            V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_hat(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_hat(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_hat(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                Policy4_ford3_hat(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_ii_d3e(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_d3e(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_hat(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
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
                DiscountedEV_under=beta*EVbase_qh;
                DiscountedEV_hat=beta0beta*EVbase_qh;
                DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_hat,N_d1,1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint_hat(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_hat(d2aprimez);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint_hat(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint_hat(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2bothz=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp_hat(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp_under(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(0:1:(N_semiz)-1),-1);
                V_ford3_under(:,semizblock,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_hat(1,:,semizblock,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_hat(2,:,semizblock,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_hat(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                    Policy4_ford3_hat(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    isInfLower = (ReturnMatrix_ii_d3ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_d3ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_hat(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z_hat=DiscountedEV_hat(:,:,:,:,z_c);
                DiscountedEV_z_under=DiscountedEV_under(:,:,:,:,z_c);
                DiscountedEVinterp_z_hat=DiscountedEVinterp_hat(:,:,:,:,z_c);
                DiscountedEVinterp_z_under=DiscountedEVinterp_under(:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_z_hat,N_d1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint_hat(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_z_hat(d2aprime);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint_hat(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint_hat(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp_z_hat(d2a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp_z_under(d2a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
            maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1);
            V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_hat(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_hat(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_hat(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                    Policy4_ford3_hat(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower = (ReturnMatrix_ii_d3ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_d3ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_hat(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_hat(flat_idx),[1,N_a,N_bothz,N_e]);

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    Vunderbar(:,:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(d3lin-1)),[N_a,N_bothz,N_e]);
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

    EVpre=sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_hat,N_d1,1,1,1,1);

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint_hat(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_d3,[],2);
                    midpoint_hat(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint_hat(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothz=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp_hat(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp_under(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
            maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(0:1:(N_bothz)-1),-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(N_bothz)*(0:1:(N_e)-1),-2);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eBind;
            Policy4_ford3_hat(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_hat(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_hat(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
            Policy4_ford3_hat(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d12);
            linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind + N_d12*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_ii_d3(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_d3(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_hat(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+repelem(DiscountedEV_hat,N_d1,1,1,1,1);

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                midpoint_hat(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e=ReturnMatrix_ii_d3e+DiscountedEV_hat(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii_d3e,[],2);
                        midpoint_hat(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint_hat(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+reshape(DiscountedEVinterp_hat(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_ii_d3e+reshape(DiscountedEVinterp_under(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
            maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(0:1:(N_bothz)-1),-1);
            V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_hat(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_hat(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_hat(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                Policy4_ford3_hat(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_ii_d3e(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_d3e(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_hat(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
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
                DiscountedEV_under=beta*EVbase_qh;
                DiscountedEV_hat=beta0beta*EVbase_qh;
                DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_hat,N_d1,1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint_hat(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_hat(d2aprimez);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint_hat(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint_hat(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2bothz=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp_hat(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp_under(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(0:1:(N_semiz)-1),-1);
                V_ford3_under(:,semizblock,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_hat(1,:,semizblock,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_hat(2,:,semizblock,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_hat(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                    Policy4_ford3_hat(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                    isInfLower = (ReturnMatrix_ii_d3ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_d3ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_hat(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_z_hat=DiscountedEV_hat(:,:,:,:,z_c);
                DiscountedEV_z_under=DiscountedEV_under(:,:,:,:,z_c);
                DiscountedEVinterp_z_hat=DiscountedEVinterp_hat(:,:,:,:,z_c);
                DiscountedEVinterp_z_under=DiscountedEVinterp_under(:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_z_hat,N_d1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint_hat(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_z_hat(d2aprime);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint_hat(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint_hat(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint_hat=max(min(midpoint_hat,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp_z_hat(d2a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_ii_d3ze+reshape(DiscountedEVinterp_z_under(d2a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
            maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1);
            V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_hat(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_hat(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_hat(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                    Policy4_ford3_hat(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower = (ReturnMatrix_ii_d3ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_d3ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_hat(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,jj)=V_jj;
    Policy(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(2,:,:,:,jj)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_hat(flat_idx),[1,N_a,N_bothz,N_e]);

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    Vunderbar(:,:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(d3lin-1)),[N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];


end
