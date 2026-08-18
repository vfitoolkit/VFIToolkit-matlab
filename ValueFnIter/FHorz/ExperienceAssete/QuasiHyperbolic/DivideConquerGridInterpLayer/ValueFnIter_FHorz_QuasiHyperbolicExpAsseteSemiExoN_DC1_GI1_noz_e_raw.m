function [V,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteSemiExoN_DC1_GI1_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic + ExperienceAssete + SemiExo + DivideConquer + GridInterpLayer.
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% semiz is semi-exog state, e is i.i.d. start-of-period (required); no z
% aprimeFn = aprimeFn(d2, a2, e, ...)   (depends on current e; not on z or semiz)
%
% Naive QH dual pass. Both value functions are computed with the FULL divide-conquer + grid-interp
% machinery (each with its own midpoint localization over the fine grid):
%   Valt/Policyalt maximise  F + beta*EV        (the exponential value)
%   V/Policy       maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Valt (the exponential continuation value).
% Policy rows: (d1, d2, d3, midpoint, L2ind) + L2flag appended; Policyalt likewise.
%
% lowmemory levels {0,1,2} implemented (shocks: semiz + e iid).


N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
d2ind=repelem(gpuArray(1:1:N_d2)',N_d1,1);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_semiz,N_e,N_j,'gpuArray'); % (d1, d2, d3, midpoint, L2ind)
PolicyL2flag=2*ones(1,N_a,N_semiz,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper
Valt=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policyalt=zeros(5,N_a,N_semiz,N_e,N_j,'gpuArray');
PolicyL2flagalt=2*ones(1,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate
if vfoptions.lowmemory==0
    midpoint=zeros(N_d12,1,N_a1,N_a2,N_semiz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d12,1,N_a1,N_a2,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint=zeros(N_d12,1,N_a1,N_a2,'gpuArray');
end

% Per-d3 arrays (alt=exponential [F+beta*EV], tilde=QH-perceived [F+beta0*beta*EV])
V_ford3_alt=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy4_ford3_alt=zeros(4,N_a,N_semiz,N_e,N_d3,'gpuArray'); % (d1, d2, midpoint, L2ind)
flag_ford3_alt=2*ones(1,N_a,N_semiz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy4_ford3_tilde=zeros(4,N_a,N_semiz,N_e,N_d3,'gpuArray');
flag_ford3_tilde=2*ones(1,N_a,N_semiz,N_e,N_d3,'gpuArray');

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
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3); % already includes -1
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1

semiz_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_semiz,n_e, d123_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*semizBind+N_d12*N_a*N_semiz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_alt(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy4_ford3_alt(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d12);
            linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind + N_d12*n2long*N_a*N_semiz*eBind;
            linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind + N_d12*n2long*N_a*N_semiz*eBind;
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

                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                [~,maxindex1]=max(ReturnMatrix_ii,[],2);

                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        [~,maxindex]=max(ReturnMatrix_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end
    elseif vfoptions.lowmemory==2 % loop semiz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

                    midpoint(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            [~,maxindex]=max(ReturnMatrix_ii,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    % Max over d3; terminal period has no continuation, so QH-perceived and exponential coincide
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    temp=4*((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_semiz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_semiz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_semiz,N_e]);
    flat_idx=(1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_semiz,N_e]);
    Valt(:,:,:,N_j)=V(:,:,:,N_j);
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j);
    PolicyL2flagalt(1,:,:,:,N_j)=PolicyL2flag(1,:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex=reshape(aprimeIndex,[N_d2*N_a1,N_a2,1,N_e]);
    aprimeplus1Index=reshape(aprimeplus1Index,[N_d2*N_a1,N_a2,1,N_e]);
    aprimeProbs_d2a1a2e=reshape(aprimeProbs_d2a1a2e,[N_d2*N_a1,N_a2,1,N_e]);

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;


    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_semiz]);

            lin_lower=aprimeIndex+semiz_offset;
            lin_upper=aprimeplus1Index+semiz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
            entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta*EVreshape,N_d1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind+N_d2*N_a1*N_a2*N_semiz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=ReturnMatrix_ii+beta*EVreshape(d2aprimeze);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end
            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_semiz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothze=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind+N_d2*N_a1prime*N_a2*N_semiz*shiftdim((0:1:N_e-1),-4);
            entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(d2a1primea2bothze(:)),[N_d12*n2long,N_a1*N_a2,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*semizBind+N_d12*N_a*N_semiz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_alt(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy4_ford3_alt(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset = ceil(maxindexL2/N_d12);
            linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind + N_d12*n2long*N_a*N_semiz*eBind;
            linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind + N_d12*n2long*N_a*N_semiz*eBind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

            %% Vtilde (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
            entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta0beta*EVreshape,N_d1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind+N_d2*N_a1*N_a2*N_semiz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape(d2aprimeze);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end
            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_semiz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothze=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind+N_d2*N_a1prime*N_a2*N_semiz*shiftdim((0:1:N_e-1),-4);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(d2a1primea2bothze(:)),[N_d12*n2long,N_a1*N_a2,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*semizBind+N_d12*N_a*N_semiz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_tilde(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy4_ford3_tilde(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset = ceil(maxindexL2/N_d12);
            linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind + N_d12*n2long*N_a*N_semiz*eBind;
            linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind + N_d12*n2long*N_a*N_semiz*eBind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_tilde(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_semiz]);

            lin_lower=aprimeIndex+semiz_offset;
            lin_upper=aprimeplus1Index+semiz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                EVreshape_e=EVreshape(:,:,:,:,:,e_c);
                EVinterp_e=EVinterp(:,:,:,:,:,e_c);

                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
                entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta*EVreshape_e,N_d1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                        entireRHS_ii=ReturnMatrix_ii+beta*EVreshape_e(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end
                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_e(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

                %% Vtilde (QH-perceived): F + beta0beta*EV, full divide-conquer + grid-interp
                entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta0beta*EVreshape_e,N_d1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_e(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end
                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_e(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_tilde(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end
    elseif vfoptions.lowmemory==2 %% loop semiz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_semiz]);
            EV1=EV_2D(aprimeIndex+semiz_offset);
            EV2=EV_2D(aprimeplus1Index+semiz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    EVreshape_ze=EVreshape(:,:,:,:,z_c,e_c);
                    EVinterp_ze=EVinterp(:,:,:,:,z_c,e_c);

                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta*EVreshape_ze,N_d1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii=ReturnMatrix_ii+beta*EVreshape_ze(d2aprime);
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end
                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_ze(d2a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

                    %% Vtilde (QH-perceived): F + beta0beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta0beta*EVreshape_ze,N_d1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_ze(d2aprime);
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end
                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_ze(d2a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_tilde(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    % Max over d3 for tilde (QH-perceived) -> V/Policy
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    temp=4*((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_semiz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_semiz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_semiz,N_e]);
    flat_idx=(1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_semiz,N_e]);

    % Max over d3 for alt (exponential) -> Valt/Policyalt
    [Valt_jj,maxindexalt]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=Valt_jj;
    Policyalt(3,:,:,:,N_j)=shiftdim(maxindexalt,-1);
    maxindexalt=reshape(maxindexalt,[N_a*N_semiz*N_e,1]);
    tempalt=4*((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindexalt-1)-1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+tempalt),[1,N_a,N_semiz,N_e]);
    Policyalt(2,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+tempalt),[1,N_a,N_semiz,N_e]);
    Policyalt(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+tempalt),[1,N_a,N_semiz,N_e]);
    Policyalt(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+tempalt),[1,N_a,N_semiz,N_e]);
    flat_idxalt=(1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindexalt-1);
    PolicyL2flagalt(1,:,:,:,N_j)=reshape(flag_ford3_alt(flat_idxalt),[1,N_a,N_semiz,N_e]);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex=reshape(aprimeIndex,[N_d2*N_a1,N_a2,1,N_e]);
    aprimeplus1Index=reshape(aprimeplus1Index,[N_d2*N_a1,N_a2,1,N_e]);
    aprimeProbs_d2a1a2e=reshape(aprimeProbs_d2a1a2e,[N_d2*N_a1,N_a2,1,N_e]);

    % Continuation value is the exponential value (Valt), integrated over e'
    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);


    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_semiz]);

            lin_lower=aprimeIndex+semiz_offset;
            lin_upper=aprimeplus1Index+semiz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
            entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta*EVreshape,N_d1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind+N_d2*N_a1*N_a2*N_semiz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=ReturnMatrix_ii+beta*EVreshape(d2aprimeze);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end
            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_semiz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothze=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind+N_d2*N_a1prime*N_a2*N_semiz*shiftdim((0:1:N_e-1),-4);
            entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(d2a1primea2bothze(:)),[N_d12*n2long,N_a1*N_a2,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*semizBind+N_d12*N_a*N_semiz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_alt(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy4_ford3_alt(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset = ceil(maxindexL2/N_d12);
            linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind + N_d12*n2long*N_a*N_semiz*eBind;
            linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind + N_d12*n2long*N_a*N_semiz*eBind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

            %% Vtilde (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
            entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta0beta*EVreshape,N_d1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind+N_d2*N_a1*N_a2*N_semiz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape(d2aprimeze);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end
            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_semiz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothze=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind+N_d2*N_a1prime*N_a2*N_semiz*shiftdim((0:1:N_e-1),-4);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(d2a1primea2bothze(:)),[N_d12*n2long,N_a1*N_a2,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*semizBind+N_d12*N_a*N_semiz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_tilde(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy4_ford3_tilde(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset = ceil(maxindexL2/N_d12);
            linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind + N_d12*n2long*N_a*N_semiz*eBind;
            linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind + N_d12*n2long*N_a*N_semiz*eBind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_tilde(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_semiz]);

            lin_lower=aprimeIndex+semiz_offset;
            lin_upper=aprimeplus1Index+semiz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                EVreshape_e=EVreshape(:,:,:,:,:,e_c);
                EVinterp_e=EVinterp(:,:,:,:,:,e_c);

                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
                entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta*EVreshape_e,N_d1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                        entireRHS_ii=ReturnMatrix_ii+beta*EVreshape_e(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end
                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_e(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

                %% Vtilde (QH-perceived): F + beta0beta*EV, full divide-conquer + grid-interp
                entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta0beta*EVreshape_e,N_d1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_e(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end
                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_e(d2a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_tilde(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end
    elseif vfoptions.lowmemory==2 %% loop semiz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_semiz]);
            EV1=EV_2D(aprimeIndex+semiz_offset);
            EV2=EV_2D(aprimeplus1Index+semiz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    EVreshape_ze=EVreshape(:,:,:,:,z_c,e_c);
                    EVinterp_ze=EVinterp(:,:,:,:,z_c,e_c);

                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    %% Valt (exponential): F + beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta*EVreshape_ze,N_d1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii=ReturnMatrix_ii+beta*EVreshape_ze(d2aprime);
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end
                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_ze(d2a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

                    %% Vtilde (QH-perceived): F + beta0beta*EV, full divide-conquer + grid-interp
                    entireRHS_ii=ReturnMatrix_ii_d3+repelem(beta0beta*EVreshape_ze,N_d1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_ze(d2aprime);
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end
                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_ze(d2a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset = ceil(maxindexL2/N_d12);
                    linidx_lower = d_ind                   + N_d12*n2long*aind;
                    linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind;
                    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_tilde(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    % Max over d3 for tilde (QH-perceived) -> V/Policy
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    V(:,:,:,jj)=V_jj;
    Policy(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    temp=4*((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_semiz,N_e]);
    Policy(2,:,:,:,jj)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_semiz,N_e]);
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_semiz,N_e]);
    flat_idx=(1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_semiz,N_e]);

    % Max over d3 for alt (exponential) -> Valt/Policyalt
    [Valt_jj,maxindexalt]=max(V_ford3_alt,[],4);
    Valt(:,:,:,jj)=Valt_jj;
    Policyalt(3,:,:,:,jj)=shiftdim(maxindexalt,-1);
    maxindexalt=reshape(maxindexalt,[N_a*N_semiz*N_e,1]);
    tempalt=4*((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindexalt-1)-1);
    Policyalt(1,:,:,:,jj)=reshape(Policy4_ford3_alt(1+tempalt),[1,N_a,N_semiz,N_e]);
    Policyalt(2,:,:,:,jj)=reshape(Policy4_ford3_alt(2+tempalt),[1,N_a,N_semiz,N_e]);
    Policyalt(4,:,:,:,jj)=reshape(Policy4_ford3_alt(3+tempalt),[1,N_a,N_semiz,N_e]);
    Policyalt(5,:,:,:,jj)=reshape(Policy4_ford3_alt(4+tempalt),[1,N_a,N_semiz,N_e]);
    flat_idxalt=(1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindexalt-1);
    PolicyL2flagalt(1,:,:,:,jj)=reshape(flag_ford3_alt(flat_idxalt),[1,N_a,N_semiz,N_e]);
end


%% Switch from midpoint to lower grid index (QH-perceived Policy)
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

%% Switch from midpoint to lower grid index (exponential Policyalt)
adjustalt=(Policyalt(5,:,:,:,:)<1+n2short+1);
Policyalt(4,:,:,:,:)=Policyalt(4,:,:,:,:)-adjustalt;
Policyalt(5,:,:,:,:)=adjustalt.*Policyalt(5,:,:,:,:)+(1-adjustalt).*(Policyalt(5,:,:,:,:)-n2short-1);

Policyalt=[Policyalt; PolicyL2flagalt];


end
