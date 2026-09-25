function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetsemizN_DC1_GI1_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Naive quasi-hyperbolic discounting + ExperienceAsset driven by semiz, Divide-and-Conquer (DC1 over a1prime) with grid interpolation layer (GI1) on a1.
% d2 determines the experience asset a2, a1 is the standard endogenous state.
% d3 determines the semi-exogenous state semiz.
% Naive: two passes over the same candidate set,
%   Valt/Policyalt maximise  F + beta*EV        (the exponential value)
%   Vtilde/Policy  maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0 is received as a trailing input.
% The two discount factors generally pick different DC brackets AND different GI midpoints, so the beta
% pass uses maxgap_V and the beta0*beta pass uses maxgap, and each pass re-derives its own loweredge,
% midpoint, a1primeindexes(fine) and level-2 return matrix; the level-1 return matrix is shared.
% Each pass also gets its own max over d3.
% Backward EV uses Valt (the exponential continuation value).
% The a2prime lottery is semiz-dependent (aprimeFn=aprimeFn(d2,a2,semiz,...)), so a2primeIndex/
% a2primeProbs carry the semiz dimension; that is resolved inside EV before any discounting, so the
% dual touches only the discount factors, not the lottery.
% d2 determines experience asset, d3 determines semi-exog state (no d1)
% a1 is standard endogenous state, a2 is experience asset (z-dependent aprimeFn)
% z is exogenous markov state (required), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, semiz, ...)
% DC + GI splice (no L2flag scaffold).

n_bothz=n_semiz;

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_bothz=N_semiz;
N_e=prod(n_e);

Valt=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray');
Policy(5,:,:,:,:)=2; % L2 flag: 1=all to lower, 2=usual, 3=all to upper
Policyalt=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray'); % exponential discounter optimal [d; a1prime midpoint; a1primeL2ind]
Policyalt(5,:,:,:,:)=2; % L2 flag for the exponential discounter policy

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=semiz_gridvals_J;

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_bothz=ones(1,length(n_semiz));
end

% Preallocate
if vfoptions.lowmemory==0
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_bothz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_bothz,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint=zeros(N_d2,1,N_a1,N_a2,'gpuArray');
end

V_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy3_ford3_alt=zeros(3,N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy3_ford3_tilde=zeros(3,N_a,N_bothz,N_e,N_d3,'gpuArray');
flag_ford3_alt=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray'); % L2 flag per d3, aggregated after d3 max
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
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3_alt(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_alt(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford3_alt(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
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
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_alt(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_alt(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d2);
                linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end

    elseif vfoptions.lowmemory==2

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
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3_alt(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_alt(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                  + N_d2*n2long*aind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Vtilde(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=3*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy3_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy3_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy3_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    Policy(5,:,:,:,N_j)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz,N_e]);
    % terminal: QH and exponential discounter coincide
    Valt(:,:,:,N_j)=Vtilde(:,:,:,N_j);
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j);
    Policyalt(5,:,:,:,N_j)=Policy(5,:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2semiz=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=aprimeIndex;
    aprimeplus1Index_full=aprimeplus1Index;
    aprimeProbs_full=aprimeProbs_d2a1a2semiz;

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

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
            entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);

            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            %% Valt (beta)

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_alt; % broadcasts over e

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            maxgap_V=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap_V(ii));
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]); % midpoint (hence gather) is e-dependent at lowmemory=0
            [Vtempii,maxindexL2alt]=max(entireRHS_ii_d3,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_indalt=rem(maxindexL2alt-1,N_d2)+1;
            allindalt=d_indalt+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3_alt(1,:,:,:,d3_c)=d_indalt;
            Policy3_ford3_alt(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allindalt)),-1);
            Policy3_ford3_alt(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2alt/N_d2),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offsetalt = ceil(maxindexL2alt/N_d2);
            linidx_loweralt = d_indalt                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upperalt = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
            isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
            inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
            inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)),-1);

            %% Vtilde (beta0*beta)

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_tilde; % broadcasts over e

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]); % midpoint (hence gather) is e-dependent at lowmemory=0
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3_tilde(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_tilde(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford3_tilde(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_tilde(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end

    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

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
            entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);

            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                %% Valt (beta)

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+DiscountedEV_alt;

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap_V=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap_V(ii));
                        a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii_d3e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2alt]=max(entireRHS_ii_d3e,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_indalt=rem(maxindexL2alt-1,N_d2)+1;
                allindalt=d_indalt+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_alt(1,:,:,e_c,d3_c)=d_indalt;
                Policy3_ford3_alt(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allindalt)),-1);
                Policy3_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2alt/N_d2),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offsetalt = ceil(maxindexL2alt/N_d2);
                linidx_loweralt = d_indalt                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upperalt = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
                isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
                inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
                inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)),-1);

                %% Vtilde (beta0*beta)

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+DiscountedEV_tilde;

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii_d3e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_tilde(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_tilde(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3_tilde(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d2);
                linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_tilde(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end

    elseif vfoptions.lowmemory==2

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

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
            entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);

            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,z_c);
                DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,z_c);
                DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,z_c);
                DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    %% Valt (beta)

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_alt_z;

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;

                    maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_V(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap_V(ii));
                            a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_dc+DiscountedEV_alt_z(d2aprime);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt_z(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2alt]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_indalt=rem(maxindexL2alt-1,N_d2)+1;
                    allindalt=d_indalt+N_d2*aind;
                    Policy3_ford3_alt(1,:,z_c,e_c,d3_c)=d_indalt;
                    Policy3_ford3_alt(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allindalt)),-1);
                    Policy3_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2alt/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offsetalt = ceil(maxindexL2alt/N_d2);
                    linidx_loweralt = d_indalt                  + N_d2*n2long*aind;
                    linidx_upperalt = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
                    isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
                    inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
                    inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)),-1);

                    %% Vtilde (beta0*beta)

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_tilde_z;

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(d2aprime);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde_z(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3_tilde(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_tilde(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_tilde(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                  + N_d2*n2long*aind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_tilde(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    Vtilde(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=3*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy3_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy3_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy3_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    Policy(5,:,:,:,N_j)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_bothz,N_e]);

    % Max over d3 for alt (the exponential discounter) -> Valt/Policyalt
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jjalt;
    Policyalt(2,:,:,:,N_j)=shiftdim(maxindexalt,-1);
    maxindexalt=reshape(maxindexalt,[N_a*N_bothz*N_e,1]);
    tempalt=3*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1) -1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy3_ford3_alt(1+tempalt),[1,N_a,N_bothz,N_e]);
    Policyalt(3,:,:,:,N_j)=reshape(Policy3_ford3_alt(2+tempalt),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,N_j)=reshape(Policy3_ford3_alt(3+tempalt),[1,N_a,N_bothz,N_e]);
    flat_idxalt=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1);
    Policyalt(5,:,:,:,N_j)=reshape(flag_ford3_alt(flat_idxalt),[1,N_a,N_bothz,N_e]);
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
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2semiz=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=aprimeIndex;
    aprimeplus1Index_full=aprimeplus1Index;
    aprimeProbs_full=aprimeProbs_d2a1a2semiz;

    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

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
            entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);

            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            % First layer
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            %% Valt (beta)

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_alt;

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            % Second layer of DC
            maxgap_V=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap_V(ii));
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimez);
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
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]); % midpoint (hence gather) is e-dependent at lowmemory=0
            [Vtempii,maxindexL2alt]=max(entireRHS_ii_d3,[],1);

            % Store each d3 (later max over d3)
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_indalt=rem(maxindexL2alt-1,N_d2)+1;
            allindalt=d_indalt+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3_alt(1,:,:,:,d3_c)=d_indalt;
            Policy3_ford3_alt(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allindalt)),-1);
            Policy3_ford3_alt(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2alt/N_d2),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offsetalt = ceil(maxindexL2alt/N_d2);
            linidx_loweralt = d_indalt                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upperalt = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
            isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
            inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
            inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
            flag_ford3_alt(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)),-1);

            %% Vtilde (beta0*beta)

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_tilde;

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            % Second layer of DC
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimez);
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
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]); % midpoint (hence gather) is e-dependent at lowmemory=0
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);

            % Store each d3 (later max over d3)
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3_tilde(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_tilde(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford3_tilde(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_tilde(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end

    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

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
            entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);

            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                %% Valt (beta)

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+DiscountedEV_alt;

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap_V=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap_V(ii));
                        a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii_d3e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2alt]=max(entireRHS_ii_d3e,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_indalt=rem(maxindexL2alt-1,N_d2)+1;
                allindalt=d_indalt+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_alt(1,:,:,e_c,d3_c)=d_indalt;
                Policy3_ford3_alt(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allindalt)),-1);
                Policy3_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2alt/N_d2),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offsetalt = ceil(maxindexL2alt/N_d2);
                linidx_loweralt = d_indalt                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upperalt = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
                isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
                inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
                inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
                flag_ford3_alt(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)),-1);

                %% Vtilde (beta0*beta)

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+DiscountedEV_tilde;

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii_d3e=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii_d3e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                entireRHS_ii_d3e=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_tilde(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_tilde(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3_tilde(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d2);
                linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_tilde(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end

    elseif vfoptions.lowmemory==2

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

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
            entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);

            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,z_c);
                DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,z_c);
                DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,z_c);
                DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    %% Valt (beta)

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_alt_z;

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;

                    maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap_V(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap_V(ii));
                            a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_dc+DiscountedEV_alt_z(d2aprime);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt_z(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2alt]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_indalt=rem(maxindexL2alt-1,N_d2)+1;
                    allindalt=d_indalt+N_d2*aind;
                    Policy3_ford3_alt(1,:,z_c,e_c,d3_c)=d_indalt;
                    Policy3_ford3_alt(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allindalt)),-1);
                    Policy3_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2alt/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offsetalt = ceil(maxindexL2alt/N_d2);
                    linidx_loweralt = d_indalt                  + N_d2*n2long*aind;
                    linidx_upperalt = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
                    isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
                    inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
                    inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
                    flag_ford3_alt(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)),-1);

                    %% Vtilde (beta0*beta)

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+DiscountedEV_tilde_z;

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii_d3ze=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(d2aprime);
                            [~,maxindex]=max(entireRHS_ii_d3ze,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_d3ze=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde_z(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3_tilde(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_tilde(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_tilde(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                  + N_d2*n2long*aind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3_tilde(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    Vtilde(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=3*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,jj)=reshape(Policy3_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,jj)=reshape(Policy3_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy3_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    Policy(5,:,:,:,jj)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_bothz,N_e]);

    % Max over d3 for alt (the exponential discounter) -> Valt/Policyalt
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],4);
    Valt(:,:,:,jj)=V_jjalt;
    Policyalt(2,:,:,:,jj)=shiftdim(maxindexalt,-1);
    maxindexalt=reshape(maxindexalt,[N_a*N_bothz*N_e,1]);
    tempalt=3*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1) -1);
    Policyalt(1,:,:,:,jj)=reshape(Policy3_ford3_alt(1+tempalt),[1,N_a,N_bothz,N_e]);
    Policyalt(3,:,:,:,jj)=reshape(Policy3_ford3_alt(2+tempalt),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,jj)=reshape(Policy3_ford3_alt(3+tempalt),[1,N_a,N_bothz,N_e]);
    flat_idxalt=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1);
    Policyalt(5,:,:,:,jj)=reshape(flag_ford3_alt(flat_idxalt),[1,N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(4,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-(n2short+1)*(~adjust);

adjustalt=(Policyalt(4,:,:,:,:)<1+n2short+1);
Policyalt(3,:,:,:,:)=Policyalt(3,:,:,:,:)-adjustalt;
Policyalt(4,:,:,:,:)=Policyalt(4,:,:,:,:)-(n2short+1)*(~adjustalt);


end
