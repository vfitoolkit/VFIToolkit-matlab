function [V,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteSemiExoN_GI1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic + ExperienceAssete + SemiExo, GridInterpLayer (GI1, with d1).
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (optional), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, e, ...)   (depends on current e; not on z or semiz)
%
% Naive QH dual pass over the same GI argmax axis the exponential SemiExo ze GI1 raw maxes over:
%   Valt/Policyalt maximise  F + beta*EV        (the exponential value)
%   V/Policy       maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Valt (the exponential continuation value).
% Policy/Policyalt both carry the GI midpoint (row 4) + L2 index (row 5) + L2flag (appended row 6).
%
% lowmemory levels {0,1,2,3} implemented (shocks: z markov + semiz + e iid).

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
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
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray');
Valt=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policyalt=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray');
PolicyaltL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Per-d3 workspaces (alt=exponential @beta, tilde=QH-perceived @beta0beta)
V_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_alt=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray');
flag_ford3_alt=2*ones(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_tilde=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray');
flag_ford3_tilde=2*ones(N_a,N_bothz,N_e,N_d3,'gpuArray');

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-3);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
eind=shiftdim(gpuArray(0:1:N_e-1),-2);

bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % Terminal period: no continuation, so QH-perceived value equals exponential value
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            [~,maxindex]=max(ReturnMatrix_d3,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_alt(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint(d_ind+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eind)),-1);
            Policy4_ford3_alt(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset=ceil(maxindexL2/N_d12);
            linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind+N_d12*n2long*N_a*N_bothz*eind;
            linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind+N_d12*n2long*N_a*N_bothz*eind;
            isInfLower=(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_alt(:,:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
                [~,maxindex]=max(ReturnMatrix_d3e,[],2);

                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_alt(:,:,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
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
                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);
                    [~,maxindex]=max(ReturnMatrix_d3e,[],2);

                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset=ceil(maxindexL2/N_d12);
                    linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                    linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
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
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                    [~,maxindex]=max(ReturnMatrix_d3ze,[],2);

                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3ze,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset=ceil(maxindexL2/N_d12);
                    linidx_lower=d_ind+N_d12*n2long*aind;
                    linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind;
                    isInfLower=(ReturnMatrix_ii_d3ze(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_d3ze(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
                end
            end
        end
    end
    % Max over d3 and unpack (alt = exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policyalt(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(2,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]);
    PolicyaltL2flag(1,:,:,:,N_j)=reshape(flag_ford3_alt((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    % Terminal: perceived == exponential
    V(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy(:,:,:,:,N_j)=Policyalt(:,:,:,:,N_j);
    PolicyL2flag(:,:,:,:,N_j)=PolicyaltL2flag(:,:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex=reshape(aprimeIndex,[N_d2*N_a1,N_a2,1,N_e]);
    aprimeplus1Index=reshape(aprimeplus1Index,[N_d2*N_a1,N_a2,1,N_e]);
    aprimeProbs_d2a1a2e=reshape(aprimeProbs_d2a1a2e,[N_d2*N_a1,N_a2,1,N_e]);

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

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

            lin_lower=aprimeIndex+bothz_offset;
            lin_upper=aprimeplus1Index+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_alt=beta*EVbase;
            DiscountedEV_tilde=beta0beta*EVbase;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEV_alt=repelem(DiscountedEV_alt,N_d1,1);
            DiscountedEV_tilde=repelem(DiscountedEV_tilde,N_d1,1);
            DiscountedEVinterp_alt=repelem(DiscountedEVinterp_alt,N_d1,1);
            DiscountedEVinterp_tilde=repelem(DiscountedEVinterp_tilde,N_d1,1);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            % alt (exponential): F + beta*EV
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_alt;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d12a1primea2bothze=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*bothzind+N_d12*N_a1prime*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
            entireRHS_ii_alt=ReturnMatrix_ii_alt+reshape(DiscountedEVinterp_alt(d12a1primea2bothze(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_alt(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint(d_ind+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eind)),-1);
            Policy4_ford3_alt(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset=ceil(maxindexL2/N_d12);
            linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind+N_d12*n2long*N_a*N_bothz*eind;
            linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind+N_d12*n2long*N_a*N_bothz*eind;
            isInfLower=(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_alt(:,:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);

            % tilde (QH-perceived): F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_tilde;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d12a1primea2bothze=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*bothzind+N_d12*N_a1prime*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
            entireRHS_ii_tilde=ReturnMatrix_ii_tilde+reshape(DiscountedEVinterp_tilde(d12a1primea2bothze(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_tilde(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint(d_ind+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eind)),-1);
            Policy4_ford3_tilde(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset=ceil(maxindexL2/N_d12);
            linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind+N_d12*n2long*N_a*N_bothz*eind;
            linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind+N_d12*n2long*N_a*N_bothz*eind;
            isInfLower=(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_tilde(:,:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex+bothz_offset;
            lin_upper=aprimeplus1Index+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_alt=beta*EVbase;
            DiscountedEV_tilde=beta0beta*EVbase;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEV_alt=repelem(DiscountedEV_alt,N_d1,1);
            DiscountedEV_tilde=repelem(DiscountedEV_tilde,N_d1,1);
            DiscountedEVinterp_alt=repelem(DiscountedEVinterp_alt,N_d1,1);
            DiscountedEVinterp_tilde=repelem(DiscountedEVinterp_tilde,N_d1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                % alt (exponential): F + beta*EV
                DiscountedEV_alt_e=DiscountedEV_alt(:,:,:,:,:,e_c);
                DiscountedEVinterp_alt_e=DiscountedEVinterp_alt(:,:,:,:,:,e_c);
                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_alt_e;
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d12a1primea2bothz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*shiftdim((0:1:N_bothz-1),-3);
                entireRHS_ii_alt=ReturnMatrix_ii_alt+reshape(DiscountedEVinterp_alt_e(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_alt(:,:,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);

                % tilde (QH-perceived): F + beta0*beta*EV
                DiscountedEV_tilde_e=DiscountedEV_tilde(:,:,:,:,:,e_c);
                DiscountedEVinterp_tilde_e=DiscountedEVinterp_tilde(:,:,:,:,:,e_c);
                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_tilde_e;
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d12a1primea2bothz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*shiftdim((0:1:N_bothz-1),-3);
                entireRHS_ii_tilde=ReturnMatrix_ii_tilde+reshape(DiscountedEVinterp_tilde_e(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_tilde(:,:,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
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
            EV1=EV_2D(aprimeIndex+bothz_offset);
            EV2=EV_2D(aprimeplus1Index+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_alt=beta*EVbase;
            DiscountedEV_tilde=beta0beta*EVbase;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEV_alt=repelem(DiscountedEV_alt,N_d1,1);
            DiscountedEV_tilde=repelem(DiscountedEV_tilde,N_d1,1);
            DiscountedEVinterp_alt=repelem(DiscountedEVinterp_alt,N_d1,1);
            DiscountedEVinterp_tilde=repelem(DiscountedEVinterp_tilde,N_d1,1);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                DiscountedEV_alt_zb=DiscountedEV_alt(:,:,:,:,semizblock,:);
                DiscountedEVinterp_alt_zb=DiscountedEVinterp_alt(:,:,:,:,semizblock,:);
                DiscountedEV_tilde_zb=DiscountedEV_tilde(:,:,:,:,semizblock,:);
                DiscountedEVinterp_tilde_zb=DiscountedEVinterp_tilde(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    % alt (exponential): F + beta*EV
                    DiscountedEV_alt_zbe=DiscountedEV_alt_zb(:,:,:,:,:,e_c);
                    DiscountedEVinterp_alt_zbe=DiscountedEVinterp_alt_zb(:,:,:,:,:,e_c);
                    entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_alt_zbe;
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d12a1primea2bothz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*shiftdim((0:1:N_semiz-1),-3);
                    entireRHS_ii_alt=ReturnMatrix_ii_alt+reshape(DiscountedEVinterp_alt_zbe(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset=ceil(maxindexL2/N_d12);
                    linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                    linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);

                    % tilde (QH-perceived): F + beta0*beta*EV
                    DiscountedEV_tilde_zbe=DiscountedEV_tilde_zb(:,:,:,:,:,e_c);
                    DiscountedEVinterp_tilde_zbe=DiscountedEVinterp_tilde_zb(:,:,:,:,:,e_c);
                    entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_tilde_zbe;
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d12a1primea2bothz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*shiftdim((0:1:N_semiz-1),-3);
                    entireRHS_ii_tilde=ReturnMatrix_ii_tilde+reshape(DiscountedEVinterp_tilde_zbe(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,semizblock,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_tilde(2,:,semizblock,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_tilde(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_tilde(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset=ceil(maxindexL2/N_d12);
                    linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                    linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
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
            EV1=EV_2D(aprimeIndex+bothz_offset);
            EV2=EV_2D(aprimeplus1Index+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_alt=beta*EVbase;
            DiscountedEV_tilde=beta0beta*EVbase;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    % alt (exponential): F + beta*EV
                    DiscountedEV_alt_ze=repelem(DiscountedEV_alt(:,:,:,:,z_c,e_c),N_d1,1);
                    DiscountedEVinterp_alt_ze=repelem(DiscountedEVinterp_alt(:,:,:,:,z_c,e_c),N_d1,1);
                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_alt_ze;
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d12a1primea2=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind;
                    entireRHS_ii_alt=ReturnMatrix_ii_alt+reshape(DiscountedEVinterp_alt_ze(d12a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset=ceil(maxindexL2/N_d12);
                    linidx_lower=d_ind+N_d12*n2long*aind;
                    linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind;
                    isInfLower=(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);

                    % tilde (QH-perceived): F + beta0*beta*EV
                    DiscountedEV_tilde_ze=repelem(DiscountedEV_tilde(:,:,:,:,z_c,e_c),N_d1,1);
                    DiscountedEVinterp_tilde_ze=repelem(DiscountedEVinterp_tilde(:,:,:,:,z_c,e_c),N_d1,1);
                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_tilde_ze;
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d12a1primea2=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind;
                    entireRHS_ii_tilde=ReturnMatrix_ii_tilde+reshape(DiscountedEVinterp_tilde_ze(d12a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset=ceil(maxindexL2/N_d12);
                    linidx_lower=d_ind+N_d12*n2long*aind;
                    linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind;
                    isInfLower=(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
                end
            end
        end
    end

    % Max over d3 and unpack (alt = exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policyalt(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(2,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]);
    PolicyaltL2flag(1,:,:,:,N_j)=reshape(flag_ford3_alt((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);

    % Max over d3 and unpack (tilde = QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_tilde((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
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
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex+bothz_offset;
            lin_upper=aprimeplus1Index+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_alt=beta*EVbase;
            DiscountedEV_tilde=beta0beta*EVbase;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEV_alt=repelem(DiscountedEV_alt,N_d1,1);
            DiscountedEV_tilde=repelem(DiscountedEV_tilde,N_d1,1);
            DiscountedEVinterp_alt=repelem(DiscountedEVinterp_alt,N_d1,1);
            DiscountedEVinterp_tilde=repelem(DiscountedEVinterp_tilde,N_d1,1);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            % alt (exponential): F + beta*EV
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_alt;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d12a1primea2bothze=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*bothzind+N_d12*N_a1prime*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
            entireRHS_ii_alt=ReturnMatrix_ii_alt+reshape(DiscountedEVinterp_alt(d12a1primea2bothze(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_alt(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint(d_ind+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eind)),-1);
            Policy4_ford3_alt(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset=ceil(maxindexL2/N_d12);
            linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind+N_d12*n2long*N_a*N_bothz*eind;
            linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind+N_d12*n2long*N_a*N_bothz*eind;
            isInfLower=(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_alt(:,:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);

            % tilde (QH-perceived): F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_tilde;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d12a1primea2bothze=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*bothzind+N_d12*N_a1prime*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
            entireRHS_ii_tilde=ReturnMatrix_ii_tilde+reshape(DiscountedEVinterp_tilde(d12a1primea2bothze(:)),[N_d12*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_tilde(3,:,:,:,d3_c)=shiftdim(squeeze(midpoint(d_ind+N_d12*aind+N_d12*N_a*bothzBind+N_d12*N_a*N_bothz*eind)),-1);
            Policy4_ford3_tilde(4,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset=ceil(maxindexL2/N_d12);
            linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind+N_d12*n2long*N_a*N_bothz*eind;
            linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind+N_d12*n2long*N_a*N_bothz*eind;
            isInfLower=(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_tilde(:,:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex+bothz_offset;
            lin_upper=aprimeplus1Index+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_alt=beta*EVbase;
            DiscountedEV_tilde=beta0beta*EVbase;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEV_alt=repelem(DiscountedEV_alt,N_d1,1);
            DiscountedEV_tilde=repelem(DiscountedEV_tilde,N_d1,1);
            DiscountedEVinterp_alt=repelem(DiscountedEVinterp_alt,N_d1,1);
            DiscountedEVinterp_tilde=repelem(DiscountedEVinterp_tilde,N_d1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                % alt (exponential): F + beta*EV
                DiscountedEV_alt_e=DiscountedEV_alt(:,:,:,:,:,e_c);
                DiscountedEVinterp_alt_e=DiscountedEVinterp_alt(:,:,:,:,:,e_c);
                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_alt_e;
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d12a1primea2bothz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*shiftdim((0:1:N_bothz-1),-3);
                entireRHS_ii_alt=ReturnMatrix_ii_alt+reshape(DiscountedEVinterp_alt_e(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_alt(:,:,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);

                % tilde (QH-perceived): F + beta0*beta*EV
                DiscountedEV_tilde_e=DiscountedEV_tilde(:,:,:,:,:,e_c);
                DiscountedEVinterp_tilde_e=DiscountedEVinterp_tilde(:,:,:,:,:,e_c);
                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_tilde_e;
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d12a1primea2bothz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*shiftdim((0:1:N_bothz-1),-3);
                entireRHS_ii_tilde=ReturnMatrix_ii_tilde+reshape(DiscountedEVinterp_tilde_e(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_tilde(:,:,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
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
            EV1=EV_2D(aprimeIndex+bothz_offset);
            EV2=EV_2D(aprimeplus1Index+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_alt=beta*EVbase;
            DiscountedEV_tilde=beta0beta*EVbase;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEV_alt=repelem(DiscountedEV_alt,N_d1,1);
            DiscountedEV_tilde=repelem(DiscountedEV_tilde,N_d1,1);
            DiscountedEVinterp_alt=repelem(DiscountedEVinterp_alt,N_d1,1);
            DiscountedEVinterp_tilde=repelem(DiscountedEVinterp_tilde,N_d1,1);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                DiscountedEV_alt_zb=DiscountedEV_alt(:,:,:,:,semizblock,:);
                DiscountedEVinterp_alt_zb=DiscountedEVinterp_alt(:,:,:,:,semizblock,:);
                DiscountedEV_tilde_zb=DiscountedEV_tilde(:,:,:,:,semizblock,:);
                DiscountedEVinterp_tilde_zb=DiscountedEVinterp_tilde(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    % alt (exponential): F + beta*EV
                    DiscountedEV_alt_zbe=DiscountedEV_alt_zb(:,:,:,:,:,e_c);
                    DiscountedEVinterp_alt_zbe=DiscountedEVinterp_alt_zb(:,:,:,:,:,e_c);
                    entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_alt_zbe;
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d12a1primea2bothz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*shiftdim((0:1:N_semiz-1),-3);
                    entireRHS_ii_alt=ReturnMatrix_ii_alt+reshape(DiscountedEVinterp_alt_zbe(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_alt(1,:,semizblock,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,semizblock,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_alt(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset=ceil(maxindexL2/N_d12);
                    linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                    linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);

                    % tilde (QH-perceived): F + beta0*beta*EV
                    DiscountedEV_tilde_zbe=DiscountedEV_tilde_zb(:,:,:,:,:,e_c);
                    DiscountedEVinterp_tilde_zbe=DiscountedEVinterp_tilde_zb(:,:,:,:,:,e_c);
                    entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_tilde_zbe;
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d12a1primea2bothz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*shiftdim((0:1:N_semiz-1),-3);
                    entireRHS_ii_tilde=ReturnMatrix_ii_tilde+reshape(DiscountedEVinterp_tilde_zbe(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,semizblock,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_tilde(2,:,semizblock,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_tilde(3,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_tilde(4,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset=ceil(maxindexL2/N_d12);
                    linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                    linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
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
            EV1=EV_2D(aprimeIndex+bothz_offset);
            EV2=EV_2D(aprimeplus1Index+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_alt=beta*EVbase;
            DiscountedEV_tilde=beta0beta*EVbase;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    % alt (exponential): F + beta*EV
                    DiscountedEV_alt_ze=repelem(DiscountedEV_alt(:,:,:,:,z_c,e_c),N_d1,1);
                    DiscountedEVinterp_alt_ze=repelem(DiscountedEVinterp_alt(:,:,:,:,z_c,e_c),N_d1,1);
                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_alt_ze;
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d12a1primea2=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind;
                    entireRHS_ii_alt=ReturnMatrix_ii_alt+reshape(DiscountedEVinterp_alt_ze(d12a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset=ceil(maxindexL2/N_d12);
                    linidx_lower=d_ind+N_d12*n2long*aind;
                    linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind;
                    isInfLower=(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);

                    % tilde (QH-perceived): F + beta0*beta*EV
                    DiscountedEV_tilde_ze=repelem(DiscountedEV_tilde(:,:,:,:,z_c,e_c),N_d1,1);
                    DiscountedEVinterp_tilde_ze=repelem(DiscountedEVinterp_tilde(:,:,:,:,z_c,e_c),N_d1,1);
                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_tilde_ze;
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d12a1primea2=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind;
                    entireRHS_ii_tilde=ReturnMatrix_ii_tilde+reshape(DiscountedEVinterp_tilde_ze(d12a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d12)+1;
                    allind=d_ind+N_d12*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1;
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1);
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                    L2offset=ceil(maxindexL2/N_d12);
                    linidx_lower=d_ind+N_d12*n2long*aind;
                    linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind;
                    isInfLower=(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
                end
            end
        end
    end

    % Max over d3 and unpack (alt = exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,jj)=V_jj;
    Policyalt(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policyalt(1,:,:,:,jj)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(2,:,:,:,jj)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(4,:,:,:,jj)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]);
    Policyalt(5,:,:,:,jj)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]);
    PolicyaltL2flag(1,:,:,:,jj)=reshape(flag_ford3_alt((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);

    % Max over d3 and unpack (tilde = QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    V(:,:,:,jj)=V_jj;
    Policy(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(2,:,:,:,jj)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]);
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_tilde((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);
Policy=[Policy; PolicyL2flag];

adjust_alt=(Policyalt(5,:,:,:,:)<1+n2short+1);
Policyalt(4,:,:,:,:)=Policyalt(4,:,:,:,:)-adjust_alt;
Policyalt(5,:,:,:,:)=adjust_alt.*Policyalt(5,:,:,:,:)+(1-adjust_alt).*(Policyalt(5,:,:,:,:)-n2short-1);
Policyalt=[Policyalt; PolicyaltL2flag];


end
