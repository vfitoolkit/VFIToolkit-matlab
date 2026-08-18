function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoN_GI1_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state (no d1)
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state
% aprimeFn = aprimeFn(d2, a2, z, ...)

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;

Valt=zeros(N_a,N_bothz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_j,'gpuArray');
% Policy storage with d2, d3, a1prime_midpoint, a1primeL2ind
Policyalt=zeros(4,N_a,N_bothz,N_j,'gpuArray');
Policy=zeros(4,N_a,N_bothz,N_j,'gpuArray');
PolicyL2flagalt=2*ones(1,N_a,N_bothz,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_bothz,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

V_ford3_alt=zeros(N_a,N_bothz,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy3_ford3_alt=zeros(3,N_a,N_bothz,N_d3,'gpuArray');
Policy3_ford3_tilde=zeros(3,N_a,N_bothz,N_d3,'gpuArray');
flag_ford3_alt=2*ones(N_a,N_bothz,N_d3,'gpuArray');
flag_ford3_tilde=2*ones(N_a,N_bothz,N_d3,'gpuArray');

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-3);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);

bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            [~,maxindex]=max(ReturnMatrix_d3,[],2);

            
            midpoint_alt=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3,[],1);

            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
            Policy3_ford3_alt(1,:,:,d3_c)=d_ind;
            Policy3_ford3_alt(2,:,:,d3_c)=shiftdim(squeeze(midpoint_alt(allind)),-1);
            Policy3_ford3_alt(3,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            L2offset=ceil(maxindexL2/N_d2);
            linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
            linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
            isInfLower=(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_alt(:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);
                [~,maxindex]=max(ReturnMatrix_d3z,[],2);

                midpoint_alt=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3z,[],1);

                V_ford3_alt(:,semizblock,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*semizBind;
                Policy3_ford3_alt(1,:,semizblock,d3_c)=d_ind;
                Policy3_ford3_alt(2,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_alt(allind)),-1);
                Policy3_ford3_alt(3,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                L2offset=ceil(maxindexL2/N_d2);
                linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                isInfLower=(ReturnMatrix_ii_d3z(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_d3z(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_alt(:,semizblock,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                [~,maxindex]=max(ReturnMatrix_d3z,[],2);

                midpoint_alt=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3z,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind;
                Policy3_ford3_alt(1,:,z_c,d3_c)=d_ind;
                Policy3_ford3_alt(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind)),-1);
                Policy3_ford3_alt(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                L2offset=ceil(maxindexL2/N_d2);
                linidx_lower=d_ind+N_d2*n2long*aind;
                linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind;
                isInfLower=(ReturnMatrix_ii_d3z(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_d3z(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_alt(:,z_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
            end
        end
    end
    % Max over d3 and unpack
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,N_j)=V_jj;
    Policyalt(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=3*((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)-1);
    Policyalt(1,:,:,N_j)=reshape(Policy3_ford3_alt(1+temp),[1,N_a,N_bothz]); % d2
    Policyalt(3,:,:,N_j)=reshape(Policy3_ford3_alt(2+temp),[1,N_a,N_bothz]); % midpoint_alt
    Policyalt(4,:,:,N_j)=reshape(Policy3_ford3_alt(3+temp),[1,N_a,N_bothz]); % L2
    PolicyL2flagalt(1,:,:,N_j)=reshape(flag_ford3_alt((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy(:,:,:,N_j)=Policyalt(:,:,:,N_j);
    PolicyL2flag(:,:,:,N_j)=PolicyL2flagalt(:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
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

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            % --- alt pass ---

            entireRHS_d3_alt=ReturnMatrix_d3+DiscountedEV_alt;

            [~,maxindex_alt]=max(entireRHS_d3_alt,[],2);

            midpoint_alt=max(min(maxindex_alt,n_a1(1)-1),2);
            a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothz_alt=(1:1:N_d2)'+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii_alt,1);
            d_ind_alt=rem(maxindexL2_alt-1,N_d2)+1;
            allind_alt=d_ind_alt+N_d2*aind+N_d2*N_a*bothzBind;
            Policy3_ford3_alt(1,:,:,d3_c)=d_ind_alt;
            Policy3_ford3_alt(2,:,:,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
            Policy3_ford3_alt(3,:,:,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d2),-1);
            L2offset_alt=ceil(maxindexL2_alt/N_d2);
            linidx_lower_alt=d_ind_alt+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
            linidx_upper_alt=d_ind_alt+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
            isInfLower_alt=(ReturnMatrix_ii_d3_alt(linidx_lower_alt)==-Inf);
            isInfUpper_alt=(ReturnMatrix_ii_d3_alt(linidx_upper_alt)==-Inf);
            inLowerStrict_alt=(L2offset_alt>=2)&(L2offset_alt<=n2short+1);
            inUpperStrict_alt=(L2offset_alt>=n2short+3)&(L2offset_alt<=n2long-1);
            flag_ford3_alt(:,:,d3_c)=shiftdim(2+(inLowerStrict_alt&isInfLower_alt)-(inUpperStrict_alt&isInfUpper_alt),1);

            % --- tilde pass ---

            entireRHS_d3_tilde=ReturnMatrix_d3+DiscountedEV_tilde;

            [~,maxindex_tilde]=max(entireRHS_d3_tilde,[],2);

            midpoint_tilde=max(min(maxindex_tilde,n_a1(1)-1),2);
            a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothz_tilde=(1:1:N_d2)'+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii_tilde,1);
            d_ind_tilde=rem(maxindexL2_tilde-1,N_d2)+1;
            allind_tilde=d_ind_tilde+N_d2*aind+N_d2*N_a*bothzBind;
            Policy3_ford3_tilde(1,:,:,d3_c)=d_ind_tilde;
            Policy3_ford3_tilde(2,:,:,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
            Policy3_ford3_tilde(3,:,:,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d2),-1);
            L2offset_tilde=ceil(maxindexL2_tilde/N_d2);
            linidx_lower_tilde=d_ind_tilde+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
            linidx_upper_tilde=d_ind_tilde+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
            isInfLower_tilde=(ReturnMatrix_ii_d3_tilde(linidx_lower_tilde)==-Inf);
            isInfUpper_tilde=(ReturnMatrix_ii_d3_tilde(linidx_upper_tilde)==-Inf);
            inLowerStrict_tilde=(L2offset_tilde>=2)&(L2offset_tilde<=n2short+1);
            inUpperStrict_tilde=(L2offset_tilde>=n2short+3)&(L2offset_tilde<=n2long-1);
            flag_ford3_tilde(:,:,d3_c)=shiftdim(2+(inLowerStrict_tilde&isInfLower_tilde)-(inUpperStrict_tilde&isInfUpper_tilde),1);
        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3);
                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);

                EV=V_Jplus1.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV(isnan(EV))=0;
                EV=sum(EV,2);
                EV_2D=reshape(EV,[N_a,N_semiz]);

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

                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);

                % --- alt pass ---

                entireRHS_d3z_alt=ReturnMatrix_d3z+DiscountedEV_alt;

                [~,maxindex_alt]=max(entireRHS_d3z_alt,[],2);

                midpoint_alt=max(min(maxindex_alt,n_a1(1)-1),2);
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                d2a1primea2semiz_alt=(1:1:N_d2)'+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+reshape(DiscountedEVinterp_alt(d2a1primea2semiz_alt(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3z_alt,[],1);
                V_ford3_alt(:,semizblock,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt=rem(maxindexL2_alt-1,N_d2)+1;
                allind_alt=d_ind_alt+N_d2*aind+N_d2*N_a*semizBind;
                Policy3_ford3_alt(1,:,semizblock,d3_c)=d_ind_alt;
                Policy3_ford3_alt(2,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                Policy3_ford3_alt(3,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d2),-1);
                L2offset_alt=ceil(maxindexL2_alt/N_d2);
                linidx_lower_alt=d_ind_alt+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                linidx_upper_alt=d_ind_alt+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                isInfLower_alt=(ReturnMatrix_ii_d3z_alt(linidx_lower_alt)==-Inf);
                isInfUpper_alt=(ReturnMatrix_ii_d3z_alt(linidx_upper_alt)==-Inf);
                inLowerStrict_alt=(L2offset_alt>=2)&(L2offset_alt<=n2short+1);
                inUpperStrict_alt=(L2offset_alt>=n2short+3)&(L2offset_alt<=n2long-1);
                flag_ford3_alt(:,semizblock,d3_c)=shiftdim(2+(inLowerStrict_alt&isInfLower_alt)-(inUpperStrict_alt&isInfUpper_alt),1);

                % --- tilde pass ---

                entireRHS_d3z_tilde=ReturnMatrix_d3z+DiscountedEV_tilde;

                [~,maxindex_tilde]=max(entireRHS_d3z_tilde,[],2);

                midpoint_tilde=max(min(maxindex_tilde,n_a1(1)-1),2);
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                d2a1primea2semiz_tilde=(1:1:N_d2)'+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2semiz_tilde(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3z_tilde,[],1);
                V_ford3_tilde(:,semizblock,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde=rem(maxindexL2_tilde-1,N_d2)+1;
                allind_tilde=d_ind_tilde+N_d2*aind+N_d2*N_a*semizBind;
                Policy3_ford3_tilde(1,:,semizblock,d3_c)=d_ind_tilde;
                Policy3_ford3_tilde(2,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                Policy3_ford3_tilde(3,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d2),-1);
                L2offset_tilde=ceil(maxindexL2_tilde/N_d2);
                linidx_lower_tilde=d_ind_tilde+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                linidx_upper_tilde=d_ind_tilde+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                isInfLower_tilde=(ReturnMatrix_ii_d3z_tilde(linidx_lower_tilde)==-Inf);
                isInfUpper_tilde=(ReturnMatrix_ii_d3z_tilde(linidx_upper_tilde)==-Inf);
                inLowerStrict_tilde=(L2offset_tilde>=2)&(L2offset_tilde<=n2short+1);
                inUpperStrict_tilde=(L2offset_tilde>=n2short+3)&(L2offset_tilde<=n2long-1);
                flag_ford3_tilde(:,semizblock,d3_c)=shiftdim(2+(inLowerStrict_tilde&isInfLower_tilde)-(inUpperStrict_tilde&isInfUpper_tilde),1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
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
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);

            % --- alt pass ---
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,z_c);
                DiscountedEVinterp_z_alt=DiscountedEVinterp_alt(:,:,:,:,z_c);


                entireRHS_d3z_alt=ReturnMatrix_d3z+DiscountedEV_z_alt;

                [~,maxindex_alt]=max(entireRHS_d3z_alt,[],2);

                midpoint_alt=max(min(maxindex_alt,n_a1(1)-1),2);
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                d2a1primea2_alt=(1:1:N_d2)'+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+reshape(DiscountedEVinterp_z_alt(d2a1primea2_alt(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3z_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt=rem(maxindexL2_alt-1,N_d2)+1;
                allind_alt=d_ind_alt+N_d2*aind;
                Policy3_ford3_alt(1,:,z_c,d3_c)=d_ind_alt;
                Policy3_ford3_alt(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                Policy3_ford3_alt(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d2),-1);
                L2offset_alt=ceil(maxindexL2_alt/N_d2);
                linidx_lower_alt=d_ind_alt+N_d2*n2long*aind;
                linidx_upper_alt=d_ind_alt+N_d2*(n2long-1)+N_d2*n2long*aind;
                isInfLower_alt=(ReturnMatrix_ii_d3z_alt(linidx_lower_alt)==-Inf);
                isInfUpper_alt=(ReturnMatrix_ii_d3z_alt(linidx_upper_alt)==-Inf);
                inLowerStrict_alt=(L2offset_alt>=2)&(L2offset_alt<=n2short+1);
                inUpperStrict_alt=(L2offset_alt>=n2short+3)&(L2offset_alt<=n2long-1);
                flag_ford3_alt(:,z_c,d3_c)=shiftdim(2+(inLowerStrict_alt&isInfLower_alt)-(inUpperStrict_alt&isInfUpper_alt),1);

            % --- tilde pass ---
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,z_c);
                DiscountedEVinterp_z_tilde=DiscountedEVinterp_tilde(:,:,:,:,z_c);


                entireRHS_d3z_tilde=ReturnMatrix_d3z+DiscountedEV_z_tilde;

                [~,maxindex_tilde]=max(entireRHS_d3z_tilde,[],2);

                midpoint_tilde=max(min(maxindex_tilde,n_a1(1)-1),2);
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                d2a1primea2_tilde=(1:1:N_d2)'+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+reshape(DiscountedEVinterp_z_tilde(d2a1primea2_tilde(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3z_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde=rem(maxindexL2_tilde-1,N_d2)+1;
                allind_tilde=d_ind_tilde+N_d2*aind;
                Policy3_ford3_tilde(1,:,z_c,d3_c)=d_ind_tilde;
                Policy3_ford3_tilde(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                Policy3_ford3_tilde(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d2),-1);
                L2offset_tilde=ceil(maxindexL2_tilde/N_d2);
                linidx_lower_tilde=d_ind_tilde+N_d2*n2long*aind;
                linidx_upper_tilde=d_ind_tilde+N_d2*(n2long-1)+N_d2*n2long*aind;
                isInfLower_tilde=(ReturnMatrix_ii_d3z_tilde(linidx_lower_tilde)==-Inf);
                isInfUpper_tilde=(ReturnMatrix_ii_d3z_tilde(linidx_upper_tilde)==-Inf);
                inLowerStrict_tilde=(L2offset_tilde>=2)&(L2offset_tilde<=n2short+1);
                inUpperStrict_tilde=(L2offset_tilde>=n2short+3)&(L2offset_tilde<=n2long-1);
                flag_ford3_tilde(:,z_c,d3_c)=shiftdim(2+(inLowerStrict_tilde&isInfLower_tilde)-(inUpperStrict_tilde&isInfUpper_tilde),1);
            end
        end
    end

    % Max over d3 and unpack
    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,N_j)=V_jj;
    Policyalt(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=3*((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)-1);
    Policyalt(1,:,:,N_j)=reshape(Policy3_ford3_alt(1+temp),[1,N_a,N_bothz]); % d2
    Policyalt(3,:,:,N_j)=reshape(Policy3_ford3_alt(2+temp),[1,N_a,N_bothz]); % midpoint
    Policyalt(4,:,:,N_j)=reshape(Policy3_ford3_alt(3+temp),[1,N_a,N_bothz]); % L2
    PolicyL2flagalt(1,:,:,N_j)=reshape(flag_ford3_alt((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=3*((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)-1);
    Policy(1,:,:,N_j)=reshape(Policy3_ford3_tilde(1+temp),[1,N_a,N_bothz]); % d2
    Policy(3,:,:,N_j)=reshape(Policy3_ford3_tilde(2+temp),[1,N_a,N_bothz]); % midpoint
    Policy(4,:,:,N_j)=reshape(Policy3_ford3_tilde(3+temp),[1,N_a,N_bothz]); % L2
    PolicyL2flag(1,:,:,N_j)=reshape(flag_ford3_tilde((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);

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

    EVpre=Valt(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
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

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            % --- alt pass ---

            entireRHS_d3_alt=ReturnMatrix_d3+DiscountedEV_alt;

            [~,maxindex_alt]=max(entireRHS_d3_alt,[],2);

            midpoint_alt=max(min(maxindex_alt,n_a1(1)-1),2);
            a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothz_alt=(1:1:N_d2)'+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii_alt,1);
            d_ind_alt=rem(maxindexL2_alt-1,N_d2)+1;
            allind_alt=d_ind_alt+N_d2*aind+N_d2*N_a*bothzBind;
            Policy3_ford3_alt(1,:,:,d3_c)=d_ind_alt;
            Policy3_ford3_alt(2,:,:,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
            Policy3_ford3_alt(3,:,:,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d2),-1);
            L2offset_alt=ceil(maxindexL2_alt/N_d2);
            linidx_lower_alt=d_ind_alt+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
            linidx_upper_alt=d_ind_alt+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
            isInfLower_alt=(ReturnMatrix_ii_d3_alt(linidx_lower_alt)==-Inf);
            isInfUpper_alt=(ReturnMatrix_ii_d3_alt(linidx_upper_alt)==-Inf);
            inLowerStrict_alt=(L2offset_alt>=2)&(L2offset_alt<=n2short+1);
            inUpperStrict_alt=(L2offset_alt>=n2short+3)&(L2offset_alt<=n2long-1);
            flag_ford3_alt(:,:,d3_c)=shiftdim(2+(inLowerStrict_alt&isInfLower_alt)-(inUpperStrict_alt&isInfUpper_alt),1);

            % --- tilde pass ---

            entireRHS_d3_tilde=ReturnMatrix_d3+DiscountedEV_tilde;

            [~,maxindex_tilde]=max(entireRHS_d3_tilde,[],2);

            midpoint_tilde=max(min(maxindex_tilde,n_a1(1)-1),2);
            a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothz_tilde=(1:1:N_d2)'+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii_tilde,1);
            d_ind_tilde=rem(maxindexL2_tilde-1,N_d2)+1;
            allind_tilde=d_ind_tilde+N_d2*aind+N_d2*N_a*bothzBind;
            Policy3_ford3_tilde(1,:,:,d3_c)=d_ind_tilde;
            Policy3_ford3_tilde(2,:,:,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
            Policy3_ford3_tilde(3,:,:,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d2),-1);
            L2offset_tilde=ceil(maxindexL2_tilde/N_d2);
            linidx_lower_tilde=d_ind_tilde+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
            linidx_upper_tilde=d_ind_tilde+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
            isInfLower_tilde=(ReturnMatrix_ii_d3_tilde(linidx_lower_tilde)==-Inf);
            isInfUpper_tilde=(ReturnMatrix_ii_d3_tilde(linidx_upper_tilde)==-Inf);
            inLowerStrict_tilde=(L2offset_tilde>=2)&(L2offset_tilde<=n2short+1);
            inUpperStrict_tilde=(L2offset_tilde>=n2short+3)&(L2offset_tilde<=n2long-1);
            flag_ford3_tilde(:,:,d3_c)=shiftdim(2+(inLowerStrict_tilde&isInfLower_tilde)-(inUpperStrict_tilde&isInfUpper_tilde),1);
        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3);
                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV(isnan(EV))=0;
                EV=sum(EV,2);
                EV_2D=reshape(EV,[N_a,N_semiz]);

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

                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);

                % --- alt pass ---

                entireRHS_d3z_alt=ReturnMatrix_d3z+DiscountedEV_alt;

                [~,maxindex_alt]=max(entireRHS_d3z_alt,[],2);

                midpoint_alt=max(min(maxindex_alt,n_a1(1)-1),2);
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                d2a1primea2semiz_alt=(1:1:N_d2)'+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+reshape(DiscountedEVinterp_alt(d2a1primea2semiz_alt(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3z_alt,[],1);
                V_ford3_alt(:,semizblock,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt=rem(maxindexL2_alt-1,N_d2)+1;
                allind_alt=d_ind_alt+N_d2*aind+N_d2*N_a*semizBind;
                Policy3_ford3_alt(1,:,semizblock,d3_c)=d_ind_alt;
                Policy3_ford3_alt(2,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                Policy3_ford3_alt(3,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d2),-1);
                L2offset_alt=ceil(maxindexL2_alt/N_d2);
                linidx_lower_alt=d_ind_alt+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                linidx_upper_alt=d_ind_alt+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                isInfLower_alt=(ReturnMatrix_ii_d3z_alt(linidx_lower_alt)==-Inf);
                isInfUpper_alt=(ReturnMatrix_ii_d3z_alt(linidx_upper_alt)==-Inf);
                inLowerStrict_alt=(L2offset_alt>=2)&(L2offset_alt<=n2short+1);
                inUpperStrict_alt=(L2offset_alt>=n2short+3)&(L2offset_alt<=n2long-1);
                flag_ford3_alt(:,semizblock,d3_c)=shiftdim(2+(inLowerStrict_alt&isInfLower_alt)-(inUpperStrict_alt&isInfUpper_alt),1);

                % --- tilde pass ---

                entireRHS_d3z_tilde=ReturnMatrix_d3z+DiscountedEV_tilde;

                [~,maxindex_tilde]=max(entireRHS_d3z_tilde,[],2);

                midpoint_tilde=max(min(maxindex_tilde,n_a1(1)-1),2);
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                d2a1primea2semiz_tilde=(1:1:N_d2)'+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2semiz_tilde(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3z_tilde,[],1);
                V_ford3_tilde(:,semizblock,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde=rem(maxindexL2_tilde-1,N_d2)+1;
                allind_tilde=d_ind_tilde+N_d2*aind+N_d2*N_a*semizBind;
                Policy3_ford3_tilde(1,:,semizblock,d3_c)=d_ind_tilde;
                Policy3_ford3_tilde(2,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                Policy3_ford3_tilde(3,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d2),-1);
                L2offset_tilde=ceil(maxindexL2_tilde/N_d2);
                linidx_lower_tilde=d_ind_tilde+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                linidx_upper_tilde=d_ind_tilde+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                isInfLower_tilde=(ReturnMatrix_ii_d3z_tilde(linidx_lower_tilde)==-Inf);
                isInfUpper_tilde=(ReturnMatrix_ii_d3z_tilde(linidx_upper_tilde)==-Inf);
                inLowerStrict_tilde=(L2offset_tilde>=2)&(L2offset_tilde<=n2short+1);
                inUpperStrict_tilde=(L2offset_tilde>=n2short+3)&(L2offset_tilde<=n2long-1);
                flag_ford3_tilde(:,semizblock,d3_c)=shiftdim(2+(inLowerStrict_tilde&isInfLower_tilde)-(inUpperStrict_tilde&isInfUpper_tilde),1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
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
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);

            % --- alt pass ---
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,z_c);
                DiscountedEVinterp_z_alt=DiscountedEVinterp_alt(:,:,:,:,z_c);


                entireRHS_d3z_alt=ReturnMatrix_d3z+DiscountedEV_z_alt;

                [~,maxindex_alt]=max(entireRHS_d3z_alt,[],2);

                midpoint_alt=max(min(maxindex_alt,n_a1(1)-1),2);
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                d2a1primea2_alt=(1:1:N_d2)'+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+reshape(DiscountedEVinterp_z_alt(d2a1primea2_alt(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3z_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt=rem(maxindexL2_alt-1,N_d2)+1;
                allind_alt=d_ind_alt+N_d2*aind;
                Policy3_ford3_alt(1,:,z_c,d3_c)=d_ind_alt;
                Policy3_ford3_alt(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                Policy3_ford3_alt(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d2),-1);
                L2offset_alt=ceil(maxindexL2_alt/N_d2);
                linidx_lower_alt=d_ind_alt+N_d2*n2long*aind;
                linidx_upper_alt=d_ind_alt+N_d2*(n2long-1)+N_d2*n2long*aind;
                isInfLower_alt=(ReturnMatrix_ii_d3z_alt(linidx_lower_alt)==-Inf);
                isInfUpper_alt=(ReturnMatrix_ii_d3z_alt(linidx_upper_alt)==-Inf);
                inLowerStrict_alt=(L2offset_alt>=2)&(L2offset_alt<=n2short+1);
                inUpperStrict_alt=(L2offset_alt>=n2short+3)&(L2offset_alt<=n2long-1);
                flag_ford3_alt(:,z_c,d3_c)=shiftdim(2+(inLowerStrict_alt&isInfLower_alt)-(inUpperStrict_alt&isInfUpper_alt),1);

            % --- tilde pass ---
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,z_c);
                DiscountedEVinterp_z_tilde=DiscountedEVinterp_tilde(:,:,:,:,z_c);


                entireRHS_d3z_tilde=ReturnMatrix_d3z+DiscountedEV_z_tilde;

                [~,maxindex_tilde]=max(entireRHS_d3z_tilde,[],2);

                midpoint_tilde=max(min(maxindex_tilde,n_a1(1)-1),2);
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                d2a1primea2_tilde=(1:1:N_d2)'+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+reshape(DiscountedEVinterp_z_tilde(d2a1primea2_tilde(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3z_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde=rem(maxindexL2_tilde-1,N_d2)+1;
                allind_tilde=d_ind_tilde+N_d2*aind;
                Policy3_ford3_tilde(1,:,z_c,d3_c)=d_ind_tilde;
                Policy3_ford3_tilde(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                Policy3_ford3_tilde(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d2),-1);
                L2offset_tilde=ceil(maxindexL2_tilde/N_d2);
                linidx_lower_tilde=d_ind_tilde+N_d2*n2long*aind;
                linidx_upper_tilde=d_ind_tilde+N_d2*(n2long-1)+N_d2*n2long*aind;
                isInfLower_tilde=(ReturnMatrix_ii_d3z_tilde(linidx_lower_tilde)==-Inf);
                isInfUpper_tilde=(ReturnMatrix_ii_d3z_tilde(linidx_upper_tilde)==-Inf);
                inLowerStrict_tilde=(L2offset_tilde>=2)&(L2offset_tilde<=n2short+1);
                inUpperStrict_tilde=(L2offset_tilde>=n2short+3)&(L2offset_tilde<=n2long-1);
                flag_ford3_tilde(:,z_c,d3_c)=shiftdim(2+(inLowerStrict_tilde&isInfLower_tilde)-(inUpperStrict_tilde&isInfUpper_tilde),1);
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,jj)=V_jj;
    Policyalt(2,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=3*((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)-1);
    Policyalt(1,:,:,jj)=reshape(Policy3_ford3_alt(1+temp),[1,N_a,N_bothz]);
    Policyalt(3,:,:,jj)=reshape(Policy3_ford3_alt(2+temp),[1,N_a,N_bothz]);
    Policyalt(4,:,:,jj)=reshape(Policy3_ford3_alt(3+temp),[1,N_a,N_bothz]);
    PolicyL2flagalt(1,:,:,jj)=reshape(flag_ford3_alt((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,jj)=V_jj;
    Policy(2,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=3*((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)-1);
    Policy(1,:,:,jj)=reshape(Policy3_ford3_tilde(1+temp),[1,N_a,N_bothz]);
    Policy(3,:,:,jj)=reshape(Policy3_ford3_tilde(2+temp),[1,N_a,N_bothz]);
    Policy(4,:,:,jj)=reshape(Policy3_ford3_tilde(3+temp),[1,N_a,N_bothz]);
    PolicyL2flag(1,:,:,jj)=reshape(flag_ford3_tilde((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);

end


%% Switch from midpoint to lower grid index
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

adjustalt=(Policyalt(4,:,:,:)<1+n2short+1);
Policyalt(3,:,:,:)=Policyalt(3,:,:,:)-adjustalt;
Policyalt(4,:,:,:)=adjustalt.*Policyalt(4,:,:,:)+(1-adjustalt).*(Policyalt(4,:,:,:)-n2short-1);

Policyalt=[Policyalt; PolicyL2flagalt];


end
