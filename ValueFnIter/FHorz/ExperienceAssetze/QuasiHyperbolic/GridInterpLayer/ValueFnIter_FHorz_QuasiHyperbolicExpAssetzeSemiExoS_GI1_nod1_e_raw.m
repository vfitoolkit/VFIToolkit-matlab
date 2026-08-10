function [V,Policy,Valt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExoS_GI1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic + ExperienceAssetze + SemiExo, GridInterpLayer (GI1, no d1).
% d2 determines experience asset, d3 determines semi-exog state (no d1)
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, z, e, ...)   (depends on BOTH current z and current e)
%
% Sophisticated QH over the same GI argmax axis the exponential SemiExo ze GI1 raw maxes over:
%   V/Policy come from the  F + beta0*beta*EV  argmax (QH-perceived); Policy carries the GI
%     midpoint (row 3) + L2 index (row 4) + L2flag (appended row 5).
%   Valt (=Vunderbar) is the  F + beta*EV  RHS GATHERED at that same GI argmax (NOT re-maximised).
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Valt (Vunderbar).
%
% lowmemory levels {0,1,2,3} implemented (shocks: z markov + semiz + e iid).

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
Policy=zeros(4,N_a,N_bothz,N_e,N_j,'gpuArray'); % (d2, d3, midpoint, L2ind)
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray');
Valt=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Per-d3 workspaces (hat=QH-perceived @beta0beta argmax, under=beta-RHS gathered at that argmax)
V_ford3_hat=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy3_ford3_hat=zeros(3,N_a,N_bothz,N_e,N_d3,'gpuArray'); % (d2, midpoint, L2ind)
flag_ford3_hat=2*ones(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);
Nrow=N_d2*n2long; % row dimension (d2*fine-a1prime) of the fine RHS, for the under gather

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
    % Terminal period: no continuation, so Vunderbar equals Vhat
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            [~,maxindex]=max(ReturnMatrix_d3,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            Policy3_ford3_hat(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_hat(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eind)),-1);
            Policy3_ford3_hat(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            L2offset=ceil(maxindexL2/N_d2);
            linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind+N_d2*n2long*N_a*N_bothz*eind;
            linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind+N_d2*n2long*N_a*N_bothz*eind;
            isInfLower=(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_hat(:,:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
                [~,maxindex]=max(ReturnMatrix_d3e,[],2);

                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_hat(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_hat(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3_hat(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                L2offset=ceil(maxindexL2/N_d2);
                linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
                linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_hat(:,:,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
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
                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);
                    [~,maxindex]=max(ReturnMatrix_d3e,[],2);

                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind+N_d2*N_a*semizBind;
                    Policy3_ford3_hat(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy3_ford3_hat(2,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_hat(3,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    L2offset=ceil(maxindexL2/N_d2);
                    linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                    linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
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
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                    [~,maxindex]=max(ReturnMatrix_d3ze,[],2);

                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3ze,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3_hat(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_hat(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_hat(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    L2offset=ceil(maxindexL2/N_d2);
                    linidx_lower=d_ind+N_d2*n2long*aind;
                    linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind;
                    isInfLower=(ReturnMatrix_ii_d3ze(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_d3ze(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
                end
            end
        end
    end
    % Max over d3 and unpack (hat = QH-perceived)
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=3*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy3_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy3_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy3_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    % Terminal: Vunderbar == Vhat
    Valt(:,:,:,N_j)=V(:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1,1);
    aprimeProbs_d2a1a2ze=repmat(a2primeProbs,N_a1,1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz,1);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz,1);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2ze,1,1,N_semiz,1);

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
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

            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(beta*EVbase,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            % hat (QH-perceived): argmax of F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_hat;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothze=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind+N_d2*N_a1prime*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
            entireRHS_ii_hat=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_hat(d2a1primea2bothze(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            Policy3_ford3_hat(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_hat(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eind)),-1);
            Policy3_ford3_hat(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            L2offset=ceil(maxindexL2/N_d2);
            linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind+N_d2*n2long*N_a*N_bothz*eind;
            linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind+N_d2*n2long*N_a*N_bothz*eind;
            isInfLower=(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_hat(:,:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);

            % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
            entireRHS_ii_under=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_under(d2a1primea2bothze(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]);
            maxindexfull=maxindexL2+Nrow*(0:1:N_a-1)+shiftdim(Nrow*N_a*(0:1:N_bothz-1),-1)+shiftdim(Nrow*N_a*N_bothz*(0:1:N_e-1),-2);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
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

            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(beta*EVbase,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                % hat (QH-perceived): argmax of F + beta0*beta*EV
                DiscountedEV_hat_e=DiscountedEV_hat(:,:,:,:,:,e_c);
                DiscountedEVinterp_hat_e=DiscountedEVinterp_hat(:,:,:,:,:,e_c);
                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_hat_e;
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*shiftdim((0:1:N_bothz-1),-3);
                entireRHS_ii_hat=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_hat_e(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_hat(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_hat(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3_hat(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                L2offset=ceil(maxindexL2/N_d2);
                linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
                linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_hat(:,:,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
                % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                DiscountedEVinterp_under_e=DiscountedEVinterp_under(:,:,:,:,:,e_c);
                entireRHS_ii_under=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_under_e(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                maxindexfull=maxindexL2+Nrow*(0:1:N_a-1)+shiftdim(Nrow*N_a*(0:1:N_bothz-1),-1);
                V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
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
            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(beta*EVbase,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                DiscountedEV_hat_zb=DiscountedEV_hat(:,:,:,:,semizblock,:);
                DiscountedEVinterp_hat_zb=DiscountedEVinterp_hat(:,:,:,:,semizblock,:);
                DiscountedEVinterp_under_zb=DiscountedEVinterp_under(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV
                    DiscountedEV_hat_zbe=DiscountedEV_hat_zb(:,:,:,:,:,e_c);
                    DiscountedEVinterp_hat_zbe=DiscountedEVinterp_hat_zb(:,:,:,:,:,e_c);
                    entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_hat_zbe;
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*shiftdim((0:1:N_semiz-1),-3);
                    entireRHS_ii_hat=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_hat_zbe(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind+N_d2*N_a*semizBind;
                    Policy3_ford3_hat(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy3_ford3_hat(2,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_hat(3,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    L2offset=ceil(maxindexL2/N_d2);
                    linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                    linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
                    % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                    DiscountedEVinterp_under_zbe=DiscountedEVinterp_under_zb(:,:,:,:,:,e_c);
                    entireRHS_ii_under=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_under_zbe(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                    maxindexfull=maxindexL2+Nrow*(0:1:N_a-1)+shiftdim(Nrow*N_a*(0:1:N_semiz-1),-1);
                    V_ford3_under(:,semizblock,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
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
            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(beta*EVbase,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV
                    DiscountedEV_hat_ze=DiscountedEV_hat(:,:,:,:,z_c,e_c);
                    DiscountedEVinterp_hat_ze=DiscountedEVinterp_hat(:,:,:,:,z_c,e_c);
                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_hat_ze;
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_hat=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_hat_ze(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3_hat(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_hat(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_hat(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    L2offset=ceil(maxindexL2/N_d2);
                    linidx_lower=d_ind+N_d2*n2long*aind;
                    linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind;
                    isInfLower=(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
                    % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                    DiscountedEVinterp_under_ze=DiscountedEVinterp_under(:,:,:,:,z_c,e_c);
                    entireRHS_ii_under=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_under_ze(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    maxindexfull=maxindexL2+Nrow*(0:1:N_a-1);
                    V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
                end
            end
        end
    end

    % Max over d3 (hat = QH-perceived) and gather Vunderbar at the same chosen d3
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=3*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy3_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy3_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy3_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Valt(:,:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[N_a,N_bothz,N_e]);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1,1);
    aprimeProbs_d2a1a2ze=repmat(a2primeProbs,N_a1,1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz,1);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz,1);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2ze,1,1,N_semiz,1);

    % Continuation value is Vunderbar (Valt), integrated over e'
    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

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

            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(beta*EVbase,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            % hat (QH-perceived): argmax of F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_hat;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothze=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind+N_d2*N_a1prime*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
            entireRHS_ii_hat=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_hat(d2a1primea2bothze(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            Policy3_ford3_hat(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_hat(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eind)),-1);
            Policy3_ford3_hat(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            L2offset=ceil(maxindexL2/N_d2);
            linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind+N_d2*n2long*N_a*N_bothz*eind;
            linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind+N_d2*n2long*N_a*N_bothz*eind;
            isInfLower=(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_hat(:,:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);

            % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
            entireRHS_ii_under=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_under(d2a1primea2bothze(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]);
            maxindexfull=maxindexL2+Nrow*(0:1:N_a-1)+shiftdim(Nrow*N_a*(0:1:N_bothz-1),-1)+shiftdim(Nrow*N_a*N_bothz*(0:1:N_e-1),-2);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
        end
    elseif vfoptions.lowmemory==1
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

            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(beta*EVbase,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                % hat (QH-perceived): argmax of F + beta0*beta*EV
                DiscountedEV_hat_e=DiscountedEV_hat(:,:,:,:,:,e_c);
                DiscountedEVinterp_hat_e=DiscountedEVinterp_hat(:,:,:,:,:,e_c);
                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_hat_e;
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*shiftdim((0:1:N_bothz-1),-3);
                entireRHS_ii_hat=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_hat_e(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3_hat(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_hat(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3_hat(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                L2offset=ceil(maxindexL2/N_d2);
                linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
                linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*bothzBind;
                isInfLower=(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_hat(:,:,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
                % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                DiscountedEVinterp_under_e=DiscountedEVinterp_under(:,:,:,:,:,e_c);
                entireRHS_ii_under=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_under_e(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                maxindexfull=maxindexL2+Nrow*(0:1:N_a-1)+shiftdim(Nrow*N_a*(0:1:N_bothz-1),-1);
                V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
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
            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(beta*EVbase,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                DiscountedEV_hat_zb=DiscountedEV_hat(:,:,:,:,semizblock,:);
                DiscountedEVinterp_hat_zb=DiscountedEVinterp_hat(:,:,:,:,semizblock,:);
                DiscountedEVinterp_under_zb=DiscountedEVinterp_under(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV
                    DiscountedEV_hat_zbe=DiscountedEV_hat_zb(:,:,:,:,:,e_c);
                    DiscountedEVinterp_hat_zbe=DiscountedEVinterp_hat_zb(:,:,:,:,:,e_c);
                    entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_hat_zbe;
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*shiftdim((0:1:N_semiz-1),-3);
                    entireRHS_ii_hat=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_hat_zbe(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind+N_d2*N_a*semizBind;
                    Policy3_ford3_hat(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy3_ford3_hat(2,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_hat(3,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    L2offset=ceil(maxindexL2/N_d2);
                    linidx_lower=d_ind+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                    linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind+N_d2*n2long*N_a*semizBind;
                    isInfLower=(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
                    % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                    DiscountedEVinterp_under_zbe=DiscountedEVinterp_under_zb(:,:,:,:,:,e_c);
                    entireRHS_ii_under=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_under_zbe(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                    maxindexfull=maxindexL2+Nrow*(0:1:N_a-1)+shiftdim(Nrow*N_a*(0:1:N_semiz-1),-1);
                    V_ford3_under(:,semizblock,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
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
            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVbase=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(beta*EVbase,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV
                    DiscountedEV_hat_ze=DiscountedEV_hat(:,:,:,:,z_c,e_c);
                    DiscountedEVinterp_hat_ze=DiscountedEVinterp_hat(:,:,:,:,z_c,e_c);
                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_hat_ze;
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    entireRHS_ii_hat=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_hat_ze(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3_hat(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_hat(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3_hat(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    L2offset=ceil(maxindexL2/N_d2);
                    linidx_lower=d_ind+N_d2*n2long*aind;
                    linidx_upper=d_ind+N_d2*(n2long-1)+N_d2*n2long*aind;
                    isInfLower=(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                    inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                    inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                    flag_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
                    % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                    DiscountedEVinterp_under_ze=DiscountedEVinterp_under(:,:,:,:,z_c,e_c);
                    entireRHS_ii_under=ReturnMatrix_ii_hat+reshape(DiscountedEVinterp_under_ze(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    maxindexfull=maxindexL2+Nrow*(0:1:N_a-1);
                    V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
                end
            end
        end
    end

    % Max over d3 (hat = QH-perceived) and gather Vunderbar at the same chosen d3
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    V(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=3*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy3_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,jj)=reshape(Policy3_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy3_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_hat((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Valt(:,:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(4,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1);
Policy=[Policy; PolicyL2flag];


end
