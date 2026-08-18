function [V,Policy,Valt]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteSemiExoS_GI2A_e_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic + ExperienceAssete + SemiExo, GI2A (two standard assets, with d1).
% d1 is any other decision, d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is the grid-interpolated standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% z is exogenous Markov, semiz is semi-exogenous; bothz=(semiz,z) with semiz varying fastest.
% aprimeFn = aprimeFn(d2, a3, e, ...)   (depends on current e; not on z or semiz)
%
% Sophisticated QH over the same GI2A argmax axis the exponential SemiExo ze GI2A raw maxes over:
%   V/Policy come from the  F + beta0*beta*EV  argmax (QH-perceived); Policy stores
%     (d12, d3, a1prime-midpoint, a2prime, a1prime-L2index) plus the appended L2flag row
%     (d12 is the joint (d1,d2) index).
%   Valt (=Vunderbar) is the  F + beta*EV  RHS GATHERED at that same GI2A argmax (NOT re-maximised).
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Valt (Vunderbar).
%
% lowmemory: 3 shocks {z,semiz,e} => levels {0,1,2,3}.
%   =0 vectorise bothz and e; =1 loop e (bothz parallel); =2 outer-loop z / inner-loop e (semiz parallel); =3 joint bothz outer / inner-loop e.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

% Per-dim factored a3 grid for the ReturnFn builder (l_a3==1: 1 column, l_a3==2: 2 columns)
a3_gridvals=CreateGridvals(n_a3,a3_grid,1);

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray'); % (d12, d3, a1prime-midpoint, a2prime, a1primeL2ind)
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
Valt=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]; maps d12-index to d2-component

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_semiz=[n_semiz,ones(1,length(n_z))];
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Per-d3 workspaces (hat=QH-perceived @beta0beta argmax, under=beta-RHS gathered at that argmax)
V_ford3_hat=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_hat=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray'); % (d12, a1prime-midpoint, a2prime, a1primeL2ind)
flag_ford3_hat=2*ones(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);
Nrow=N_d12*n2long*N_a2; % row dimension (d12 * fine-a1prime * a2prime) of the fine RHS, for the under gather

aind=gpuArray(0:1:N_a-1);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % Terminal period: no continuation, so Vunderbar equals Vhat
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_hat(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_hat(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_hat(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_hat(4,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_hat(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_hat(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_hat(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_hat(4,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex]=max(ReturnMatrix_d3e,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                    V_ford3_hat(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_hat(1,:,zind,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,zind,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_hat(3,:,zind,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,zind,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
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
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex]=max(ReturnMatrix_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3ze,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_hat(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_hat(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_d3ze(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end
    % Max over d3 and unpack (hat = QH-perceived)
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]); % d12
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]); % a2prime
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    % Terminal: Vunderbar == Vhat
    Valt(:,:,:,N_j)=V(:,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a3, n_e, d2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_e] (scalar exp-asset only; aprimeFn sees current e, not z nor semiz)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);

    a3pIdx_repd=reshape(repmat(a3primeIndex,N_a1*N_a2,1,1),[N_d2*N_a1*N_a2,N_a3,1,N_e]); % no z dependence -> singleton current-bothz slot
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(reshape(repmat(a3primeProbs,N_a1*N_a2,1,1),[N_d2*N_a1*N_a2,N_a3,1,N_e]),1,1,1,1,N_bothz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,1,N_e,N_bothz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,1,N_e,N_bothz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    % EV_aprime is [N_d2*N_a1*N_a2,N_a3,1,N_e,N_bothz] (current-bothz slot is singleton: aprime is z-independent), trailing dim is bothzprime (d3-independent)

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(beta*EVbase,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

            % hat (QH-perceived): argmax of F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+repelem(DiscountedEV_hat,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
            entireRHS_ii_hat=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_hat(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_hat(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_hat(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_hat(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_hat(4,:,:,:,d3_c)=maxindexL2a1;
            linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

            % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
            entireRHS_ii_under=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_under(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            maxindexfull=maxindexL2 + Nrow*aind + Nrow*N_a*bothzBind + Nrow*N_a*N_bothz*eBind;
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(beta*EVbase,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_hat_e=DiscountedEV_hat(:,:,:,:,:,:,:,e_c);
                DiscountedEVinterp_hat_e=DiscountedEVinterp_hat(:,:,:,:,:,:,:,e_c);
                DiscountedEVinterp_under_e=DiscountedEVinterp_under(:,:,:,:,:,:,:,e_c);

                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);

                % hat (QH-perceived): argmax of F + beta0*beta*EV
                entireRHS_d3e=ReturnMatrix_d3e+repelem(DiscountedEV_hat_e,N_d1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);
                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_hat=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_hat_e(aprime),[N_d12*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_hat(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_hat(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_hat(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_hat(4,:,:,e_c,d3_c)=maxindexL2a1;
                linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                entireRHS_ii_under=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_under_e(aprime),[N_d12*n2long*N_a2,N_a,N_bothz]);
                maxindexfull=maxindexL2 + Nrow*aind + Nrow*N_a*bothzBind;
                V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(beta*EVbase,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_hat_ze=DiscountedEV_hat(:,:,:,:,:,:,zind,e_c);
                    DiscountedEVinterp_hat_ze=DiscountedEVinterp_hat(:,:,:,:,:,:,zind,e_c);
                    DiscountedEVinterp_under_ze=DiscountedEVinterp_under(:,:,:,:,:,:,zind,e_c);

                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV
                    entireRHS_d3e=ReturnMatrix_d3e+repelem(DiscountedEV_hat_ze,N_d1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);
                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_hat=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_hat_ze(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                    V_ford3_hat(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_hat(1,:,zind,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,zind,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_hat(3,:,zind,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,zind,e_c,d3_c)=maxindexL2a1;
                    linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                    % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                    entireRHS_ii_under=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_under_ze(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    maxindexfull=maxindexL2 + Nrow*aind + Nrow*N_a*semizBind;
                    V_ford3_under(:,zind,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(beta*EVbase,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_hat_ze=DiscountedEV_hat(:,:,:,:,:,:,z_c,e_c);
                    DiscountedEVinterp_hat_ze=DiscountedEVinterp_hat(:,:,:,:,:,:,z_c,e_c);
                    DiscountedEVinterp_under_ze=DiscountedEVinterp_under(:,:,:,:,:,:,z_c,e_c);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV
                    entireRHS_d3ze=ReturnMatrix_d3ze+repelem(DiscountedEV_hat_ze,N_d1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);
                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_hat=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_hat_ze(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_hat(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_hat(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                    % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                    entireRHS_ii_under=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_under_ze(aprime),[N_d12*n2long*N_a2,N_a]);
                    maxindexfull=maxindexL2 + Nrow*aind;
                    V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
                end
            end
        end
    end

    % Max over d3 (hat = QH-perceived) and gather Vunderbar at the same chosen d3
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]); % d12
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]); % a2prime
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
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

    % Continuation value is Vunderbar (Valt), integrated over e'
    EVpre=squeeze(sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a3, n_e, d2_gridvals, a3_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_e] (scalar exp-asset only; aprimeFn sees current e, not z nor semiz)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);

    a3pIdx_repd=reshape(repmat(a3primeIndex,N_a1*N_a2,1,1),[N_d2*N_a1*N_a2,N_a3,1,N_e]); % no z dependence -> singleton current-bothz slot
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(reshape(repmat(a3primeProbs,N_a1*N_a2,1,1),[N_d2*N_a1*N_a2,N_a3,1,N_e]),1,1,1,1,N_bothz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,1,N_e,N_bothz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,1,N_e,N_bothz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    % EV_aprime is [N_d2*N_a1*N_a2,N_a3,1,N_e,N_bothz] (current-bothz slot is singleton: aprime is z-independent), trailing dim is bothzprime (d3-independent)

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(beta*EVbase,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);

            % hat (QH-perceived): argmax of F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+repelem(DiscountedEV_hat,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
            entireRHS_ii_hat=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_hat(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_hat(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_hat(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_hat(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_hat(4,:,:,:,d3_c)=maxindexL2a1;
            linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

            % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
            entireRHS_ii_under=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_under(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            maxindexfull=maxindexL2 + Nrow*aind + Nrow*N_a*bothzBind + Nrow*N_a*N_bothz*eBind;
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(beta*EVbase,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_hat_e=DiscountedEV_hat(:,:,:,:,:,:,:,e_c);
                DiscountedEVinterp_hat_e=DiscountedEVinterp_hat(:,:,:,:,:,:,:,e_c);
                DiscountedEVinterp_under_e=DiscountedEVinterp_under(:,:,:,:,:,:,:,e_c);

                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);

                % hat (QH-perceived): argmax of F + beta0*beta*EV
                entireRHS_d3e=ReturnMatrix_d3e+repelem(DiscountedEV_hat_e,N_d1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);
                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_hat=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_hat_e(aprime),[N_d12*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_hat(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_hat(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_hat(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_hat(4,:,:,e_c,d3_c)=maxindexL2a1;
                linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                entireRHS_ii_under=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_under_e(aprime),[N_d12*n2long*N_a2,N_a,N_bothz]);
                maxindexfull=maxindexL2 + Nrow*aind + Nrow*N_a*bothzBind;
                V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(beta*EVbase,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_hat_ze=DiscountedEV_hat(:,:,:,:,:,:,zind,e_c);
                    DiscountedEVinterp_hat_ze=DiscountedEVinterp_hat(:,:,:,:,:,:,zind,e_c);
                    DiscountedEVinterp_under_ze=DiscountedEVinterp_under(:,:,:,:,:,:,zind,e_c);

                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV
                    entireRHS_d3e=ReturnMatrix_d3e+repelem(DiscountedEV_hat_ze,N_d1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);
                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_hat=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_hat_ze(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                    V_ford3_hat(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_hat(1,:,zind,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,zind,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_hat(3,:,zind,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,zind,e_c,d3_c)=maxindexL2a1;
                    linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                    % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                    entireRHS_ii_under=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_under_ze(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    maxindexfull=maxindexL2 + Nrow*aind + Nrow*N_a*semizBind;
                    V_ford3_under(:,zind,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*EVbase;
            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(beta*EVbase,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_hat_ze=DiscountedEV_hat(:,:,:,:,:,:,z_c,e_c);
                    DiscountedEVinterp_hat_ze=DiscountedEVinterp_hat(:,:,:,:,:,:,z_c,e_c);
                    DiscountedEVinterp_under_ze=DiscountedEVinterp_under(:,:,:,:,:,:,z_c,e_c);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV
                    entireRHS_d3ze=ReturnMatrix_d3ze+repelem(DiscountedEV_hat_ze,N_d1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);
                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_hat=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_hat=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_hat_ze(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_hat,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_hat(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_hat(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,z_c,e_c,d3_c)=maxindexL2a1;
                    linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_hat(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_hat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                    % under: F + beta*EV GATHERED at the hat GI argmax (same fine grid, not re-maximised)
                    entireRHS_ii_under=reshape(ReturnMatrix_ii_hat+DiscountedEVinterp_under_ze(aprime),[N_d12*n2long*N_a2,N_a]);
                    maxindexfull=maxindexL2 + Nrow*aind;
                    V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_ii_under(maxindexfull),1);
                end
            end
        end
    end

    % Max over d3 (hat = QH-perceived) and gather Vunderbar at the same chosen d3
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    V(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]); % d12
    Policy(3,:,:,:,jj)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]); % a2prime
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_hat((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Valt(:,:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);
Policy=[Policy; PolicyL2flag];


end
