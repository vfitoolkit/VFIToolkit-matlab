function [V,Policy]=ValueFnIter_FHorz_ExpAssetzeSemiExo_GI2A_nod1_e_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, n_e, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% SemiExo graft of ValueFnIter_FHorz_ExpAssetze_GI2A_nod1_e_raw (which supplies the GI2A grid-interp math).
% d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is the grid-interpolated standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% z is exogenous Markov, semiz is semi-exogenous; bothz=(semiz,z) with semiz varying fastest.
% aprimeFn = aprimeFn(d2, a3, z, e, ...)   (depends on BOTH current z and current e)
% Policy stores (d2, d3, a1prime-midpoint, a2prime, a1prime-L2index) plus the appended L2flag row.
% lowmemory: 3 shocks {z,semiz,e} => levels {0,1,2,3}.
%   =0 vectorise bothz and e; =1 loop e (bothz parallel); =2 outer-loop z / inner-loop e (semiz parallel); =3 joint bothz outer / inner-loop e.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
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
Policy=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray'); % (d2, d3, a1prime-midpoint, a2prime, a1primeL2ind)
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_semiz=[n_semiz,ones(1,length(n_z))];
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate (for the d3-loop sections, which loop over d3 and then max over d3)
V_ford3_jj=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_jj=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray'); % (d2, a1prime-midpoint, a2prime, a1primeL2ind)
flag_ford3_jj=2*ones(N_a,N_bothz,N_e,N_d3,'gpuArray');

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_jj(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_jj(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_jj(4,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_jj(4,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex]=max(ReturnMatrix_d3e,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                    V_ford3_jj(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_jj(1,:,zind,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,zind,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,zind,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,zind,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
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
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex]=max(ReturnMatrix_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_d3ze(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end
    % Max over d3 and unpack
    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_jj(1+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_jj(2+temp),[1,N_a,N_bothz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_jj(3+temp),[1,N_a,N_bothz,N_e]); % a2prime
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_jj(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_jj((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a3, n_z, n_e, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % l_a3==1: a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z,N_e] (legacy lower-corner)
    % l_a3==2: a3primeIndex/a3primeProbs are [l_a3,N_d2,N_a3,N_z,N_e] (per-dim factored)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);

    if length(n_a3)==1
        a3pIdx_repd=repelem(repmat(a3primeIndex,N_a1*N_a2,1,1,1),1,1,N_semiz,1); % expand current z to bothz (semiz fastest)
        aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
        aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
        aprimeProbs=repmat(repelem(repmat(a3primeProbs,N_a1*N_a2,1,1,1),1,1,N_semiz,1),1,1,1,1,N_bothz);

        Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;
        EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    else
        % l_a3==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a3_1=n_a3(1);
        loIdx_1_repd=repelem(repmat(reshape(a3primeIndex(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1);
        loIdx_2_repd=repelem(repmat(reshape(a3primeIndex(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1);
        prob_1_exp=repmat(repelem(repmat(reshape(a3primeProbs(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1),1,1,1,1,N_bothz);
        prob_2_exp=repmat(repelem(repmat(reshape(a3primeProbs(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1),1,1,1,1,N_bothz);

        a3_kron_ll= loIdx_1_repd   +n_a3_1*(loIdx_2_repd-1);
        a3_kron_hl=(loIdx_1_repd+1)+n_a3_1*(loIdx_2_repd-1);
        a3_kron_lh= loIdx_1_repd   +n_a3_1* loIdx_2_repd;
        a3_kron_hh=(loIdx_1_repd+1)+n_a3_1* loIdx_2_repd;

        aprime_ll=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_ll-1);
        aprime_hl=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hl-1);
        aprime_lh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_lh-1);
        aprime_hh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hh-1);

        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV_aprime=c_loy+c_hiy;
    end
    % EV_aprime is [N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz], trailing dim is bothzprime (d3-independent)

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_jj(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_jj(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_jj(4,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_e=DiscountedEV(:,:,:,:,:,:,:,e_c);
                DiscountedEVinterp_e=DiscountedEVinterp(:,:,:,:,:,:,:,e_c);

                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_e;
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_d3e=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_e(aprime),[N_d2*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_jj(4,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,zind,e_c);
                    DiscountedEVinterp_ze=DiscountedEVinterp(:,:,:,:,:,:,zind,e_c);

                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_ze;
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_d3e=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_ze(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                    V_ford3_jj(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_jj(1,:,zind,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,zind,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,zind,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,zind,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,z_c,e_c);
                    DiscountedEVinterp_ze=DiscountedEVinterp(:,:,:,:,:,:,z_c,e_c);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_ze;
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_d3ze=reshape(ReturnMatrix_ii_d3ze+DiscountedEVinterp_ze(aprime),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_d3ze(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end

    % Max over d3 and unpack
    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_jj(1+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_jj(2+temp),[1,N_a,N_bothz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_jj(3+temp),[1,N_a,N_bothz,N_e]); % a2prime
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_jj(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_jj((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
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

    EVpre=squeeze(sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a3, n_z, n_e, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % l_a3==1: a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z,N_e] (legacy lower-corner)
    % l_a3==2: a3primeIndex/a3primeProbs are [l_a3,N_d2,N_a3,N_z,N_e] (per-dim factored)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);

    if length(n_a3)==1
        a3pIdx_repd=repelem(repmat(a3primeIndex,N_a1*N_a2,1,1,1),1,1,N_semiz,1); % expand current z to bothz (semiz fastest)
        aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
        aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
        aprimeProbs=repmat(repelem(repmat(a3primeProbs,N_a1*N_a2,1,1,1),1,1,N_semiz,1),1,1,1,1,N_bothz);

        Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;
        EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    else
        % l_a3==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a3_1=n_a3(1);
        loIdx_1_repd=repelem(repmat(reshape(a3primeIndex(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1);
        loIdx_2_repd=repelem(repmat(reshape(a3primeIndex(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1);
        prob_1_exp=repmat(repelem(repmat(reshape(a3primeProbs(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1),1,1,1,1,N_bothz);
        prob_2_exp=repmat(repelem(repmat(reshape(a3primeProbs(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1),1,1,N_semiz,1),1,1,1,1,N_bothz);

        a3_kron_ll= loIdx_1_repd   +n_a3_1*(loIdx_2_repd-1);
        a3_kron_hl=(loIdx_1_repd+1)+n_a3_1*(loIdx_2_repd-1);
        a3_kron_lh= loIdx_1_repd   +n_a3_1* loIdx_2_repd;
        a3_kron_hh=(loIdx_1_repd+1)+n_a3_1* loIdx_2_repd;

        aprime_ll=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_ll-1);
        aprime_hl=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hl-1);
        aprime_lh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_lh-1);
        aprime_hh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hh-1);

        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV_aprime=c_loy+c_hiy;
    end
    % EV_aprime is [N_d2*N_a1*N_a2,N_a3,N_bothz,N_e,N_bothz], trailing dim is bothzprime (d3-independent)

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_bothz*shiftdim((0:1:N_e-1),-6);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_jj(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_jj(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_jj(4,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_e=DiscountedEV(:,:,:,:,:,:,:,e_c);
                DiscountedEVinterp_e=DiscountedEVinterp(:,:,:,:,:,:,:,e_c);

                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_e;
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_d3e=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_e(aprime),[N_d2*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_jj(4,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,zind,e_c);
                    DiscountedEVinterp_ze=DiscountedEVinterp(:,:,:,:,:,:,zind,e_c);

                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_ze;
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_d3e=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_ze(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                    V_ford3_jj(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_jj(1,:,zind,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,zind,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,zind,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,zind,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*reshape(pi_bothz,[1,1,N_bothz,1,N_bothz]);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,5));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,z_c,e_c);
                    DiscountedEVinterp_ze=DiscountedEVinterp(:,:,:,:,:,:,z_c,e_c);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_ze;
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_d3ze=reshape(ReturnMatrix_ii_d3ze+DiscountedEVinterp_ze(aprime),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_d3ze(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end

    % Max over d3 and unpack
    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_jj(1+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(3,:,:,:,jj)=reshape(Policy4_ford3_jj(2+temp),[1,N_a,N_bothz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_jj(3+temp),[1,N_a,N_bothz,N_e]); % a2prime
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_jj(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_jj((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];


end
