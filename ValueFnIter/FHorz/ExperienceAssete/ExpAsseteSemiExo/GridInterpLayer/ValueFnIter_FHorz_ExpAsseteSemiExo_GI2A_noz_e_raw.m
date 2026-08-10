function [V,Policy]=ValueFnIter_FHorz_ExpAsseteSemiExo_GI2A_noz_e_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_semiz, n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% SemiExo graft of ValueFnIter_FHorz_ExpAssete_GI2A_noz_e_raw (which supplies the GI2A grid-interp math).
% d1 is any other decision, d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is the grid-interpolated standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% semiz is semi-exogenous, no z; semiz fills the z-slot of the return fn.
% aprimeFn = aprimeFn(d2, a3, e, ...)   (depends on current e; scalar experience asset)
% Policy stores (d12, d3, a1prime-midpoint, a2prime, a1prime-L2index) plus the appended L2flag row (d12 is the joint (d1,d2) index).
% lowmemory: 2 shocks {semiz,e} => levels {0,1,2}.
%   =0 vectorise semiz and e; =1 loop e (semiz parallel); =2 loop semiz and e.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_semiz,N_e,N_j,'gpuArray'); % (d12, d3, a1prime-midpoint, a2prime, a1primeL2ind)
PolicyL2flag=2*ones(1,N_a,N_semiz,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

%%
d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]; maps d12-index to d2-component

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate (for the d3-loop sections, which loop over d3 and then max over d3)
V_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy4_ford3_jj=zeros(4,N_a,N_semiz,N_e,N_d3,'gpuArray'); % (d12, a1prime-midpoint, a2prime, a1primeL2ind)
flag_ford3_jj=2*ones(N_a,N_semiz,N_e,N_d3,'gpuArray');

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);

semiz_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind + N_d12*N_a2*N_a*N_semiz*eBind;
            Policy4_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_jj(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_jj(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_jj(4,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind + N_d12*n2long*N_a2*N_a*N_semiz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind + N_d12*n2long*N_a2*N_a*N_semiz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                Policy4_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_jj(4,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2 % loop semiz and e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex]=max(ReturnMatrix_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
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
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    temp=4*((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_jj(1+temp),[1,N_a,N_semiz,N_e]); % d12
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_jj(2+temp),[1,N_a,N_semiz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_jj(3+temp),[1,N_a,N_semiz,N_e]); % a2prime
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_jj(4+temp),[1,N_a,N_semiz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_jj((1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3)); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a3, n_e, d2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_e] (N_e here is the current e)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1); % [N_d2*N_a1*N_a2,N_a3,N_e]
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs_folda3e=repmat(a3primeProbs,N_a1*N_a2,1,1);
    % Insert singleton semiz dim at position 3 so we can broadcast with semiz_offset
    aprimeIndex=reshape(aprimeIndex,[N_d2*N_a1*N_a2,N_a3,1,N_e]);
    aprimeplus1Index=reshape(aprimeplus1Index,[N_d2*N_a1*N_a2,N_a3,1,N_e]);
    aprimeProbs_folda3e=reshape(aprimeProbs_folda3e,[N_d2*N_a1*N_a2,N_a3,1,N_e]);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_semiz]);

            lin_lower=aprimeIndex+semiz_offset; % [N_d2*N_a1*N_a2,N_a3,N_semiz,N_e]
            lin_upper=aprimeplus1Index+semiz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_folda3e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_semiz,N_e]

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_d3=ReturnMatrix_d3+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind + N_d12*N_a2*N_a*N_semiz*eBind;
            Policy4_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_jj(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_jj(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_jj(4,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind + N_d12*n2long*N_a2*N_a*N_semiz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind + N_d12*n2long*N_a2*N_a*N_semiz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
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
            aprimeProbs_d3=repmat(aprimeProbs_folda3e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_e=DiscountedEV(:,:,:,:,:,:,:,e_c);
                DiscountedEVinterp_e=DiscountedEVinterp(:,:,:,:,:,:,:,e_c);

                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_d3e=ReturnMatrix_d3e+repelem(DiscountedEV_e,N_d1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                entireRHS_ii_d3e=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_e(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                Policy4_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_jj(4,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2 % loop semiz and e
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
            aprimeProbs_d3=repmat(aprimeProbs_folda3e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,z_c,e_c);
                    DiscountedEVinterp_ze=DiscountedEVinterp(:,:,:,:,:,:,z_c,e_c);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3ze=ReturnMatrix_d3ze+repelem(DiscountedEV_ze,N_d1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_d3ze=reshape(ReturnMatrix_ii_d3ze+DiscountedEVinterp_ze(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
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
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    temp=4*((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_jj(1+temp),[1,N_a,N_semiz,N_e]); % d12
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_jj(2+temp),[1,N_a,N_semiz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_jj(3+temp),[1,N_a,N_semiz,N_e]); % a2prime
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_jj(4+temp),[1,N_a,N_semiz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_jj((1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
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

    EVpre=squeeze(sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3)); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a3, n_e, d2_gridvals, a3_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_e] (N_e here is the current e)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1); % [N_d2*N_a1*N_a2,N_a3,N_e]
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs_folda3e=repmat(a3primeProbs,N_a1*N_a2,1,1);
    % Insert singleton semiz dim at position 3 so we can broadcast with semiz_offset
    aprimeIndex=reshape(aprimeIndex,[N_d2*N_a1*N_a2,N_a3,1,N_e]);
    aprimeplus1Index=reshape(aprimeplus1Index,[N_d2*N_a1*N_a2,N_a3,1,N_e]);
    aprimeProbs_folda3e=reshape(aprimeProbs_folda3e,[N_d2*N_a1*N_a2,N_a3,1,N_e]);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_semiz]);

            lin_lower=aprimeIndex+semiz_offset; % [N_d2*N_a1*N_a2,N_a3,N_semiz,N_e]
            lin_upper=aprimeplus1Index+semiz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_folda3e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_semiz,N_e]

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_d3=ReturnMatrix_d3+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1fine*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind + N_d12*N_a2*N_a*N_semiz*eBind;
            Policy4_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_jj(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_jj(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_jj(4,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind + N_d12*n2long*N_a2*N_a*N_semiz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind + N_d12*n2long*N_a2*N_a*N_semiz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
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
            aprimeProbs_d3=repmat(aprimeProbs_folda3e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_e=DiscountedEV(:,:,:,:,:,:,:,e_c);
                DiscountedEVinterp_e=DiscountedEVinterp(:,:,:,:,:,:,:,e_c);

                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_d3e=ReturnMatrix_d3e+repelem(DiscountedEV_e,N_d1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                entireRHS_ii_d3e=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_e(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                Policy4_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_jj(4,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2 % loop semiz and e
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
            aprimeProbs_d3=repmat(aprimeProbs_folda3e,1,1,N_semiz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7,8]),a1prime_grid),[2,1,3,4,5,6,7,8]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,z_c,e_c);
                    DiscountedEVinterp_ze=DiscountedEVinterp(:,:,:,:,:,:,z_c,e_c);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3ze=ReturnMatrix_d3ze+repelem(DiscountedEV_ze,N_d1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_d3ze=reshape(ReturnMatrix_ii_d3ze+DiscountedEVinterp_ze(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_jj(4,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
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
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    temp=4*((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_jj(1+temp),[1,N_a,N_semiz,N_e]); % d12
    Policy(3,:,:,:,jj)=reshape(Policy4_ford3_jj(2+temp),[1,N_a,N_semiz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_jj(3+temp),[1,N_a,N_semiz,N_e]); % a2prime
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_jj(4+temp),[1,N_a,N_semiz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_jj((1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];


end
