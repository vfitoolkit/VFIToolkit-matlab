function [V,Policy3]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI2A_nod1_noz_e_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_semiz, n_e, n_u, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J, e_gridvals_J, u_gridvals, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% nod1, noz, e+semiz graft of ValueFnIter_FHorz_ExpAsset_GI2A_nod1_noz_e_raw (grid interpolation layer on a1).
% d2 determines experience asset (a3), d3 determines semi-exog state (semiz). No d1, no Markov z.
% a1 is the grid-interpolated standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% semiz is the semi-exogenous state (plays the role of bothz as there is no z); e is i.i.d.
% Policy3 stores (d2, d3, joint(a1prime,a2prime)); the 3rd row is a1prime_lower+N_a1*(a2prime-1).
% Two extra rows are appended for the grid-interp layer: a1prime L2 index and L2flag.
% lowmemory: 2 shocks {semiz,e} => levels {0,1,2}.
%   =0 vectorise semiz and e; =1 loop e (semiz parallel); =2 outer-loop semiz / inner-loop e.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_u=prod(n_u);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,joint(a1prime,a2prime) seperately
Policy3=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray'); % 1=d2, 2=d3, 3=joint(a1prime,a2prime), 4=a1prime L2
PolicyL2flag=2*ones(1,N_a,N_semiz,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

%% GI setup
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1
if vfoptions.lowmemory==0
    semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % dim 3
    eBind=shiftdim(gpuArray(0:1:N_e-1),-2); % dim 4
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % dim 3
elseif vfoptions.lowmemory==2
    special_n_semiz=ones(1,length(n_semiz));
    special_n_e=ones(1,length(n_e));
end

% Preallocate (for the max over d3)
V_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy3_ford3_jj=zeros(3,N_a,N_semiz,N_e,N_d3,'gpuArray'); % 1=d2, 2=joint(a1prime midpoint,a2prime), 3=a1prime L2 (d3 is the loop index)
flag_ford3_jj=2*ones(N_a,N_semiz,N_e,N_d3,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put u into third dimension
%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];

            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind + N_d2*N_a2*N_a*N_semiz*eBind;
            Policy3_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_jj(2,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy3_ford3_jj(3,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind + N_d2*n2long*N_a2*N_a*N_semiz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind + N_d2*n2long*N_a2*N_a*N_semiz*eBind;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                Policy3_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy3_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex]=max(ReturnMatrix_ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy3_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy3_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d3
    V(:,:,:,N_j)=V_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    temp=3*( (1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1) -1);
    Policy3(1,:,:,:,N_j)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_semiz,N_e]); % d2
    Policy3(3,:,:,:,N_j)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_semiz,N_e]); % joint(a1prime midpoint,a2prime)
    Policy3(4,:,:,:,N_j)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_semiz,N_e]); % a1prime L2
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_jj((1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3)); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_semiz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz], indexed by semizprime (d3-independent)
    EV_aprime=squeeze(sum((EV_aprime.*pi_u),3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]
    EV_aprime(isnan(EV_aprime))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS=ReturnMatrix+DiscountedEV;
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprime),[N_d2*n2long*N_a2,N_a,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind + N_d2*N_a2*N_a*N_semiz*eBind;
            Policy3_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_jj(2,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
            Policy3_ford3_jj(3,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind + N_d2*n2long*N_a2*N_a*N_semiz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind + N_d2*n2long*N_a2*N_a*N_semiz*eBind;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_e=ReturnMatrix_e+DiscountedEV;
                [~,maxindex]=max(entireRHS_e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                Policy3_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
                Policy3_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z;
                    [~,maxindex]=max(entireRHS_ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_z(aprime),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy3_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
                    Policy3_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d3
    V(:,:,:,N_j)=V_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    temp=3*( (1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1) -1);
    Policy3(1,:,:,:,N_j)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_semiz,N_e]); % d2
    Policy3(3,:,:,:,N_j)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_semiz,N_e]); % joint(a1prime midpoint,a2prime)
    Policy3(4,:,:,:,N_j)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_semiz,N_e]); % a1prime L2
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_jj((1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3)); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_semiz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz], indexed by semizprime (d3-independent)
    EV_aprime=squeeze(sum((EV_aprime.*pi_u),3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]
    EV_aprime(isnan(EV_aprime))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS=ReturnMatrix+DiscountedEV;
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprime),[N_d2*n2long*N_a2,N_a,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind + N_d2*N_a2*N_a*N_semiz*eBind;
            Policy3_ford3_jj(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3_jj(2,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
            Policy3_ford3_jj(3,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind + N_d2*n2long*N_a2*N_a*N_semiz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind + N_d2*n2long*N_a2*N_a*N_semiz*eBind;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_e=ReturnMatrix_e+DiscountedEV;
                [~,maxindex]=max(entireRHS_e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                Policy3_ford3_jj(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
                Policy3_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z;
                    [~,maxindex]=max(entireRHS_ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_z(aprime),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy3_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
                    Policy3_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d3
    V(:,:,:,jj)=V_jj;
    Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    temp=3*( (1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1) -1);
    Policy3(1,:,:,:,jj)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_semiz,N_e]); % d2
    Policy3(3,:,:,:,jj)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_semiz,N_e]); % joint(a1prime midpoint,a2prime)
    Policy3(4,:,:,:,jj)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_semiz,N_e]); % a1prime L2
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_jj((1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
end


%% With grid interpolation, switch from midpoint to lower grid index
% Currently Policy3(3,:) is joint(a1prime midpoint,a2prime), and Policy3(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy3(3,:) to joint(lower a1prime grid point,a2prime) and then have Policy3(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy3(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy3(3,:,:,:,:)=Policy3(3,:,:,:,:)-adjust; % decrement a1prime component of the joint (midpoint>=2 keeps it within the same a2prime block)
Policy3(4,:,:,:,:)=adjust.*Policy3(4,:,:,:,:)+(1-adjust).*(Policy3(4,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy3=[Policy3; PolicyL2flag];


end
