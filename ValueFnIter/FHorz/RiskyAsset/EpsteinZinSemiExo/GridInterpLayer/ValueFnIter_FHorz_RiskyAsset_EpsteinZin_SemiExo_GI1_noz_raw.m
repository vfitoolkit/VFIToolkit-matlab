function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_SemiExo_GI1_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% No z. bothz collapses to semiz; pi_bothz = pi_semiz(:,:,d4) (no kron).
%
% Grid-interpolation-layer version of the Epstein-Zin riskyasset+semiz solver.
% Grafts the Epstein-Zin transforms onto ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_noz_raw.
% See ValueFnIter_FHorz_RiskyAsset_EpsteinZin_SemiExo_GI1_raw for the ordering
% of the Epstein-Zin transform chain under GI (interp of the transformed EV
% before ^ezc6; d2 refine after the transform chain with the ezc9 trick, coarse
% and fine separately; d2 policy read off the fine refine; L2flag off the
% transformed return matrix).

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_u=prod(n_u);

N_d13=N_d1*N_d3;

% For aprimeFn (d2,d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
% Policy: rows (d1,d2,d3,d4,a1prime_low,L2ind,L2flag)
Policy=zeros(7,N_a,N_semiz,N_j,'gpuArray');

%%
u_grid=gpuArray(u_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
d23_grid=gpuArray(d23_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d13_gridvals=gpuArray(CreateGridvals([n_d1,n_d3],[d1_grid;d3_grid],1));

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_semiz-1),-3);
zindB=shiftdim(gpuArray(0:1:N_semiz-1),-1);

% Preallocate per-d4 slabs
V_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
% Per-d4 policy encoding (no d4 baked in): d13 + N_d13*(a1prime_mid-1) + N_d13*N_a1*(L2ind-1)
Policy_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
flag_ford4_jj=2*ones(N_a,N_semiz,N_d4,'gpuArray');
d2index_ford4_jj=ones(N_a,N_semiz,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
elseif vfoptions.EZoneminusbeta==2
    ezc1=1-sj(N_j)*DiscountFactorParamsVec;
end

% If there is a warm-glow at end of the final period, evaluate the warmglowfn
% (warm-glow depends only on a2prime; the u-lottery turns it into a function of d23)
if warmglow==1
    WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,N_j);
    WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec); % This depends on a2prime
    WGmatrix=WGmatrixraw;
    WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(N_j);
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity

    % Switch WGmatrix from being in terms of a2prime to being in terms of d23 (in expectation because of the u shocks)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
    % So I will add a check for when this happens, and then overwrite those (by setting the interpolation probs to zero)
    skipinterpWG=logical(WGmatrix(a2primeIndex)==WGmatrix(a2primeIndex+1));
    aprimeProbsWG=a2primeProbs; % [N_d23,N_u]
    aprimeProbsWG(skipinterpWG)=0;
    WG1=WGmatrix(a2primeIndex).*aprimeProbsWG; % probability of lower grid point
    WG2=WGmatrix(a2primeIndex+1).*(1-aprimeProbsWG); % probability of upper grid point
    % If WG1 or WG2 is infinite, and probability is zero, we will get a nan, so get rid of these
    WG1(isnan(WG1))=0;
    WG2(isnan(WG2))=0;
    % Expectation over u (using pi_u), and then add the lower and upper
    WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2); % [N_d23,1]

    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrix(becareful)=0;
    end
else
    WGmatrix=0;
end

if ~isfield(vfoptions,'V_Jplus1')
    if warmglow==1
        % Refine d2 out of the warm-glow (the only continuation term in the terminal period)
        [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3]),[],1);
        WGcol=ezc9*reshape(WGmatrix_onlyd3,[N_d3,1]); % constant in a1prime (and a,semiz)
    else
        WGcol=zeros(N_d3,1,'gpuArray');
    end
    WGcol_d13=repelem(WGcol,N_d1,1); % [N_d13,1]

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,n_a1,n_a2,n_semiz, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_d4).*(ReturnMatrix_d4~=0)); % finite but not zero
            ReturnMatrix_d4(becareful)=(ezc1*ReturnMatrix_d4(becareful).^ezc2(N_j)).^ezc7(N_j);
            ReturnMatrix_d4(ReturnMatrix_d4==0)=-Inf;
            entireRHS=ReturnMatrix_d4+WGcol_d13; % warm-glow (zero if not using); constant in a1prime
            [~,maxindex_d4]=max(entireRHS,[],2);

            midpoint_d4=max(min(maxindex_d4,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_d4+(midpoint_d4-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
            ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
            entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d13,n2long,N_a1,N_a2,N_semiz])+WGcol_d13,[N_d13*n2long,N_a1*N_a2,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
            mid_at=shiftdim(squeeze(midpoint_d4(allind)),-1);
            L2offset=ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            Policy_ford4_jj(:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
            % d2 at j=N_j
            if warmglow==1
                d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a,N_semiz]
                d2index_ford4_jj(:,:,d4_c)=shiftdim(d2index(d3opt),1); % d2 comes from refining the warm-glow (no a nor semiz in WGmatrix)
            else
                d2index_ford4_jj(:,:,d4_c)=1; % d2 meaningless at j=N_j without warm-glow
            end
        end
    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d4z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,n_a1,n_a2,special_n_semiz, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_d4z).*(ReturnMatrix_d4z~=0)); % finite but not zero
                ReturnMatrix_d4z(becareful)=(ezc1*ReturnMatrix_d4z(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_d4z(ReturnMatrix_d4z==0)=-Inf;
                entireRHS_z=ReturnMatrix_d4z+WGcol_d13; % warm-glow (zero if not using); constant in a1prime
                [~,maxindex_d4z]=max(entireRHS_z,[],2);

                midpoint_d4z=max(min(maxindex_d4z,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_d4z+(midpoint_d4z-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_semiz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                becareful=logical(isfinite(ReturnMatrix_ii_z).*(ReturnMatrix_ii_z~=0)); % finite but not zero
                ReturnMatrix_ii_z(becareful)=(ezc1*ReturnMatrix_ii_z(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_ii_z(ReturnMatrix_ii_z==0)=-Inf;
                entireRHS_ii=reshape(reshape(ReturnMatrix_ii_z,[N_d13,n2long,N_a1,N_a2])+WGcol_d13,[N_d13*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind;
                mid_at=midpoint_d4z(allind);
                L2offset=ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,z_c,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
                % d2 at j=N_j
                if warmglow==1
                    d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a]
                    d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2index(d3opt),1); % d2 comes from refining the warm-glow (no a nor semiz in WGmatrix)
                else
                    d2index_ford4_jj(:,z_c,d4_c)=1; % d2 meaningless at j=N_j without warm-glow
                end
            end
        end
    end
    % combine across d4 (inlined)
    [Vbest,d4winner]=max(V_ford4_jj,[],3);
    V(:,:,N_j)=Vbest;
    Ncomb=N_a*N_semiz;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_semiz]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_semiz]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_semiz]);
    d13part=rem(polenc-1,N_d13)+1;
    tmp=ceil(polenc/N_d13);
    midpart=rem(tmp-1,N_a1)+1;
    L2offset=ceil(tmp/N_a1);
    adjust=(L2offset<1+n2short+1);
    a1prime_low=midpart-adjust;
    L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);
    d1part=rem(d13part-1,N_d1)+1;
    d3part=rem(ceil(d13part/N_d1)-1,N_d3)+1;
    Policy(1,:,:,N_j)=reshape(d1part,[1,N_a,N_semiz]);
    Policy(2,:,:,N_j)=reshape(d2winner,[1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(d3part,[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(d4winner,[1,N_a,N_semiz]);
    Policy(5,:,:,N_j)=reshape(a1prime_low,[1,N_a,N_semiz]);
    Policy(6,:,:,N_j)=reshape(L2ind,[1,N_a,N_semiz]);
    Policy(7,:,:,N_j)=reshape(flagwinner,[1,N_a,N_semiz]);
else
    if warmglow==0 % if warmglow==1 these were already created above
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
        [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    end

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(N_j);
    temp(EVpre==0)=0;

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    semiz_gridvals=semiz_gridvals_J(:,:,N_j);

    % per-period inner (inlined)
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c); % no kron in noz
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            % Expectation over semiz' (on the transformed object)
            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            % u-lottery on the transformed object
            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
            EV=reshape(EV,[N_d23,N_a1,N_semiz]);

            % Interpolate the transformed EV over a1prime (BEFORE the certainty-equivalent power ^ezc6)
            EVinterp=permute(interp1(a1_gridvals,permute(EV,[2,1,3]),a1prime_grid),[2,1,3]); % [N_d23,N_a1prime,N_semiz]

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise
            temp4=EV;
            temp4interp=EVinterp;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,N_a1,N_semiz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV==0)&(WGmatrixbig==0))=0; % Is actually zero
                WGmatrixbigfine=WGmatrix.*ones(1,N_a1prime,N_semiz);
                becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixbigfine)); % both are finite
                temp4interp(becareful)=(sj(N_j)*temp4interp(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbigfine(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4interp((EVinterp==0)&(WGmatrixbigfine==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV==0)=0;
                temp4interp(isfinite(temp4interp))=(sj(N_j)*temp4interp(isfinite(temp4interp)).^ezc8(N_j)).^ezc6(N_j);
                temp4interp(EVinterp==0)=0;
            end

            % Refine d2 out of the continuation (ezc9 handles the sign so the max is correct), coarse and fine
            temp4_onlyd3=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,N_semiz]),[],1);
            [temp4interp_onlyd3,d2indexfine]=max(ezc9*ezc3*reshape((~isinf(temp4interp)).*temp4interp,[N_d2,N_d3*N_a1prime,N_semiz]),[],1);
            d2indexfine_resh=reshape(d2indexfine,[N_d3,N_a1prime,N_semiz]);

            % DiscountedEV (the ezc9 outside undoes the sign flip used inside the refine)
            DiscountedEV=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1,N_semiz]);
            DiscountedEVinterp=DiscountFactorParamsVec*ezc9*reshape(temp4interp_onlyd3,[N_d3,N_a1prime,1,1,N_semiz]);
            DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
            DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

            % Level-1 Return at coarse a1prime grid
            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,n_a1,n_a2,n_semiz, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals, ReturnFnParamsVec,1,0);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_d4).*(ReturnMatrix_d4~=0)); % finite but not zero
            temp2=ReturnMatrix_d4;
            temp2(becareful)=ReturnMatrix_d4(becareful).^ezc2(N_j);
            temp2(ReturnMatrix_d4==0)=-Inf;
            entireRHS=ezc1*temp2+DiscountedEV_d13;
            temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
            entireRHS(temp5)=entireRHS(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS(entireRHS==0)=-Inf;

            [~,maxindex]=max(entireRHS,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals, ReturnFnParamsVec,2,0);
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;
            da1primez=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1)+N_d13*N_a1prime*zind;
            entireRHS_ii=ezc1*temp2_ii+reshape(DiscountedEVinterp_d13(da1primez(:)),[N_d13*n2long,N_a1*N_a2,N_semiz]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
            mid_at=shiftdim(squeeze(midpoint(allind)),-1);
            L2offset=ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
            isInfLower    = (temp2_ii(linidx_lower) == -Inf);
            isInfUpper    = (temp2_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
            % d2 lookup per (a,semiz) — read off the FINE refine at the chosen fine a1prime point
            d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
            a1mid=midpoint(allind);
            a1fine=(a1mid+(a1mid-1)*n2short)+L2offset-n2short-2; % chosen fine a1prime index
            zlin=shiftdim(gpuArray(0:N_semiz-1),-1);
            linlookup=d3opt+N_d3*(a1fine-1)+N_d3*N_a1prime*zlin;
            d2index_ford4_jj(:,:,d4_c)=shiftdim(d2indexfine_resh(linlookup),1);
        end

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        special_n_semiz=ones(1,length(n_semiz));
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals(z_c,:);

                % Expectation over semiz' (on the transformed object)
                EV_z=temp.*pi_semizd4(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);
                EV_z=reshape(EV_z,[N_a,1]);

                % u-lottery on the transformed object
                skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
                aprimeProbs=repmat(a2primeProbs,N_a1,1);
                aprimeProbs(skipinterp)=0;
                aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);

                EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
                EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
                EV_z=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
                EV_z=reshape(EV_z,[N_d23,N_a1]);

                % Interpolate the transformed EV over a1prime (BEFORE the certainty-equivalent power ^ezc6)
                EVinterp_z=permute(interp1(a1_gridvals,permute(EV_z,[2,1]),a1prime_grid),[2,1]); % [N_d23,N_a1prime]

                % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise
                temp4=EV_z;
                temp4interp=EVinterp_z;
                if warmglow==1
                    WGmatrixbig=WGmatrix.*ones(1,N_a1);
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_z==0)&(WGmatrixbig==0))=0; % Is actually zero
                    WGmatrixbigfine=WGmatrix.*ones(1,N_a1prime);
                    becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixbigfine)); % both are finite
                    temp4interp(becareful)=(sj(N_j)*temp4interp(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbigfine(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4interp((EVinterp_z==0)&(WGmatrixbigfine==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_z==0)=0;
                    temp4interp(isfinite(temp4interp))=(sj(N_j)*temp4interp(isfinite(temp4interp)).^ezc8(N_j)).^ezc6(N_j);
                    temp4interp(EVinterp_z==0)=0;
                end

                % Refine d2 out of the continuation (ezc9 handles the sign so the max is correct), coarse and fine
                temp4_onlyd3=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1]),[],1);
                [temp4interp_onlyd3,d2indexfine]=max(ezc9*ezc3*reshape((~isinf(temp4interp)).*temp4interp,[N_d2,N_d3*N_a1prime]),[],1);
                d2indexfine_z=reshape(d2indexfine,[N_d3,N_a1prime]);

                % DiscountedEV (the ezc9 outside undoes the sign flip used inside the refine)
                DiscountedEV_z=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1]);
                DiscountedEVinterp_z=DiscountFactorParamsVec*ezc9*reshape(temp4interp_onlyd3,[N_d3,N_a1prime,1,1]);
                DiscountedEV_d13_z=repelem(DiscountedEV_z,N_d1,1);
                DiscountedEVinterp_d13_z=repelem(DiscountedEVinterp_z,N_d1,1);

                ReturnMatrix_d4z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,n_a1,n_a2,special_n_semiz, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_d4z).*(ReturnMatrix_d4z~=0)); % finite but not zero
                temp2=ReturnMatrix_d4z;
                temp2(becareful)=ReturnMatrix_d4z(becareful).^ezc2(N_j);
                temp2(ReturnMatrix_d4z==0)=-Inf;
                entireRHS_z=ezc1*temp2+DiscountedEV_d13_z;
                temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
                entireRHS_z(temp5)=entireRHS_z(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_z(entireRHS_z==0)=-Inf;

                [~,maxindex]=max(entireRHS_z,[],2);

                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_semiz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                becareful=logical(isfinite(ReturnMatrix_ii_z).*(ReturnMatrix_ii_z~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii_z;
                temp2_ii(becareful)=ReturnMatrix_ii_z(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_ii_z==0)=-Inf;
                da1prime=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1);
                entireRHS_ii_z=ezc1*temp2_ii+reshape(DiscountedEVinterp_d13_z(da1prime(:)),[N_d13*n2long,N_a1*N_a2]);
                temp5=logical(isfinite(entireRHS_ii_z).*(entireRHS_ii_z~=0));
                entireRHS_ii_z(temp5)=entireRHS_ii_z(temp5).^ezc7(N_j);
                entireRHS_ii_z(entireRHS_ii_z==0)=-Inf;
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind;
                mid_at=midpoint(allind);
                L2offset=ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                isInfLower    = (temp2_ii(linidx_lower) == -Inf);
                isInfUpper    = (temp2_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,z_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
                % d2 lookup per a — read off the FINE refine at the chosen fine a1prime point
                d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                a1mid=midpoint(allind);
                a1fine=(a1mid+(a1mid-1)*n2short)+L2offset-n2short-2; % chosen fine a1prime index
                linlookup=d3opt+N_d3*(a1fine-1);
                d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2indexfine_z(linlookup),1);
            end
        end
    end

    % combine across d4 (inlined)
    [Vbest,d4winner]=max(V_ford4_jj,[],3);
    V(:,:,N_j)=Vbest;
    Ncomb=N_a*N_semiz;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_semiz]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_semiz]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_semiz]);
    d13part=rem(polenc-1,N_d13)+1;
    tmp=ceil(polenc/N_d13);
    midpart=rem(tmp-1,N_a1)+1;
    L2offset=ceil(tmp/N_a1);
    adjust=(L2offset<1+n2short+1);
    a1prime_low=midpart-adjust;
    L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);
    d1part=rem(d13part-1,N_d1)+1;
    d3part=rem(ceil(d13part/N_d1)-1,N_d3)+1;
    Policy(1,:,:,N_j)=reshape(d1part,[1,N_a,N_semiz]);
    Policy(2,:,:,N_j)=reshape(d2winner,[1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(d3part,[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(d4winner,[1,N_a,N_semiz]);
    Policy(5,:,:,N_j)=reshape(a1prime_low,[1,N_a,N_semiz]);
    Policy(6,:,:,N_j)=reshape(L2ind,[1,N_a,N_semiz]);
    Policy(7,:,:,N_j)=reshape(flagwinner,[1,N_a,N_semiz]);
end


%% Iterate backwards
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    % If there is a warm-glow, evaluate the warmglowfn
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        % Switch WGmatrix from being in terms of a2prime to being in terms of d23 (in expectation because of the u shocks)
        skipinterpWG=logical(WGmatrix(a2primeIndex)==WGmatrix(a2primeIndex+1));
        aprimeProbsWG=a2primeProbs; % [N_d23,N_u]
        aprimeProbsWG(skipinterpWG)=0;
        WG1=WGmatrix(a2primeIndex).*aprimeProbsWG; % probability of lower grid point
        WG2=WGmatrix(a2primeIndex+1).*(1-aprimeProbsWG); % probability of upper grid point
        WG1(isnan(WG1))=0;
        WG2(isnan(WG2))=0;
        WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2); % [N_d23,1]
    end

    EVpre=V(:,:,jj+1);

    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    semiz_gridvals=semiz_gridvals_J(:,:,jj);

    % per-period inner (inlined)
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c); % no kron in noz
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            % Expectation over semiz' (on the transformed object)
            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            % u-lottery on the transformed object
            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
            EV=reshape(EV,[N_d23,N_a1,N_semiz]);

            % Interpolate the transformed EV over a1prime (BEFORE the certainty-equivalent power ^ezc6)
            EVinterp=permute(interp1(a1_gridvals,permute(EV,[2,1,3]),a1prime_grid),[2,1,3]); % [N_d23,N_a1prime,N_semiz]

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise
            temp4=EV;
            temp4interp=EVinterp;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,N_a1,N_semiz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV==0)&(WGmatrixbig==0))=0; % Is actually zero
                WGmatrixbigfine=WGmatrix.*ones(1,N_a1prime,N_semiz);
                becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixbigfine)); % both are finite
                temp4interp(becareful)=(sj(jj)*temp4interp(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbigfine(becareful).^ezc8(jj)).^ezc6(jj);
                temp4interp((EVinterp==0)&(WGmatrixbigfine==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV==0)=0;
                temp4interp(isfinite(temp4interp))=(sj(jj)*temp4interp(isfinite(temp4interp)).^ezc8(jj)).^ezc6(jj);
                temp4interp(EVinterp==0)=0;
            end

            % Refine d2 out of the continuation (ezc9 handles the sign so the max is correct), coarse and fine
            temp4_onlyd3=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,N_semiz]),[],1);
            [temp4interp_onlyd3,d2indexfine]=max(ezc9*ezc3*reshape((~isinf(temp4interp)).*temp4interp,[N_d2,N_d3*N_a1prime,N_semiz]),[],1);
            d2indexfine_resh=reshape(d2indexfine,[N_d3,N_a1prime,N_semiz]);

            % DiscountedEV (the ezc9 outside undoes the sign flip used inside the refine)
            DiscountedEV=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1,N_semiz]);
            DiscountedEVinterp=DiscountFactorParamsVec*ezc9*reshape(temp4interp_onlyd3,[N_d3,N_a1prime,1,1,N_semiz]);
            DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
            DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

            % Level-1 Return at coarse a1prime grid
            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,n_a1,n_a2,n_semiz, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals, ReturnFnParamsVec,1,0);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_d4).*(ReturnMatrix_d4~=0)); % finite but not zero
            temp2=ReturnMatrix_d4;
            temp2(becareful)=ReturnMatrix_d4(becareful).^ezc2(jj);
            temp2(ReturnMatrix_d4==0)=-Inf;
            entireRHS=ezc1*temp2+DiscountedEV_d13;
            temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
            entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS(entireRHS==0)=-Inf;

            [~,maxindex]=max(entireRHS,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals, ReturnFnParamsVec,2,0);
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;
            da1primez=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1)+N_d13*N_a1prime*zind;
            entireRHS_ii=ezc1*temp2_ii+reshape(DiscountedEVinterp_d13(da1primez(:)),[N_d13*n2long,N_a1*N_a2,N_semiz]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zindB;
            mid_at=shiftdim(squeeze(midpoint(allind)),-1);
            L2offset=ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zindB;
            isInfLower    = (temp2_ii(linidx_lower) == -Inf);
            isInfUpper    = (temp2_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
            % d2 lookup per (a,semiz) — read off the FINE refine at the chosen fine a1prime point
            d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
            a1mid=midpoint(allind);
            a1fine=(a1mid+(a1mid-1)*n2short)+L2offset-n2short-2; % chosen fine a1prime index
            zlin=shiftdim(gpuArray(0:N_semiz-1),-1);
            linlookup=d3opt+N_d3*(a1fine-1)+N_d3*N_a1prime*zlin;
            d2index_ford4_jj(:,:,d4_c)=shiftdim(d2indexfine_resh(linlookup),1);
        end

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        special_n_semiz=ones(1,length(n_semiz));
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals(z_c,:);

                % Expectation over semiz' (on the transformed object)
                EV_z=temp.*pi_semizd4(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);
                EV_z=reshape(EV_z,[N_a,1]);

                % u-lottery on the transformed object
                skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
                aprimeProbs=repmat(a2primeProbs,N_a1,1);
                aprimeProbs(skipinterp)=0;
                aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);

                EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
                EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
                EV_z=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
                EV_z=reshape(EV_z,[N_d23,N_a1]);

                % Interpolate the transformed EV over a1prime (BEFORE the certainty-equivalent power ^ezc6)
                EVinterp_z=permute(interp1(a1_gridvals,permute(EV_z,[2,1]),a1prime_grid),[2,1]); % [N_d23,N_a1prime]

                % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise
                temp4=EV_z;
                temp4interp=EVinterp_z;
                if warmglow==1
                    WGmatrixbig=WGmatrix.*ones(1,N_a1);
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_z==0)&(WGmatrixbig==0))=0; % Is actually zero
                    WGmatrixbigfine=WGmatrix.*ones(1,N_a1prime);
                    becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixbigfine)); % both are finite
                    temp4interp(becareful)=(sj(jj)*temp4interp(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbigfine(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4interp((EVinterp_z==0)&(WGmatrixbigfine==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_z==0)=0;
                    temp4interp(isfinite(temp4interp))=(sj(jj)*temp4interp(isfinite(temp4interp)).^ezc8(jj)).^ezc6(jj);
                    temp4interp(EVinterp_z==0)=0;
                end

                % Refine d2 out of the continuation (ezc9 handles the sign so the max is correct), coarse and fine
                temp4_onlyd3=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1]),[],1);
                [temp4interp_onlyd3,d2indexfine]=max(ezc9*ezc3*reshape((~isinf(temp4interp)).*temp4interp,[N_d2,N_d3*N_a1prime]),[],1);
                d2indexfine_z=reshape(d2indexfine,[N_d3,N_a1prime]);

                % DiscountedEV (the ezc9 outside undoes the sign flip used inside the refine)
                DiscountedEV_z=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1]);
                DiscountedEVinterp_z=DiscountFactorParamsVec*ezc9*reshape(temp4interp_onlyd3,[N_d3,N_a1prime,1,1]);
                DiscountedEV_d13_z=repelem(DiscountedEV_z,N_d1,1);
                DiscountedEVinterp_d13_z=repelem(DiscountedEVinterp_z,N_d1,1);

                ReturnMatrix_d4z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,n_a1,n_a2,special_n_semiz, d13_with_d4, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_d4z).*(ReturnMatrix_d4z~=0)); % finite but not zero
                temp2=ReturnMatrix_d4z;
                temp2(becareful)=ReturnMatrix_d4z(becareful).^ezc2(jj);
                temp2(ReturnMatrix_d4z==0)=-Inf;
                entireRHS_z=ezc1*temp2+DiscountedEV_d13_z;
                temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
                entireRHS_z(temp5)=entireRHS_z(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_z(entireRHS_z==0)=-Inf;

                [~,maxindex]=max(entireRHS_z,[],2);

                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_semiz, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                becareful=logical(isfinite(ReturnMatrix_ii_z).*(ReturnMatrix_ii_z~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii_z;
                temp2_ii(becareful)=ReturnMatrix_ii_z(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_ii_z==0)=-Inf;
                da1prime=(1:1:N_d13)'+N_d13*(a1primeindexesfine-1);
                entireRHS_ii_z=ezc1*temp2_ii+reshape(DiscountedEVinterp_d13_z(da1prime(:)),[N_d13*n2long,N_a1*N_a2]);
                temp5=logical(isfinite(entireRHS_ii_z).*(entireRHS_ii_z~=0));
                entireRHS_ii_z(temp5)=entireRHS_ii_z(temp5).^ezc7(jj);
                entireRHS_ii_z(entireRHS_ii_z==0)=-Inf;
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind;
                mid_at=midpoint(allind);
                L2offset=ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                isInfLower    = (temp2_ii(linidx_lower) == -Inf);
                isInfUpper    = (temp2_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,z_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(L2offset,1)-1);
                % d2 lookup per a — read off the FINE refine at the chosen fine a1prime point
                d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                a1mid=midpoint(allind);
                a1fine=(a1mid+(a1mid-1)*n2short)+L2offset-n2short-2; % chosen fine a1prime index
                linlookup=d3opt+N_d3*(a1fine-1);
                d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2indexfine_z(linlookup),1);
            end
        end
    end

    % combine across d4 (inlined)
    [Vbest,d4winner]=max(V_ford4_jj,[],3);
    V(:,:,jj)=Vbest;
    Ncomb=N_a*N_semiz;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_semiz]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_semiz]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_semiz]);
    d13part=rem(polenc-1,N_d13)+1;
    tmp=ceil(polenc/N_d13);
    midpart=rem(tmp-1,N_a1)+1;
    L2offset=ceil(tmp/N_a1);
    adjust=(L2offset<1+n2short+1);
    a1prime_low=midpart-adjust;
    L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);
    d1part=rem(d13part-1,N_d1)+1;
    d3part=rem(ceil(d13part/N_d1)-1,N_d3)+1;
    Policy(1,:,:,jj)=reshape(d1part,[1,N_a,N_semiz]);
    Policy(2,:,:,jj)=reshape(d2winner,[1,N_a,N_semiz]);
    Policy(3,:,:,jj)=reshape(d3part,[1,N_a,N_semiz]);
    Policy(4,:,:,jj)=reshape(d4winner,[1,N_a,N_semiz]);
    Policy(5,:,:,jj)=reshape(a1prime_low,[1,N_a,N_semiz]);
    Policy(6,:,:,jj)=reshape(L2ind,[1,N_a,N_semiz]);
    Policy(7,:,:,jj)=reshape(flagwinner,[1,N_a,N_semiz]);
end


end
