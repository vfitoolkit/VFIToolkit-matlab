function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_DC1_GI1_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_e,n_u,N_j, d2_grid, d3_grid, a1_grid, a2_grid, e_gridvals_J, u_grid, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No d1, no z, with e.
%
% Divide-and-conquer plus grid-interpolation-layer version of the Epstein-Zin
% riskyasset solver. Grafts the Epstein-Zin transforms onto
% ValueFnIter_FHorz_RiskyAsset_DC1_GI1_nod1_noz_e_raw. V' is transformed
% ((ezc4*V')^ezc5, masked) BEFORE all the expectations: the e'-expectation and
% the u-lottery act on the transformed object, giving one joint
% certainty-equivalent over (u,e'). The interpolation over a1prime acts on the
% transformed expectations object BEFORE the certainty-equivalent power ^ezc6
% (exact collapses under GI); the ^ezc6 transform is applied pointwise to both
% the coarse and fine objects AFTER the interpolation. d2 is refined out AFTER
% the full transform chain using the ezc9 sign trick (coarse and fine
% separately; the d2 policy is read off the fine refine at the chosen fine
% a1prime point). The return transform and the final ^ezc7 wrap each entireRHS
% before its max at every DC level. The warm-glow fn depends only on a2prime,
% so the u-lottery turns it into a function of d23 (constant in a1prime, hence
% identical on coarse and fine grids).

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_e=prod(n_e);
N_u=prod(n_u);

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_e,N_j,'gpuArray');
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d3_gridvals=CreateGridvals(n_d3,d3_grid,1);

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Setup for GI
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

% Precompute
aind=gpuArray(0:1:N_a-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-1);

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
a2Bind=gpuArray(0:1:N_a2-1);
d3ind=gpuArray(1:1:N_d3)';

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
        WGcol=ezc9*reshape(WGmatrix_onlyd3(d3ind),[N_d3,1]); % constant in a1prime (and a,e)
    else
        WGcol=zeros(N_d3,1,'gpuArray');
    end

    if vfoptions.lowmemory==0
        % Setup for DC
        midpoint_jj=zeros(N_d3,1,N_a1,N_a2,N_e,'gpuArray');
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
        entireRHS_ii=ReturnMatrix_ii+WGcol; % warm-glow (zero if not using); constant in a1prime so does not affect the layer maxes over a1prime

        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint_jj(:,1,level1ii,:,:)=maxindex1;

        % Divide-and-conquer layer 2
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                entireRHS_ii=ReturnMatrix_ii+WGcol;
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoint_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoint_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Grid interpolation layer
        midpoint_jj=max(min(midpoint_jj,n_a1(1)-1),2);
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
        entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d3,n2long,N_a1,N_a2,N_e])+WGcol,[N_d3*n2long,N_a1*N_a2,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d3_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d3_ind+N_d3*aind+N_d3*N_a*eBind;
        Policy(2,:,:,N_j)=d3_ind; % d3
        Policy(3,:,:,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);
        Policy(4,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);

        % L2flag
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*eBind;
        linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*eBind;
        ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d3,n2long,N_a1,N_a2,N_e]);
        isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

        % d2, which was not in ReturnFn
        if warmglow==1
            Policy(1,:,:,N_j)=d2index(d3_ind); % d2 comes from refining the warm-glow (no a nor e in WGmatrix)
        end

    elseif vfoptions.lowmemory>=1
        special_n_e=ones(1,length(n_e));
        % Setup for DC
        midpoint_jj=zeros(N_d3,1,N_a1,N_a2,'gpuArray');
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, ReturnFnParamsVec,1,0);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii_e).*(ReturnMatrix_ii_e~=0)); % finite but not zero
            ReturnMatrix_ii_e(becareful)=(ezc1*ReturnMatrix_ii_e(becareful).^ezc2(N_j)).^ezc7(N_j);
            ReturnMatrix_ii_e(ReturnMatrix_ii_e==0)=-Inf;
            entireRHS_ii=ReturnMatrix_ii_e+WGcol;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint_jj(:,1,level1ii,:)=maxindex1;

            % Divide-and-conquer layer 2
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii_e).*(ReturnMatrix_ii_e~=0)); % finite but not zero
                    ReturnMatrix_ii_e(becareful)=(ezc1*ReturnMatrix_ii_e(becareful).^ezc2(N_j)).^ezc7(N_j);
                    ReturnMatrix_ii_e(ReturnMatrix_ii_e==0)=-Inf;
                    entireRHS_ii=ReturnMatrix_ii_e+WGcol;
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoint_jj(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Grid interpolation layer
            midpoint_jj=max(min(midpoint_jj,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,special_n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0);
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
            ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
            entireRHS_ii=reshape(reshape(ReturnMatrix_ii,[N_d3,n2long,N_a1,N_a2])+WGcol,[N_d3*n2long,N_a1*N_a2]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            d3_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d3_ind+N_d3*aind;
            Policy(2,:,e_c,N_j)=d3_ind; % d3
            Policy(3,:,e_c,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);
            Policy(4,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);

            % L2flag
            L2offset      = ceil(maxindexL2/N_d3);
            linidx_lower  = d3_ind                   + N_d3*n2long*aind;
            linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
            ReturnMatrix_ii_resh=reshape(ReturnMatrix_ii,[N_d3,n2long,N_a1,N_a2]);
            isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

            % d2, which was not in ReturnFn
            if warmglow==1
                Policy(1,:,e_c,N_j)=d2index(d3_ind); % d2 comes from refining the warm-glow (no a nor e in WGmatrix)
            end
        end
    end

    if warmglow==0
        % d2, which was not in ReturnFn
        Policy(1,:,:,N_j)=ones(1,N_a,N_e,'gpuArray'); % d2 (because this is terminal period, choice of d2 is not actually doing anything (as it is only in the expectations term)
    end

else % V_Jplus1

    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);

    if warmglow==0 % if warmglow==1 these were already created above
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
        [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    end
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0;

    % e'-expectation on the transformed object
    EV=sum(temp.*shiftdim(pi_e_J(:,N_j+1),-1),2); % [N_a,1]

    % u-lottery on the transformed object
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a2primeProbs,N_a1,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    EV=reshape(EV,[N_d23,N_a1]);

    % Interpolate the transformed EV over a1prime (BEFORE the certainty-equivalent power ^ezc6)
    EVinterp=permute(interp1(a1_gridvals,permute(EV,[2,1]),a1prime_grid),[2,1]); % [N_d23,N_a1prime]

    % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise
    temp4=EV;
    temp4interp=EVinterp;
    if warmglow==1
        WGmatrixbig=WGmatrix.*ones(1,N_a1);
        becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
        temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
        temp4((EV==0)&(WGmatrixbig==0))=0; % Is actually zero
        WGmatrixbigfine=WGmatrix.*ones(1,N_a1prime);
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
    temp4_onlyd3=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1]),[],1);
    [temp4interp_onlyd3,d2indexfine]=max(ezc9*ezc3*reshape((~isinf(temp4interp)).*temp4interp,[N_d2,N_d3*N_a1prime]),[],1);
    d2indexfine_resh=reshape(d2indexfine,[N_d3,N_a1prime]);

    % DiscountedEV (the ezc9 outside undoes the sign flip used inside the refine)
    DiscountedEV=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1]);
    DiscountedEVinterp=DiscountFactorParamsVec*ezc9*reshape(temp4interp_onlyd3,[N_d3,N_a1prime,1,1]);

    if vfoptions.lowmemory==0
        % Setup for DC
        midpoint_jj=zeros(N_d3,1,N_a1,N_a2,N_e,'gpuArray');
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        % [N_d3, level1n, N_a1, N_a2, N_e]; broadcast DiscountedEV across a2, e
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
        temp2_ii(ReturnMatrix_ii==0)=-Inf;
        RM=reshape(temp2_ii,[N_d3,N_a1,vfoptions.level1n,N_a2,N_e]);
        DEV=reshape(DiscountedEV,[N_d3,N_a1,1,1,1]);
        entireRHS_ii=ezc1*RM+DEV;
        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
        entireRHS_ii(entireRHS_ii==0)=-Inf;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint_jj(:,1,level1ii,:,:)=maxindex1;

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                d3aprime=d3ind+N_d3*(a1primeindexes-1); % [N_d3,maxgap+1,1,N_a2]; broadcasts across e
                entireRHS_ii=ezc1*temp2_ii+DiscountedEV(d3aprime);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoint_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoint_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Grid interpolation layer
        midpoint_jj=max(min(midpoint_jj,n_a1(1)-1),2);
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
        temp2_ii(ReturnMatrix_ii==0)=-Inf;
        da1prime=d3ind+N_d3*(a1primeindexesfine-1); % [N_d3,n2long,N_a1,N_a2,N_e]
        entireRHS_ii=reshape(ezc1*reshape(temp2_ii,[N_d3,n2long,N_a1,N_a2,N_e])+reshape(DiscountedEVinterp(da1prime),[N_d3,n2long,N_a1,N_a2,N_e]),[N_d3*n2long,N_a1*N_a2,N_e]);
        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
        entireRHS_ii(entireRHS_ii==0)=-Inf;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d3_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d3_ind+N_d3*aind+N_d3*N_a*eBind;
        Policy(2,:,:,N_j)=d3_ind; % d3
        Policy(3,:,:,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);
        Policy(4,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);

        % L2flag (on the transformed return matrix: -Inf marks both infeasible and zero-return points)
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*eBind;
        linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*eBind;
        ReturnMatrix_ii_resh=reshape(temp2_ii,[N_d3,n2long,N_a1,N_a2,N_e]);
        isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

        % Get the d2Policy (read off the fine refine at the chosen fine a1prime point)
        a1mid=squeeze(midpoint_jj(allind));
        a1fine=(a1mid+(a1mid-1)*n2short)+shiftdim(L2offset,1)-n2short-2; % chosen fine a1prime index
        linlookup=shiftdim(d3_ind,1)+N_d3*(a1fine-1);
        Policy(1,:,:,N_j)=shiftdim(d2indexfine_resh(linlookup),-1);

    elseif vfoptions.lowmemory>=1
        % Setup for DC
        midpoint_jj=zeros(N_d3,1,N_a1,N_a2,'gpuArray');
        special_n_e=ones(1,length(n_e));
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, ReturnFnParamsVec,1,0);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii_e).*(ReturnMatrix_ii_e~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii_e;
            temp2_ii(becareful)=ReturnMatrix_ii_e(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_ii_e==0)=-Inf;
            RM=reshape(temp2_ii,[N_d3,N_a1,vfoptions.level1n,N_a2]);
            DEV=reshape(DiscountedEV,[N_d3,N_a1,1,1]);
            entireRHS_ii_e=ezc1*RM+DEV;
            temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
            entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);
            entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            midpoint_jj(:,1,level1ii,:)=maxindex1;

            % Divide and conquer layer 2
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii_e).*(ReturnMatrix_ii_e~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii_e;
                    temp2_ii(becareful)=ReturnMatrix_ii_e(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii_e==0)=-Inf;
                    d3aprime=d3ind+N_d3*(a1primeindexes-1);
                    entireRHS_ii_e=ezc1*temp2_ii+DiscountedEV(d3aprime);
                    temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                    entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);
                    entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                    [~,maxindex]=max(entireRHS_ii_e,[],2);
                    midpoint_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoint_jj(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Grid interpolation layer
            midpoint_jj=max(min(midpoint_jj,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,special_n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0);
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;
            da1prime=d3ind+N_d3*(a1primeindexesfine-1);
            entireRHS_ii=reshape(ezc1*reshape(temp2_ii,[N_d3,n2long,N_a1,N_a2])+reshape(DiscountedEVinterp(da1prime),[N_d3,n2long,N_a1,N_a2]),[N_d3*n2long,N_a1*N_a2]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            d3_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d3_ind+N_d3*aind;
            Policy(2,:,e_c,N_j)=d3_ind; % d3
            Policy(3,:,e_c,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);
            Policy(4,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);

            % L2flag (on the transformed return matrix: -Inf marks both infeasible and zero-return points)
            L2offset      = ceil(maxindexL2/N_d3);
            linidx_lower  = d3_ind                   + N_d3*n2long*aind;
            linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
            ReturnMatrix_ii_resh=reshape(temp2_ii,[N_d3,n2long,N_a1,N_a2]);
            isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

            % Get the d2Policy (read off the fine refine at the chosen fine a1prime point)
            a1mid=squeeze(midpoint_jj(allind));
            a1fine=(a1mid+(a1mid-1)*n2short)+L2offset-n2short-2; % chosen fine a1prime index
            linlookup=d3_ind+N_d3*(a1fine-1);
            Policy(1,:,e_c,N_j)=shiftdim(d2indexfine_resh(linlookup),-1);
        end
    end
end

%% Iterate backwards
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj));
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % Build a2primeIndex and a2primeProbs for RisykAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

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

    % Get EV in terms of next period endogenous states
    EVpre=V(:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    % e'-expectation on the transformed object
    EV=sum(temp.*shiftdim(pi_e_J(:,jj+1),-1),2); % [N_a,1]

    % u-lottery on the transformed object
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a2primeProbs,N_a1,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    EV=reshape(EV,[N_d23,N_a1]);

    % Interpolate the transformed EV over a1prime (BEFORE the certainty-equivalent power ^ezc6)
    EVinterp=permute(interp1(a1_gridvals,permute(EV,[2,1]),a1prime_grid),[2,1]); % [N_d23,N_a1prime]

    % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise
    temp4=EV;
    temp4interp=EVinterp;
    if warmglow==1
        WGmatrixbig=WGmatrix.*ones(1,N_a1);
        becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
        temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
        temp4((EV==0)&(WGmatrixbig==0))=0; % Is actually zero
        WGmatrixbigfine=WGmatrix.*ones(1,N_a1prime);
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
    temp4_onlyd3=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1]),[],1);
    [temp4interp_onlyd3,d2indexfine]=max(ezc9*ezc3*reshape((~isinf(temp4interp)).*temp4interp,[N_d2,N_d3*N_a1prime]),[],1);
    d2indexfine_resh=reshape(d2indexfine,[N_d3,N_a1prime]);

    % DiscountedEV (the ezc9 outside undoes the sign flip used inside the refine)
    DiscountedEV=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1]);
    DiscountedEVinterp=DiscountFactorParamsVec*ezc9*reshape(temp4interp_onlyd3,[N_d3,N_a1prime,1,1]);

    if vfoptions.lowmemory==0
        % Setup for DC
        midpoint_jj=zeros(N_d3,1,N_a1,N_a2,N_e,'gpuArray');
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
        % [N_d3, level1n, N_a1, N_a2, N_e]; broadcast DiscountedEV across a2, e
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
        temp2_ii(ReturnMatrix_ii==0)=-Inf;
        RM=reshape(temp2_ii,[N_d3,N_a1,vfoptions.level1n,N_a2,N_e]);
        DEV=reshape(DiscountedEV,[N_d3,N_a1,1,1,1]);
        entireRHS_ii=ezc1*RM+DEV;
        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
        entireRHS_ii(entireRHS_ii==0)=-Inf;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint_jj(:,1,level1ii,:,:)=maxindex1;

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                d3aprime=d3ind+N_d3*(a1primeindexes-1); % [N_d3,maxgap+1,1,N_a2]; broadcasts across e
                entireRHS_ii=ezc1*temp2_ii+DiscountedEV(d3aprime);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoint_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoint_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Grid interpolation layer
        midpoint_jj=max(min(midpoint_jj,n_a1(1)-1),2);
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
        temp2_ii(ReturnMatrix_ii==0)=-Inf;
        da1prime=d3ind+N_d3*(a1primeindexesfine-1); % [N_d3,n2long,N_a1,N_a2,N_e]
        entireRHS_ii=reshape(ezc1*reshape(temp2_ii,[N_d3,n2long,N_a1,N_a2,N_e])+reshape(DiscountedEVinterp(da1prime),[N_d3,n2long,N_a1,N_a2,N_e]),[N_d3*n2long,N_a1*N_a2,N_e]);
        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
        entireRHS_ii(entireRHS_ii==0)=-Inf;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);
        d3_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d3_ind+N_d3*aind+N_d3*N_a*eBind;
        Policy(2,:,:,jj)=d3_ind; % d3
        Policy(3,:,:,jj)=shiftdim(squeeze(midpoint_jj(allind)),-1);
        Policy(4,:,:,jj)=shiftdim(ceil(maxindexL2/N_d3),-1);

        % L2flag (on the transformed return matrix: -Inf marks both infeasible and zero-return points)
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*eBind;
        linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*eBind;
        ReturnMatrix_ii_resh=reshape(temp2_ii,[N_d3,n2long,N_a1,N_a2,N_e]);
        isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,jj) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

        % Get the d2Policy (read off the fine refine at the chosen fine a1prime point)
        a1mid=squeeze(midpoint_jj(allind));
        a1fine=(a1mid+(a1mid-1)*n2short)+shiftdim(L2offset,1)-n2short-2; % chosen fine a1prime index
        linlookup=shiftdim(d3_ind,1)+N_d3*(a1fine-1);
        Policy(1,:,:,jj)=shiftdim(d2indexfine_resh(linlookup),-1);

    elseif vfoptions.lowmemory>=1
        % Setup for DC
        midpoint_jj=zeros(N_d3,1,N_a1,N_a2,'gpuArray');
        special_n_e=ones(1,length(n_e));
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, ReturnFnParamsVec,1,0);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii_e).*(ReturnMatrix_ii_e~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii_e;
            temp2_ii(becareful)=ReturnMatrix_ii_e(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_ii_e==0)=-Inf;
            RM=reshape(temp2_ii,[N_d3,N_a1,vfoptions.level1n,N_a2]);
            DEV=reshape(DiscountedEV,[N_d3,N_a1,1,1]);
            entireRHS_ii_e=ezc1*RM+DEV;
            temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
            entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);
            entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            midpoint_jj(:,1,level1ii,:)=maxindex1;

            % Divide and conquer layer 2
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii_e).*(ReturnMatrix_ii_e~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii_e;
                    temp2_ii(becareful)=ReturnMatrix_ii_e(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii_e==0)=-Inf;
                    d3aprime=d3ind+N_d3*(a1primeindexes-1);
                    entireRHS_ii_e=ezc1*temp2_ii+DiscountedEV(d3aprime);
                    temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                    entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);
                    entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                    [~,maxindex]=max(entireRHS_ii_e,[],2);
                    midpoint_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoint_jj(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Grid interpolation layer
            midpoint_jj=max(min(midpoint_jj,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,special_n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0);
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;
            da1prime=d3ind+N_d3*(a1primeindexesfine-1);
            entireRHS_ii=reshape(ezc1*reshape(temp2_ii,[N_d3,n2long,N_a1,N_a2])+reshape(DiscountedEVinterp(da1prime),[N_d3,n2long,N_a1,N_a2]),[N_d3*n2long,N_a1*N_a2]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,e_c,jj)=shiftdim(Vtempii,1);
            d3_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d3_ind+N_d3*aind;
            Policy(2,:,e_c,jj)=d3_ind; % d3
            Policy(3,:,e_c,jj)=shiftdim(squeeze(midpoint_jj(allind)),-1);
            Policy(4,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d3),-1);

            % L2flag (on the transformed return matrix: -Inf marks both infeasible and zero-return points)
            L2offset      = ceil(maxindexL2/N_d3);
            linidx_lower  = d3_ind                   + N_d3*n2long*aind;
            linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
            ReturnMatrix_ii_resh=reshape(temp2_ii,[N_d3,n2long,N_a1,N_a2]);
            isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,e_c,jj) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

            % Get the d2Policy (read off the fine refine at the chosen fine a1prime point)
            a1mid=squeeze(midpoint_jj(allind));
            a1fine=(a1mid+(a1mid-1)*n2short)+L2offset-n2short-2; % chosen fine a1prime index
            linlookup=d3_ind+N_d3*(a1fine-1);
            Policy(1,:,e_c,jj)=shiftdim(d2indexfine_resh(linlookup),-1);
        end
    end
end


%% Switch Policy(3,:) from 'midpoint' to 'lower grid index'
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

end
