function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_SemiExo_DC1_GI1_noz_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% No z (only semiz); e iid.
%
% Divide-and-conquer plus grid-interpolation-layer version of the Epstein-Zin
% riskyasset+semiz solver. Grafts the Epstein-Zin transforms onto
% ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_noz_e_raw. V' is
% transformed ((ezc4*V')^ezc5, masked) BEFORE all the expectations: the e'
% expectation (e is iid, integrated first), the semiz' expectation (inside the
% d4 loop) and the u-lottery all act on the transformed object, giving one
% joint certainty-equivalent over (u,semiz',e'). The interpolation over
% a1prime acts on the transformed expectations object BEFORE the
% certainty-equivalent power ^ezc6 (exact collapses under GI); the ^ezc6
% transform is applied pointwise to both the coarse and fine objects AFTER the
% interpolation. d2 is refined out AFTER the full transform chain using the
% ezc9 sign trick (coarse and fine separately; the d2 policy is read off the
% fine refine at the chosen fine a1prime point). The return transform and the
% final ^ezc7 wrap each entireRHS before its max at every DC level.
% The warm-glow fn depends only on a2prime, so the u-lottery turns it into a
% function of d23 (constant in a1prime, hence identical on coarse and fine grids).

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
special_n_d4=ones(1,length(n_d4));
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);
N_u=prod(n_u);

N_d13=N_d1*N_d3;

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz,N_e,N_j,'gpuArray');
d2Policy=ones(1,N_a,N_semiz,N_e,N_j,'gpuArray');
d4Policy=ones(1,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
u_grid=gpuArray(u_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
d23_grid=gpuArray(d23_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d13_gridvals=gpuArray(CreateGridvals([n_d1,n_d3],[d1_grid;d3_grid],1));
d1d3d4a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,n_d4,n_a1],[d1_grid;d3_grid;d4_grid;a1_grid],1));
a1a2_gridvals=gpuArray(CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

pi_u_col=pi_u(:);

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
d3ind=repelem(gpuArray(1:1:N_d3)',N_d1,1);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>=2
    special_n_semiz=ones(1,length(n_semiz));
end

V_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
Policy3_ford4_jj=zeros(3,N_a,N_semiz,N_e,N_d4,'gpuArray');
flag_ford4_jj=2*ones(N_a,N_semiz,N_e,N_d4,'gpuArray');
d2_ford4_jj=ones(N_a,N_semiz,N_e,N_d4,'gpuArray');


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
    WGmatrix=sum((WG1.*pi_u_col'),2)+sum((WG2.*pi_u_col'),2); % [N_d23,1]

    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrix(becareful)=0;
    end
else
    WGmatrix=0;
end

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, n_e, d1d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
    ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2(N_j)).^ezc7(N_j);
    ReturnMatrix(ReturnMatrix==0)=-Inf;

    if warmglow==1
        % Refine d2 out of the warm-glow (the only continuation term in the terminal period)
        [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3]),[],1);
        % Rows of ReturnMatrix are (d1,d3,d4,a1prime): spread the refined warm-glow over d1, d4 and a1prime (it is constant in all three)
        WGcol=ezc9*repmat(reshape(WGmatrix_onlyd3(d3ind),[N_d13,1]),N_d4*N_a1,1); % [N_d13*N_d4*N_a1,1]
        entireRHS=ReturnMatrix+WGcol;
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
        d1d3_ind=rem(dindex-1,N_d13)+1;
        d1part=rem(d1d3_ind-1,N_d1)+1;
        d3part=ceil(d1d3_ind/N_d1);
        d4part=ceil(dindex/N_d13);
        a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
        Policy3(1,:,:,:,N_j)=shiftdim(d1part+N_d1*(d3part-1),-1);
        Policy3(2,:,:,:,N_j)=shiftdim(a1primepart,-1);
        Policy3(3,:,:,:,N_j)=n2short+2;
        d4Policy(1,:,:,:,N_j)=shiftdim(d4part,-1);
        d2Policy(1,:,:,:,N_j)=d2index(d3part); % d2 comes from refining the warm-glow (no a nor semiz nor e in WGmatrix)
    else
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
        d1d3_ind=rem(dindex-1,N_d13)+1;
        d1part=rem(d1d3_ind-1,N_d1)+1;
        d3part=ceil(d1d3_ind/N_d1);
        d4part=ceil(dindex/N_d13);
        a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
        Policy3(1,:,:,:,N_j)=shiftdim(d1part+N_d1*(d3part-1),-1);
        Policy3(2,:,:,:,N_j)=shiftdim(a1primepart,-1);
        Policy3(3,:,:,:,N_j)=n2short+2;
        d4Policy(1,:,:,:,N_j)=shiftdim(d4part,-1);
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]);
    if warmglow==0 % if warmglow==1 these were already created above
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
        [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    end

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end
    semiz_gridvals=semiz_gridvals_J(:,:,N_j);
    e_gridvals=e_gridvals_J(:,:,N_j);

    V_jj=zeros(N_a,N_semiz,N_e,'gpuArray');
    Policy3_jj=zeros(3,N_a,N_semiz,N_e,'gpuArray');
    PolicyL2flag_jj=2*ones(1,N_a,N_semiz,N_e,'gpuArray');
    d2Policy_jj=ones(1,N_a,N_semiz,N_e,'gpuArray');
    d4Policy_jj=ones(1,N_a,N_semiz,N_e,'gpuArray');

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Part of Epstein-Zin is before taking expectation
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0;
    % Integrate over e' first (e is iid); part of the same joint certainty-equivalent as (u,semiz')
    temp=sum(temp.*shiftdim(pi_e_J(:,N_j+1),-2),3);

    for d4_c=1:N_d4
        pi_semizd4=pi_semiz(:,:,d4_c); % no kron in noz case
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        % EV / d2index / DiscountedEV are independent of e — compute once per d4
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
        EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
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

        if vfoptions.lowmemory==0
            midpoint=zeros(N_d13,1,N_a1,N_a2,N_semiz,N_e,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(RM).*(RM~=0)); % finite but not zero
            temp2_ii=RM;
            temp2_ii(becareful)=RM(becareful).^ezc2(N_j);
            temp2_ii(RM==0)=-Inf;
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_semiz,1]);
            entireRHS_ii=ezc1*temp2_ii+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_ii(entireRHS_ii==0)=-Inf;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=ezc1*temp2_ii+DiscountedEV(d3aprimez);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz,n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,2,0);
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;
            da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBind,-2);
            entireRHS_ii=reshape(ezc1*reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,N_semiz,N_e])+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_semiz,N_e]),[N_d13*n2long,N_a1*N_a2,N_semiz,N_e]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zBind+N_d13*N_a*N_semiz*eBind;
            Policy3_ford4_jj(1,:,:,:,d4_c)=d_ind;
            Policy3_ford4_jj(2,:,:,:,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford4_jj(3,:,:,:,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

            % L2flag (on the transformed return matrix: -Inf marks both infeasible and zero-return points)
            L2offset      = ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBind + N_d13*n2long*N_a*N_semiz*eBind;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind + N_d13*n2long*N_a*N_semiz*eBind;
            ReturnMatrix_ii_resh=reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,N_semiz,N_e]);
            isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            % Get the d2Policy (read off the fine refine at the chosen fine a1prime point)
            d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
            a1mid=squeeze(midpoint(allind));
            a1fine=(a1mid+(a1mid-1)*n2short)+shiftdim(L2offset,1)-n2short-2; % chosen fine a1prime index
            zidx=repmat(gpuArray(reshape(1:N_semiz,[1,N_semiz,1])),N_a,1,N_e);
            linlookup=d3part+N_d3*(a1fine-1)+N_d3*N_a1prime*(zidx-1);
            d2_ford4_jj(:,:,:,d4_c)=d2indexfine_resh(linlookup);

        elseif vfoptions.lowmemory==1
            % Loop over e inside d4 to reduce memory footprint
            special_n_e=ones(1,length(n_e));
            for e_c=1:N_e
                e_val=e_gridvals(e_c,:);
                midpoint=zeros(N_d13,1,N_a1,N_a2,N_semiz,'gpuArray');

                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii_e,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(RM).*(RM~=0)); % finite but not zero
                temp2_ii=RM;
                temp2_ii(becareful)=RM(becareful).^ezc2(N_j);
                temp2_ii(RM==0)=-Inf;
                DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_semiz]);
                entireRHS_ii_e=ezc1*temp2_ii+DEV;
                entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;

                [~,maxindex1]=max(entireRHS_ii_e,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                        entireRHS_ii_e=ezc1*temp2_ii+DiscountedEV(d3aprimez);
                        temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                        entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);
                        entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                        [~,maxindex]=max(entireRHS_ii_e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,2,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBind,-2);
                entireRHS_ii_e=reshape(ezc1*reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,N_semiz])+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_semiz]),[N_d13*n2long,N_a1*N_a2,N_semiz]);
                temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);
                entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind+N_d13*N_a*zBind;
                Policy3_ford4_jj(1,:,:,e_c,d4_c)=d_ind;
                Policy3_ford4_jj(2,:,:,e_c,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford4_jj(3,:,:,e_c,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                % L2flag (on the transformed return matrix: -Inf marks both infeasible and zero-return points)
                L2offset      = ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
                ReturnMatrix_ii_resh=reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,N_semiz]);
                isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                % Get the d2Policy (read off the fine refine at the chosen fine a1prime point)
                d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
                a1mid=squeeze(midpoint(allind));
                a1fine=(a1mid+(a1mid-1)*n2short)+shiftdim(L2offset,1)-n2short-2; % chosen fine a1prime index
                zidx=repmat(gpuArray(1:N_semiz),N_a,1);
                linlookup=d3part+N_d3*(a1fine-1)+N_d3*N_a1prime*(zidx-1);
                d2_ford4_jj(:,:,e_c,d4_c)=d2indexfine_resh(linlookup);
            end
        elseif vfoptions.lowmemory>=2 % lm2 already does the most-looped variant, so it also serves the higher lowmemory values
            % Loop over semiz (outer) and e (inner) to reduce memory footprint
            special_n_e=ones(1,length(n_e));
            for z_c=1:N_semiz
                semiz_val=semiz_gridvals(z_c,:);
                DiscountedEV_zc=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_zc=DiscountedEVinterp(:,:,:,:,z_c);
                d2indexfine_resh_zc=d2indexfine_resh(:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    midpoint=zeros(N_d13,1,N_a1,N_a2,1,'gpuArray');

                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,1,0);
                    RM=reshape(ReturnMatrix_ii_e,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,1]);
                    % Modify the Return Function appropriately for Epstein-Zin Preferences
                    becareful=logical(isfinite(RM).*(RM~=0)); % finite but not zero
                    temp2_ii=RM;
                    temp2_ii(becareful)=RM(becareful).^ezc2(N_j);
                    temp2_ii(RM==0)=-Inf;
                    DEV=reshape(DiscountedEV_zc,[1,N_d3,N_a1,1,1,1]);
                    entireRHS_ii_e=ezc1*temp2_ii+DEV;
                    entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,1]);
                    temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                    entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                    entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;

                    [~,maxindex1]=max(entireRHS_ii_e,[],2);
                    midpoint(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,3,0);
                            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                            temp2_ii=ReturnMatrix_ii;
                            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                            temp2_ii(ReturnMatrix_ii==0)=-Inf;
                            d3aprimez=d3ind+N_d3*(a1primeindexes-1);
                            entireRHS_ii_e=ezc1*temp2_ii+DiscountedEV_zc(d3aprimez);
                            temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                            entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);
                            entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                            [~,maxindex]=max(entireRHS_ii_e,[],2);
                            midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_semiz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,2,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    da1primez=d3ind+N_d3*(a1primeindexesfine-1);
                    entireRHS_ii_e=reshape(ezc1*reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,1])+reshape(DiscountedEVinterp_zc(da1primez),[N_d13,n2long,N_a1,N_a2,1]),[N_d13*n2long,N_a1*N_a2,1]);
                    temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                    entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);
                    entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                    [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d13)+1;
                    allind=d_ind+N_d13*aind;
                    Policy3_ford4_jj(1,:,z_c,e_c,d4_c)=d_ind;
                    Policy3_ford4_jj(2,:,z_c,e_c,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford4_jj(3,:,z_c,e_c,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                    % L2flag (on the transformed return matrix: -Inf marks both infeasible and zero-return points)
                    L2offset      = ceil(maxindexL2/N_d13);
                    linidx_lower  = d_ind                    + N_d13*n2long*aind;
                    linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                    ReturnMatrix_ii_resh=reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,1]);
                    isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                    % Get the d2Policy (read off the fine refine at the chosen fine a1prime point)
                    d3part=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                    a1mid=midpoint(allind);
                    a1fine=(a1mid+(a1mid-1)*n2short)+L2offset-n2short-2; % chosen fine a1prime index
                    linlookup=d3part+N_d3*(a1fine-1);
                    d2_ford4_jj(:,z_c,e_c,d4_c)=d2indexfine_resh_zc(linlookup);
                end
            end
        end
    end

    % Cross-d4 max
    [V_jj,d4winner]=max(V_ford4_jj,[],4);
    N=N_a*N_semiz*N_e;
    P1=reshape(Policy3_ford4_jj(1,:,:,:,:),[N,N_d4]);
    P2=reshape(Policy3_ford4_jj(2,:,:,:,:),[N,N_d4]);
    P3=reshape(Policy3_ford4_jj(3,:,:,:,:),[N,N_d4]);
    F =reshape(flag_ford4_jj,[N,N_d4]);
    D2=reshape(d2_ford4_jj,[N,N_d4]);
    rowidx=(1:1:N)';
    gather_idx=rowidx+N*(reshape(d4winner,[N,1])-1);
    Policy3_jj(1,:,:,:)=shiftdim(reshape(P1(gather_idx),[N_a,N_semiz,N_e]),-1);
    Policy3_jj(2,:,:,:)=shiftdim(reshape(P2(gather_idx),[N_a,N_semiz,N_e]),-1);
    Policy3_jj(3,:,:,:)=shiftdim(reshape(P3(gather_idx),[N_a,N_semiz,N_e]),-1);
    PolicyL2flag_jj(1,:,:,:)=shiftdim(reshape(F(gather_idx),[N_a,N_semiz,N_e]),-1);
    d2Policy_jj(1,:,:,:)=shiftdim(reshape(D2(gather_idx),[N_a,N_semiz,N_e]),-1);
    d4Policy_jj(1,:,:,:)=shiftdim(d4winner,-1);

    V(:,:,:,N_j)=V_jj;
    Policy3(:,:,:,:,N_j)=Policy3_jj;
    PolicyL2flag(:,:,:,:,N_j)=PolicyL2flag_jj;
    d2Policy(:,:,:,:,N_j)=d2Policy_jj;
    d4Policy(:,:,:,:,N_j)=d4Policy_jj;
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
        WGmatrix=sum((WG1.*pi_u_col'),2)+sum((WG2.*pi_u_col'),2); % [N_d23,1]
    end

    EVpre=V(:,:,:,jj+1);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end
    % Local aliases so the inlined per-period body is otherwise verbatim
    semiz_gridvals=semiz_gridvals_J(:,:,jj);
    e_gridvals=e_gridvals_J(:,:,jj);

    V_jj=zeros(N_a,N_semiz,N_e,'gpuArray');
    Policy3_jj=zeros(3,N_a,N_semiz,N_e,'gpuArray');
    PolicyL2flag_jj=2*ones(1,N_a,N_semiz,N_e,'gpuArray');
    d2Policy_jj=ones(1,N_a,N_semiz,N_e,'gpuArray');
    d4Policy_jj=ones(1,N_a,N_semiz,N_e,'gpuArray');

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;
    % Integrate over e' first (e is iid); part of the same joint certainty-equivalent as (u,semiz')
    temp=sum(temp.*shiftdim(pi_e_J(:,jj+1),-2),3);

    for d4_c=1:N_d4
        pi_semizd4=pi_semiz(:,:,d4_c); % no kron in noz case
        d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

        % EV / d2index / DiscountedEV are independent of e — compute once per d4
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
        EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
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

        if vfoptions.lowmemory==0
            midpoint=zeros(N_d13,1,N_a1,N_a2,N_semiz,N_e,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,1,0);
            RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(RM).*(RM~=0)); % finite but not zero
            temp2_ii=RM;
            temp2_ii(becareful)=RM(becareful).^ezc2(jj);
            temp2_ii(RM==0)=-Inf;
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_semiz,1]);
            entireRHS_ii=ezc1*temp2_ii+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz,N_e]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_ii(entireRHS_ii==0)=-Inf;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=ezc1*temp2_ii+DiscountedEV(d3aprimez);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz,n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals, e_gridvals, ReturnFnParamsVec,2,0);
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;
            da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBind,-2);
            entireRHS_ii=reshape(ezc1*reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,N_semiz,N_e])+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_semiz,N_e]),[N_d13*n2long,N_a1*N_a2,N_semiz,N_e]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d13)+1;
            allind=d_ind+N_d13*aind+N_d13*N_a*zBind+N_d13*N_a*N_semiz*eBind;
            Policy3_ford4_jj(1,:,:,:,d4_c)=d_ind;
            Policy3_ford4_jj(2,:,:,:,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford4_jj(3,:,:,:,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

            % L2flag (on the transformed return matrix: -Inf marks both infeasible and zero-return points)
            L2offset      = ceil(maxindexL2/N_d13);
            linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBind + N_d13*n2long*N_a*N_semiz*eBind;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind + N_d13*n2long*N_a*N_semiz*eBind;
            ReturnMatrix_ii_resh=reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,N_semiz,N_e]);
            isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            % Get the d2Policy (read off the fine refine at the chosen fine a1prime point)
            d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
            a1mid=squeeze(midpoint(allind));
            a1fine=(a1mid+(a1mid-1)*n2short)+shiftdim(L2offset,1)-n2short-2; % chosen fine a1prime index
            zidx=repmat(gpuArray(reshape(1:N_semiz,[1,N_semiz,1])),N_a,1,N_e);
            linlookup=d3part+N_d3*(a1fine-1)+N_d3*N_a1prime*(zidx-1);
            d2_ford4_jj(:,:,:,d4_c)=d2indexfine_resh(linlookup);

        elseif vfoptions.lowmemory==1
            % Loop over e inside d4 to reduce memory footprint
            special_n_e=ones(1,length(n_e));
            for e_c=1:N_e
                e_val=e_gridvals(e_c,:);
                midpoint=zeros(N_d13,1,N_a1,N_a2,N_semiz,'gpuArray');

                ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,1,0);
                RM=reshape(ReturnMatrix_ii_e,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(RM).*(RM~=0)); % finite but not zero
                temp2_ii=RM;
                temp2_ii(becareful)=RM(becareful).^ezc2(jj);
                temp2_ii(RM==0)=-Inf;
                DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_semiz]);
                entireRHS_ii_e=ezc1*temp2_ii+DEV;
                entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz]);
                temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;

                [~,maxindex1]=max(entireRHS_ii_e,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                        entireRHS_ii_e=ezc1*temp2_ii+DiscountedEV(d3aprimez);
                        temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                        entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);
                        entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                        [~,maxindex]=max(entireRHS_ii_e,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,n_semiz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals, e_val, ReturnFnParamsVec,2,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                da1primez=d3ind+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*shiftdim(zBind,-2);
                entireRHS_ii_e=reshape(ezc1*reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,N_semiz])+reshape(DiscountedEVinterp(da1primez),[N_d13,n2long,N_a1,N_a2,N_semiz]),[N_d13*n2long,N_a1*N_a2,N_semiz]);
                temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);
                entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d13)+1;
                allind=d_ind+N_d13*aind+N_d13*N_a*zBind;
                Policy3_ford4_jj(1,:,:,e_c,d4_c)=d_ind;
                Policy3_ford4_jj(2,:,:,e_c,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford4_jj(3,:,:,e_c,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                % L2flag (on the transformed return matrix: -Inf marks both infeasible and zero-return points)
                L2offset      = ceil(maxindexL2/N_d13);
                linidx_lower  = d_ind                    + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
                ReturnMatrix_ii_resh=reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,N_semiz]);
                isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                % Get the d2Policy (read off the fine refine at the chosen fine a1prime point)
                d3part=rem(ceil(shiftdim(d_ind,1)/N_d1)-1,N_d3)+1;
                a1mid=squeeze(midpoint(allind));
                a1fine=(a1mid+(a1mid-1)*n2short)+shiftdim(L2offset,1)-n2short-2; % chosen fine a1prime index
                zidx=repmat(gpuArray(1:N_semiz),N_a,1);
                linlookup=d3part+N_d3*(a1fine-1)+N_d3*N_a1prime*(zidx-1);
                d2_ford4_jj(:,:,e_c,d4_c)=d2indexfine_resh(linlookup);
            end
        elseif vfoptions.lowmemory>=2 % lm2 already does the most-looped variant, so it also serves the higher lowmemory values
            % Loop over semiz (outer) and e (inner) to reduce memory footprint
            special_n_e=ones(1,length(n_e));
            for z_c=1:N_semiz
                semiz_val=semiz_gridvals(z_c,:);
                DiscountedEV_zc=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_zc=DiscountedEVinterp(:,:,:,:,z_c);
                d2indexfine_resh_zc=d2indexfine_resh(:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    midpoint=zeros(N_d13,1,N_a1,N_a2,1,'gpuArray');

                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,1,0);
                    RM=reshape(ReturnMatrix_ii_e,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,1]);
                    % Modify the Return Function appropriately for Epstein-Zin Preferences
                    becareful=logical(isfinite(RM).*(RM~=0)); % finite but not zero
                    temp2_ii=RM;
                    temp2_ii(becareful)=RM(becareful).^ezc2(jj);
                    temp2_ii(RM==0)=-Inf;
                    DEV=reshape(DiscountedEV_zc,[1,N_d3,N_a1,1,1,1]);
                    entireRHS_ii_e=ezc1*temp2_ii+DEV;
                    entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,1]);
                    temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                    entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
                    entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;

                    [~,maxindex1]=max(entireRHS_ii_e,[],2);
                    midpoint(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,3,0);
                            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                            temp2_ii=ReturnMatrix_ii;
                            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                            temp2_ii(ReturnMatrix_ii==0)=-Inf;
                            d3aprimez=d3ind+N_d3*(a1primeindexes-1);
                            entireRHS_ii_e=ezc1*temp2_ii+DiscountedEV_zc(d3aprimez);
                            temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                            entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);
                            entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                            [~,maxindex]=max(entireRHS_ii_e,[],2);
                            midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d3,special_n_d4],n2long,n_a1,n_a2,special_n_semiz,special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_val, e_val, ReturnFnParamsVec,2,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    da1primez=d3ind+N_d3*(a1primeindexesfine-1);
                    entireRHS_ii_e=reshape(ezc1*reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,1])+reshape(DiscountedEVinterp_zc(da1primez),[N_d13,n2long,N_a1,N_a2,1]),[N_d13*n2long,N_a1*N_a2,1]);
                    temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                    entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);
                    entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                    [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d13)+1;
                    allind=d_ind+N_d13*aind;
                    Policy3_ford4_jj(1,:,z_c,e_c,d4_c)=d_ind;
                    Policy3_ford4_jj(2,:,z_c,e_c,d4_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford4_jj(3,:,z_c,e_c,d4_c)=shiftdim(ceil(maxindexL2/N_d13),-1);

                    % L2flag (on the transformed return matrix: -Inf marks both infeasible and zero-return points)
                    L2offset      = ceil(maxindexL2/N_d13);
                    linidx_lower  = d_ind                    + N_d13*n2long*aind;
                    linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
                    ReturnMatrix_ii_resh=reshape(temp2_ii,[N_d13,n2long,N_a1,N_a2,1]);
                    isInfLower    = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                    % Get the d2Policy (read off the fine refine at the chosen fine a1prime point)
                    d3part=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                    a1mid=midpoint(allind);
                    a1fine=(a1mid+(a1mid-1)*n2short)+L2offset-n2short-2; % chosen fine a1prime index
                    linlookup=d3part+N_d3*(a1fine-1);
                    d2_ford4_jj(:,z_c,e_c,d4_c)=d2indexfine_resh_zc(linlookup);
                end
            end
        end
    end

    % Cross-d4 max
    [V_jj,d4winner]=max(V_ford4_jj,[],4);
    N=N_a*N_semiz*N_e;
    P1=reshape(Policy3_ford4_jj(1,:,:,:,:),[N,N_d4]);
    P2=reshape(Policy3_ford4_jj(2,:,:,:,:),[N,N_d4]);
    P3=reshape(Policy3_ford4_jj(3,:,:,:,:),[N,N_d4]);
    F =reshape(flag_ford4_jj,[N,N_d4]);
    D2=reshape(d2_ford4_jj,[N,N_d4]);
    rowidx=(1:1:N)';
    gather_idx=rowidx+N*(reshape(d4winner,[N,1])-1);
    Policy3_jj(1,:,:,:)=shiftdim(reshape(P1(gather_idx),[N_a,N_semiz,N_e]),-1);
    Policy3_jj(2,:,:,:)=shiftdim(reshape(P2(gather_idx),[N_a,N_semiz,N_e]),-1);
    Policy3_jj(3,:,:,:)=shiftdim(reshape(P3(gather_idx),[N_a,N_semiz,N_e]),-1);
    PolicyL2flag_jj(1,:,:,:)=shiftdim(reshape(F(gather_idx),[N_a,N_semiz,N_e]),-1);
    d2Policy_jj(1,:,:,:)=shiftdim(reshape(D2(gather_idx),[N_a,N_semiz,N_e]),-1);
    d4Policy_jj(1,:,:,:)=shiftdim(d4winner,-1);

    V(:,:,:,jj)=V_jj;
    Policy3(:,:,:,:,jj)=Policy3_jj;
    PolicyL2flag(:,:,:,:,jj)=PolicyL2flag_jj;
    d2Policy(:,:,:,:,jj)=d2Policy_jj;
    d4Policy(:,:,:,:,jj)=d4Policy_jj;
end


%% Switch Policy3(2,:) from 'midpoint' to 'lower grid index'
adjust=(Policy3(3,:,:,:,:)<1+n2short+1);
Policy3(2,:,:,:,:)=Policy3(2,:,:,:,:)-adjust;
Policy3(3,:,:,:,:)=adjust.*Policy3(3,:,:,:,:)+(1-adjust).*(Policy3(3,:,:,:,:)-n2short-1);

%% Encode Policy as component rows (with d1, no z, with e)
Policy=zeros(7,N_a,N_semiz,N_e,N_j,'gpuArray');
d13=Policy3(1,:,:,:,:);
Policy(1,:,:,:,:)=rem(d13-1,N_d1)+1;
Policy(2,:,:,:,:)=d2Policy;
Policy(3,:,:,:,:)=rem(ceil(d13/N_d1)-1,N_d3)+1;
Policy(4,:,:,:,:)=d4Policy;
Policy(5,:,:,:,:)=Policy3(2,:,:,:,:);
Policy(6,:,:,:,:)=Policy3(3,:,:,:,:);
Policy(7,:,:,:,:)=PolicyL2flag;

end
