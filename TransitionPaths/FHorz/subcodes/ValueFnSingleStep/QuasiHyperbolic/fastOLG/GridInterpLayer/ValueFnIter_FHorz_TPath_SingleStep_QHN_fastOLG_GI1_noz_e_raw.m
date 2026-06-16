function [V,Policy,Policyalt,Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_GI1_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,e), rather than standard (a,e,j)
% V is (a,j)-by-e (V carries Valt for Naive)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

Vtilde=zeros(N_a*N_j,N_e,'gpuArray'); % QH-optimal value (max after L2 refinement on beta0beta-step)
Policy=zeros(4,N_a,N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime (d, midpoint, aprimeL2ind, L2flag)
Policyalt=zeros(4,N_a,N_j,N_e,'gpuArray');

e_gridvals_J=shiftdim(e_gridvals_J,-3); % [1,1,1,N_j,N_e,l_e]

%% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
aBind=gpuArray(0:1:N_a-1);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_e (note: N_aprime is just equal to N_a)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,{vfoptions.QHadditionaldiscount},N_j);
beta0beta_J=beta0_J.*beta_J; % Discount factor between today and tomorrow.

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

% pi_e_J is (a,j)-by-e
EV=[sum(V(N_a+1:end,:).*pi_e_J(1:end-N_a,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_e_J to create expectations
EV=reshape(EV,[N_a,1,N_j]); % (aprime,1,j), 2nd dim will be autofilled with a

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV_alt=reshape(beta_J,[1,1,N_j]).*EV;
DiscountedEV_alt=repelem(shiftdim(DiscountedEV_alt,-1),N_d,1,1); % [d,aprime,1,j]
DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;
DiscountedEV=repelem(shiftdim(DiscountedEV,-1),N_d,1,1); % [d,aprime,1,j]

DiscountedEVinterp_alt=reshape(beta_J,[1,1,N_j]).*EVinterp;
DiscountedEVinterp_alt=repelem(shiftdim(DiscountedEVinterp_alt,-1),N_d,1,1); % [d,aprime,1,j]
DiscountedEVinterp=reshape(beta0beta_J,[1,1,N_j]).*EVinterp;
DiscountedEVinterp=repelem(shiftdim(DiscountedEVinterp,-1),N_d,1,1); % [d,aprime,1,j]


if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_e, N_j, d_gridvals, a_grid, a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [d,aprime,a,j,e]

    %% Valt-step (beta) -- writes V and Policyalt
    entireRHS_alt=ReturnMatrix+DiscountedEV_alt; %  [d,aprime,a,j,e]
    [~,maxindex1alt]=max(entireRHS_alt,[],2);
    midpointalt=max(min(maxindex1alt,n_a-1),2);
    aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_iialt=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn,n_d,n_e,N_j,d_gridvals,aprime_grid(aprimeindexesalt),a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    daprimejalt=(1:1:N_d)'+N_d*(aprimeindexesalt-1)+N_d*n2aprime*jind;
    entireRHS_iialt=ReturnMatrix_iialt+reshape(DiscountedEVinterp_alt(daprimejalt(:)),[N_d*n2long,N_a,N_j,N_e]);
    [V,maxindexL2alt]=max(entireRHS_iialt,[],1);
    V=reshape(V,[N_a*N_j,N_e]);
    d_indalt=rem(maxindexL2alt-1,N_d)+1;
    allindalt=d_indalt+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*eBind;
    Policyalt(1,:,:,:)=d_indalt;
    Policyalt(2,:,:,:)=shiftdim(squeeze(midpointalt(allindalt)),-1);
    Policyalt(3,:,:,:)=shiftdim(ceil(maxindexL2alt/N_d),-1);
    L2offsetalt=ceil(maxindexL2alt/N_d);
    linidx_loweralt=d_indalt                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*eBind;
    linidx_upperalt=d_indalt+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*eBind;
    isInfLoweralt=(ReturnMatrix_iialt(linidx_loweralt)==-Inf);
    isInfUpperalt=(ReturnMatrix_iialt(linidx_upperalt)==-Inf);
    inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
    inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
    Policyalt(4,:,:,:)=shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

    %% beta0beta-step -- writes Policy
    entireRHS=ReturnMatrix+DiscountedEV; %  [d,aprime,a,j,e]
    [~,maxindex1]=max(entireRHS,[],2);
    midpoint=max(min(maxindex1,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn,n_d,n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j,N_e]);
    [Vtilde,maxindexL2]=max(entireRHS_ii,[],1);
    Vtilde=reshape(Vtilde,[N_a*N_j,N_e]);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*eBind;
    Policy(1,:,:,:)=d_ind;
    Policy(2,:,:,:)=shiftdim(squeeze(midpoint(allind)),-1);
    Policy(3,:,:,:)=shiftdim(ceil(maxindexL2/N_d),-1);
    L2offset=ceil(maxindexL2/N_d);
    linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*eBind;
    linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*eBind;
    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
    inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
    Policy(4,:,:,:)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,e_c,:);

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_e, N_j, d_gridvals, a_grid, a_grid, e_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix is [d,aprime,a,j]

        %% Valt-step (beta)
        entireRHS_alt_e=ReturnMatrix_e+DiscountedEV_alt; %(d,aprime)-by-(a,j)
        [~,maxindex1alt]=max(entireRHS_alt_e,[],2);
        midpointalt=max(min(maxindex1alt,n_a-1),2);
        aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_iialt=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn,n_d,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexesalt),a_grid,e_vals,ReturnFnParamsAgeMatrix,2);
        daprimejalt=(1:1:N_d)'+N_d*(aprimeindexesalt-1)+N_d*n2aprime*jind;
        entireRHS_iialt=ReturnMatrix_iialt+reshape(DiscountedEVinterp_alt(daprimejalt(:)),[N_d*n2long,N_a,N_j]);
        [Vtemp,maxindexL2alt]=max(entireRHS_iialt,[],1);
        V(:,e_c)=reshape(Vtemp,[N_a*N_j,1]);
        d_indalt=rem(maxindexL2alt-1,N_d)+1;
        allindalt=d_indalt+N_d*aBind+N_d*N_a*jBind;
        Policyalt(1,:,:,e_c)=d_indalt;
        Policyalt(2,:,:,e_c)=shiftdim(squeeze(midpointalt(allindalt)),-1);
        Policyalt(3,:,:,e_c)=shiftdim(ceil(maxindexL2alt/N_d),-1);
        L2offsetalt=ceil(maxindexL2alt/N_d);
        linidx_loweralt=d_indalt                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
        linidx_upperalt=d_indalt+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
        isInfLoweralt=(ReturnMatrix_iialt(linidx_loweralt)==-Inf);
        isInfUpperalt=(ReturnMatrix_iialt(linidx_upperalt)==-Inf);
        inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
        inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
        Policyalt(4,:,:,e_c)=shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

        %% beta0beta-step
        entireRHS_e=ReturnMatrix_e+DiscountedEV;
        [~,maxindex1]=max(entireRHS_e,[],2);
        midpoint=max(min(maxindex1,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn,n_d,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_vals,ReturnFnParamsAgeMatrix,2);
        daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j]);
        [Vtildetemp,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,e_c)=reshape(Vtildetemp,[N_a*N_j,1]);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aBind+N_d*N_a*jBind;
        Policy(1,:,:,e_c)=d_ind;
        Policy(2,:,:,e_c)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy(3,:,:,e_c)=shiftdim(ceil(maxindexL2/N_d),-1);
        L2offset=ceil(maxindexL2/N_d);
        linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
        linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
        inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
        Policy(4,:,:,e_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
    end
end


%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1);

adjustalt=(Policyalt(3,:,:,:)<1+n2short+1);
Policyalt(2,:,:,:)=Policyalt(2,:,:,:)-adjustalt;
Policyalt(3,:,:,:)=adjustalt.*Policyalt(3,:,:,:)+(1-adjustalt).*(Policyalt(3,:,:,:)-n2short-1);


end
