function [V, Policy, Policyalt, Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_GI1_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid,e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,e), rather than standard (a,e,j)
% V is (a,j)-by-e (V carries Valt for Naive)

N_a=prod(n_a);
N_e=prod(n_e);

Vtilde=zeros(N_a*N_j,N_e,'gpuArray'); % QH-optimal value (max after L2 refinement on beta0beta-step)

e_gridvals_J=shiftdim(e_gridvals_J,-2);

%%
% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_aprime by N_a*N_j*N_e (note: N_aprime is just equal to N_a)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,{vfoptions.QHadditionaldiscount},N_j);
beta0beta_J=beta0_J.*beta_J; % Discount factor between today and tomorrow.

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

% pi_e_J is (a,j)-by-e
EV=[sum(V(N_a+1:end,:).*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_e_J to create expectations
EV=reshape(EV,[N_a,1,N_j]); % (aprime,1,j), 2nd dim will be autofilled with a

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV_alt=reshape(beta_J,[1,1,N_j]).*EV;
DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;
DiscountedEVinterp_alt=reshape(beta_J,[1,1,N_j]).*EVinterp;
DiscountedEVinterp=reshape(beta0beta_J,[1,1,N_j]).*EVinterp;

if vfoptions.lowmemory==0

    Policy=zeros(3,N_a,N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime (midpoint, aprimeL2ind, L2flag)
    Policyalt=zeros(3,N_a,N_j,N_e,'gpuArray');

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_e, N_j, a_grid, a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [aprime,a,j,e]

    %% Valt-step (beta) -- writes V and Policyalt
    entireRHS_alt=ReturnMatrix+DiscountedEV_alt; % [aprime,a,j,e]
    [~,maxindexalt]=max(entireRHS_alt,[],1);
    midpointalt=max(min(maxindexalt,n_a-1),2);
    aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_iialt=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,n_e,N_j,aprime_grid(aprimeindexesalt),a_grid,e_gridvals_J,ReturnFnParamsAgeMatrix,2);
    aprimejalt=aprimeindexesalt+n2aprime*jind;
    entireRHS_iialt=ReturnMatrix_iialt+reshape(DiscountedEVinterp_alt(aprimejalt(:)),[n2long,N_a,N_j,N_e]);
    [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
    V=reshape(Vtempii,[N_a*N_j,N_e]);
    Policyalt(1,:,:,:)=shiftdim(squeeze(midpointalt),-1);
    Policyalt(2,:,:,:)=shiftdim(maxindexL2alt,-1);
    isInfLoweralt    = (ReturnMatrix_iialt(1,     :,:,:) == -Inf);
    isInfUpperalt    = (ReturnMatrix_iialt(n2long,:,:,:) == -Inf);
    inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
    inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
    Policyalt(3,:,:,:) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

    %% beta0beta-step -- writes Policy
    entireRHS=ReturnMatrix+DiscountedEV; % [aprime,a,j,e]
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,n_e,N_j,aprime_grid(aprimeindexes),a_grid,e_gridvals_J,ReturnFnParamsAgeMatrix,2);
    aprimej=aprimeindexes+n2aprime*jind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimej(:)),[n2long,N_a,N_j,N_e]);
    [Vtildeii,maxindexL2]=max(entireRHS_ii,[],1);
    Vtilde=reshape(Vtildeii,[N_a*N_j,N_e]);
    Policy(1,:,:,:)=shiftdim(squeeze(midpoint),-1);
    Policy(2,:,:,:)=shiftdim(maxindexL2,-1);
    isInfLower    = (ReturnMatrix_ii(1,     :,:,:) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:,:,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,:,:) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_e,'gpuArray');
    Policy=zeros(3,N_a,N_j,N_e,'gpuArray');
    Policyalt=zeros(3,N_a,N_j,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,:,e_c,:);

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, special_n_e, N_j, a_grid, a_grid, e_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix is [aprime,a,j]

        %% Valt-step (beta)
        entireRHS_alt_e=ReturnMatrix_e+DiscountedEV_alt; % [aprime,a,j]
        [~,maxindexalt]=max(entireRHS_alt_e,[],1);
        midpointalt=max(min(maxindexalt,n_a-1),2);
        aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_iialt=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,special_n_e,N_j,aprime_grid(aprimeindexesalt),a_grid,e_vals,ReturnFnParamsAgeMatrix,2);
        aprimejalt=aprimeindexesalt+n2aprime*jind;
        entireRHS_iialt=ReturnMatrix_iialt+reshape(DiscountedEVinterp_alt(aprimejalt(:)),[n2long,N_a,N_j]);
        [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
        V(:,e_c)=reshape(Vtempii,[N_a*N_j,1]);
        Policyalt(1,:,:,e_c)=shiftdim(squeeze(midpointalt),-1);
        Policyalt(2,:,:,e_c)=shiftdim(maxindexL2alt,-1);
        isInfLoweralt    = (ReturnMatrix_iialt(1,     :,:) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(n2long,:,:) == -Inf);
        inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
        inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
        Policyalt(3,:,:,e_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

        %% beta0beta-step
        entireRHS_e=ReturnMatrix_e+DiscountedEV; % [aprime,a,j]
        [~,maxindex]=max(entireRHS_e,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,special_n_e,N_j,aprime_grid(aprimeindexes),a_grid,e_vals,ReturnFnParamsAgeMatrix,2);
        aprimej=aprimeindexes+n2aprime*jind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimej(:)),[n2long,N_a,N_j]);
        [Vtildeii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,e_c)=reshape(Vtildeii,[N_a*N_j,1]);
        Policy(1,:,:,e_c)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,e_c)=shiftdim(maxindexL2,-1);
        isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        Policy(3,:,:,e_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
    end

end

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:)<1+n2short+1);
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust;
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1);

adjustalt=(Policyalt(2,:,:,:)<1+n2short+1);
Policyalt(1,:,:,:)=Policyalt(1,:,:,:)-adjustalt;
Policyalt(2,:,:,:)=adjustalt.*Policyalt(2,:,:,:)+(1-adjustalt).*(Policyalt(2,:,:,:)-n2short-1);


end
