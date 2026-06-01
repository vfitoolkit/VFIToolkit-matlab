function [V,Policy,Policyalt,Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_GI1_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z (V carries Valt for Naive)
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG

N_a=prod(n_a);
N_z=prod(n_z);

Vtilde=zeros(N_a*N_j,N_z,'gpuArray'); % QH-optimal value (max after L2 refinement on beta0beta-step)
% Policy=zeros(N_a*N_j,N_z,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

z_gridvals_J=shiftdim(z_gridvals_J,-2); % [1,1,N_j,N_z,l_z]

%%
% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-2);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,{vfoptions.QHadditionaldiscount},N_j);
beta0beta_J=beta0_J.*beta_J; % Discount factor between today and tomorrow.

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    EVpre=zeros(N_a,1,N_j,N_z);
    EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % I use zeros in j=N_j so that can just use pi_z_J to create expectations
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EV=reshape(V,[N_a,1,N_j,N_z]).*shiftdim(pi_z_J,-2); % input V is already of size [N_a,N_j,N_z] and we want to use the whole thing
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
end

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV_alt=reshape(beta_J,[1,1,N_j]).*EV;
DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;
DiscountedEVinterp_alt=reshape(beta_J,[1,1,N_j]).*EVinterp;
DiscountedEVinterp=reshape(beta0beta_J,[1,1,N_j]).*EVinterp;


if vfoptions.lowmemory==0

    Policy=zeros(3,N_a,N_j,N_z,'gpuArray'); %first dim indexes the optimal choice for aprime (midpoint, aprimeL2ind, L2flag)
    Policyalt=zeros(3,N_a,N_j,N_z,'gpuArray');

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_z, N_j, a_grid, a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [aprime,a,j,z]

    %% Valt-step (beta) -- writes V and Policyalt
    entireRHS_alt=ReturnMatrix+DiscountedEV_alt; % [aprime,a,j,z]
    [~,maxindexalt]=max(entireRHS_alt,[],1);
    midpointalt=max(min(maxindexalt,n_a-1),2);
    aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_iialt=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,n_z,N_j,aprime_grid(aprimeindexesalt),a_grid,z_gridvals_J,ReturnFnParamsAgeMatrix,2);
    aprimejzalt=aprimeindexesalt+n2aprime*jind+n2aprime*N_j*zind;
    entireRHS_iialt=ReturnMatrix_iialt+reshape(DiscountedEVinterp_alt(aprimejzalt(:)),[n2long,N_a,N_j,N_z]);
    [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
    V=reshape(Vtempii,[N_a*N_j,N_z]);
    Policyalt(1,:,:,:)=shiftdim(squeeze(midpointalt),-1); % midpoint
    Policyalt(2,:,:,:)=shiftdim(maxindexL2alt,-1); % aprimeL2ind
    isInfLoweralt    = (ReturnMatrix_iialt(1,     :,:,:) == -Inf);
    isInfUpperalt    = (ReturnMatrix_iialt(n2long,:,:,:) == -Inf);
    inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
    inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
    Policyalt(3,:,:,:) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

    %% beta0beta-step -- writes Policy
    entireRHS=ReturnMatrix+DiscountedEV; % [aprime,a,j,z]
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,n_z,N_j,aprime_grid(aprimeindexes),a_grid,z_gridvals_J,ReturnFnParamsAgeMatrix,2);
    aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimejz(:)),[n2long,N_a,N_j,N_z]);
    [Vtildeii,maxindexL2]=max(entireRHS_ii,[],1);
    Vtilde=reshape(Vtildeii,[N_a*N_j,N_z]);
    Policy(1,:,:,:)=shiftdim(squeeze(midpoint),-1); % midpoint
    Policy(2,:,:,:)=shiftdim(maxindexL2,-1); % aprimeL2ind
    isInfLower    = (ReturnMatrix_ii(1,     :,:,:) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:,:,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,:,:) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

elseif vfoptions.lowmemory==1

    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,'gpuArray');
    Policy=zeros(3,N_a,N_j,N_z,'gpuArray'); %first dim indexes the optimal choice for aprime (midpoint, aprimeL2ind, L2flag)
    Policyalt=zeros(3,N_a,N_j,N_z,'gpuArray');

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
        DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,z_c);
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);
        DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,z_c);

        ReturnMatrix_z=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, special_n_z, N_j, a_grid, a_grid, z_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix_z is [aprime,a,j]

        %% Valt-step (beta)
        entireRHS_alt_z=ReturnMatrix_z+DiscountedEV_alt_z; % [aprime,a,j]
        [~,maxindexalt]=max(entireRHS_alt_z,[],1);
        midpointalt=max(min(maxindexalt,n_a-1),2);
        aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_iialt=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, special_n_z, N_j,aprime_grid(aprimeindexesalt),a_grid, z_vals,ReturnFnParamsAgeMatrix,2);
        aprimejalt=aprimeindexesalt+n2aprime*jind;
        entireRHS_iialt=ReturnMatrix_iialt+reshape(DiscountedEVinterp_alt_z(aprimejalt(:)),[n2long,N_a,N_j]);
        [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
        V(:,z_c)=reshape(Vtempii,[N_a*N_j,1]);
        Policyalt(1,:,:,z_c)=shiftdim(squeeze(midpointalt),-1); % midpoint
        Policyalt(2,:,:,z_c)=shiftdim(maxindexL2alt,-1); % aprimeL2ind
        isInfLoweralt    = (ReturnMatrix_iialt(1,     :,:) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(n2long,:,:) == -Inf);
        inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
        inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
        Policyalt(3,:,:,z_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

        %% beta0beta-step
        entireRHS_z=ReturnMatrix_z+DiscountedEV_z; % [aprime,a,j]
        [~,maxindex]=max(entireRHS_z,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, special_n_z, N_j,aprime_grid(aprimeindexes),a_grid, z_vals,ReturnFnParamsAgeMatrix,2);
        aprimej=aprimeindexes+n2aprime*jind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(aprimej(:)),[n2long,N_a,N_j]);
        [Vtildeii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,z_c)=reshape(Vtildeii,[N_a*N_j,1]);
        Policy(1,:,:,z_c)=shiftdim(squeeze(midpoint),-1); % midpoint
        Policy(2,:,:,z_c)=shiftdim(maxindexL2,-1); % aprimeL2ind
        isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        Policy(3,:,:,z_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
    end

end

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust; % lower grid point
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

adjustalt=(Policyalt(2,:,:,:)<1+n2short+1);
Policyalt(1,:,:,:)=Policyalt(1,:,:,:)-adjustalt;
Policyalt(2,:,:,:)=adjustalt.*Policyalt(2,:,:,:)+(1-adjustalt).*(Policyalt(2,:,:,:)-n2short-1);

%% fastOLG with z, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z]);
% Policy=reshape(Policy,[size(Policy,1),N_a,N_j,N_z]);


end
