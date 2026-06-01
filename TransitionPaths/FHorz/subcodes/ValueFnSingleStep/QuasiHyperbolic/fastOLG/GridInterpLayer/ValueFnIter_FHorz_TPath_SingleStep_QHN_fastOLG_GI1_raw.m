function [V,Policy,Policyalt,Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_GI1_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z (V carries Valt for Naive)
% Policy is (a,j,z)
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% V=zeros(N_a*N_j,N_z,'gpuArray');
Vtilde=zeros(N_a*N_j,N_z,'gpuArray'); % QH-optimal value (max after L2 refinement on beta0beta-step)
Policy=zeros(4,N_a,N_j,N_z,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z (channels: d, midpoint, aprimeL2ind, L2flag)
Policyalt=zeros(4,N_a,N_j,N_z,'gpuArray'); % exponential discounter optimal choice (Valt is computed at this)

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

%% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);
aBind=gpuArray(0:1:N_a-1);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
zBind=shiftdim(gpuArray(0:1:N_z-1),-2);


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
DiscountedEV_alt=repelem(shiftdim(DiscountedEV_alt,-1),N_d,1,1,1); % [d,aprime,1,j,z]
DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;
DiscountedEV=repelem(shiftdim(DiscountedEV,-1),N_d,1,1,1); % [d,aprime,1,j,z]

DiscountedEVinterp_alt=reshape(beta_J,[1,1,N_j]).*EVinterp;
DiscountedEVinterp_alt=repelem(shiftdim(DiscountedEVinterp_alt,-1),N_d,1,1,1); % [d,aprime,1,j,z]
DiscountedEVinterp=reshape(beta0beta_J,[1,1,N_j]).*EVinterp;
DiscountedEVinterp=repelem(shiftdim(DiscountedEVinterp,-1),N_d,1,1,1); % [d,aprime,1,j,z]

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_z, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [d,aprime,a,j,z]

    %% Valt-step (beta) -- writes V and Policyalt (exponential-discounter choice)
    entireRHS_alt=ReturnMatrix+DiscountedEV_alt; %  [d,aprime,a,j,z]
    [~,maxindex1alt]=max(entireRHS_alt,[],2);
    midpointalt=max(min(maxindex1alt,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    ReturnMatrix_iialt=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn,n_d,n_z,N_j,d_gridvals,aprime_grid(aprimeindexesalt),a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);
    daprimejalt=(1:1:N_d)'+N_d*(aprimeindexesalt-1)+N_d*n2aprime*jind+N_d*n2aprime*N_j*zind;
    entireRHS_iialt=ReturnMatrix_iialt+reshape(DiscountedEVinterp_alt(daprimejalt(:)),[N_d*n2long,N_a,N_j,N_z]);
    [V,maxindexL2alt]=max(entireRHS_iialt,[],1);
    V=reshape(V,[N_a*N_j,N_z]);
    d_indalt=rem(maxindexL2alt-1,N_d)+1;
    allindalt=d_indalt+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind;
    Policyalt(1,:,:,:)=d_indalt; % d
    Policyalt(2,:,:,:)=shiftdim(squeeze(midpointalt(allindalt)),-1); % midpoint
    Policyalt(3,:,:,:)=shiftdim(ceil(maxindexL2alt/N_d),-1); % aprimeL2ind
    L2offsetalt=ceil(maxindexL2alt/N_d);
    linidx_loweralt=d_indalt                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
    linidx_upperalt=d_indalt+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
    isInfLoweralt=(ReturnMatrix_iialt(linidx_loweralt)==-Inf);
    isInfUpperalt=(ReturnMatrix_iialt(linidx_upperalt)==-Inf);
    inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
    inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
    Policyalt(4,:,:,:)=shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

    %% beta0beta-step -- writes Policy (QH-optimal choice)
    entireRHS=ReturnMatrix+DiscountedEV; %  [d,aprime,a,j,z]
    [~,maxindex1]=max(entireRHS,[],2);
    midpoint=max(min(maxindex1,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn,n_d,n_z,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);
    daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind+N_d*n2aprime*N_j*zind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j,N_z]);
    [Vtilde,maxindexL2]=max(entireRHS_ii,[],1);
    Vtilde=reshape(Vtilde,[N_a*N_j,N_z]);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind;
    Policy(1,:,:,:)=d_ind; % d
    Policy(2,:,:,:)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
    Policy(3,:,:,:)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
    L2offset=ceil(maxindexL2/N_d);
    linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
    linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
    inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
    Policy(4,:,:,:)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

elseif vfoptions.lowmemory==1

    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
        DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,z_c);
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);

        ReturnMatrix_z=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a_grid, a_grid, z_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix is [d,aprime,a,j]

        %% Valt-step (beta)
        entireRHS_alt_z=ReturnMatrix_z+DiscountedEV_alt_z; %(d,aprime)-by-(a,j)
        [~,maxindex1alt]=max(entireRHS_alt_z,[],2);
        midpointalt=max(min(maxindex1alt,n_a-1),2);
        aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_iialt=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn,n_d,special_n_z,N_j,d_gridvals,aprime_grid(aprimeindexesalt),a_grid,z_vals,ReturnFnParamsAgeMatrix,2);
        daprimejalt=(1:1:N_d)'+N_d*(aprimeindexesalt-1)+N_d*n2aprime*jind;
        entireRHS_iialt=ReturnMatrix_iialt+reshape(DiscountedEVinterp_alt_z(daprimejalt(:)),[N_d*n2long,N_a,N_j]);
        [Vtemp,maxindexL2alt]=max(entireRHS_iialt,[],1);
        V(:,z_c)=reshape(Vtemp,[N_a*N_j,1]);
        d_indalt=rem(maxindexL2alt-1,N_d)+1;
        allindalt=d_indalt+N_d*aBind+N_d*N_a*jBind;
        Policyalt(1,:,:,z_c)=d_indalt; % d
        Policyalt(2,:,:,z_c)=shiftdim(squeeze(midpointalt(allindalt)),-1); % midpoint
        Policyalt(3,:,:,z_c)=shiftdim(ceil(maxindexL2alt/N_d),-1); % aprimeL2ind
        L2offsetalt=ceil(maxindexL2alt/N_d);
        linidx_loweralt=d_indalt                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
        linidx_upperalt=d_indalt+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
        isInfLoweralt=(ReturnMatrix_iialt(linidx_loweralt)==-Inf);
        isInfUpperalt=(ReturnMatrix_iialt(linidx_upperalt)==-Inf);
        inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
        inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
        Policyalt(4,:,:,z_c)=shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt),-1);

        %% beta0beta-step
        entireRHS_z=ReturnMatrix_z+DiscountedEV_z; %(d,aprime)-by-(a,j)
        [~,maxindex1]=max(entireRHS_z,[],2);
        midpoint=max(min(maxindex1,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn,n_d,special_n_z,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_vals,ReturnFnParamsAgeMatrix,2);
        daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(daprimej(:)),[N_d*n2long,N_a,N_j]);
        [Vtildetemp,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,z_c)=reshape(Vtildetemp,[N_a*N_j,1]);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aBind+N_d*N_a*jBind;
        Policy(1,:,:,z_c)=d_ind; % d
        Policy(2,:,:,z_c)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,z_c)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        L2offset=ceil(maxindexL2/N_d);
        linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
        linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
        inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
        Policy(4,:,:,z_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
    end
end



%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

adjustalt=(Policyalt(3,:,:,:)<1+n2short+1);
Policyalt(2,:,:,:)=Policyalt(2,:,:,:)-adjustalt;
Policyalt(3,:,:,:)=adjustalt.*Policyalt(3,:,:,:)+(1-adjustalt).*(Policyalt(3,:,:,:)-n2short-1);

%% fastOLG with z, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z]);
% Policy=reshape(Policy,[size(Policy,1),N_a,N_j,N_z]);


end
