function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_GI1_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z-by-e (V carries Vunderbar for Sophisticated)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Vhat=zeros(N_a*N_j,N_z,N_e,'gpuArray'); % pre-Vunderbar value (snapshot of V before the beta*EV-at-policy correction)
Policy=zeros(4,N_a,N_j,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime (d, midpoint, aprimeL2ind, L2flag)

z_gridvals_J=shiftdim(z_gridvals_J,-3);
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]);


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
eBind=shiftdim(gpuArray(0:1:N_e-1),-3);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

beta_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
beta0_J=CreateAgeMatrixFromParams(Parameters,{vfoptions.QHadditionaldiscount},N_j);
beta0beta_J=beta0_J.*beta_J; % Discount factor between today and tomorrow.

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV=reshape(beta0beta_J,[1,1,N_j]).*EV;
DiscountedEV=repelem(shiftdim(DiscountedEV,-1),N_d,1,1,1); % [d,aprime,1,j,z]

DiscountedEVinterp=reshape(beta0beta_J,[1,1,N_j]).*EVinterp;
DiscountedEVinterp=repelem(shiftdim(DiscountedEVinterp,-1),N_d,1,1,1); % [d,aprime,1,j,z]
EVinterp_d=repelem(shiftdim(EVinterp,-1),N_d,1,1,1); % [d,aprime,1,j,z], undiscounted

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [d,aprime,a,j,z,e]

    %% beta0beta-step -- writes Policy (QH-optimal choice) and Vhat
    entireRHS=ReturnMatrix+DiscountedEV; %  [d,aprime,a,j,z,e]
    [~,maxindex1]=max(entireRHS,[],2);
    midpoint=max(min(maxindex1,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d,n_z,n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind+N_d*n2aprime*N_j*zind;
    EVfine=reshape(EVinterp_d(daprimej(:)),[N_d*n2long,N_a,N_j,N_z,N_e]);
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j,N_z,N_e]);
    [Vhatii,maxindexL2]=max(entireRHS_ii,[],1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind+N_d*N_a*N_j*N_z*eBind;
    Policy(1,:,:,:,:)=d_ind;
    Policy(2,:,:,:,:)=shiftdim(squeeze(midpoint(allind)),-1);
    Policy(3,:,:,:,:)=shiftdim(ceil(maxindexL2/N_d),-1);
    L2offset=ceil(maxindexL2/N_d);
    linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind+N_d*n2long*N_a*N_j*N_z*eBind;
    linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind+N_d*n2long*N_a*N_j*N_z*eBind;
    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
    inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
    Policy(4,:,:,:,:)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

    %% Vunderbar: re-evaluate at Policy's aprime with beta
    linidx=double(reshape(maxindexL2,[1,N_a*N_j*N_z*N_e]))+N_d*n2long*(0:N_a*N_j*N_z*N_e-1);
    EV_at_policy=reshape(EVfine(linidx),[N_a,N_j,N_z,N_e]);
    V=shiftdim(Vhatii,1)+reshape(beta_J-beta0beta_J,[1,N_j,1,1]).*EV_at_policy;
    V=reshape(V,[N_a*N_j,N_z,N_e]);
    Vhat=reshape(Vhatii,[N_a*N_j,N_z,N_e]); % snapshot pre-Vunderbar

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:);

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J,e_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix is [d,aprime,a,j,z]

        %% beta0beta-step
        entireRHS_e=ReturnMatrix_e+DiscountedEV;
        [~,maxindex1]=max(entireRHS_e,[],2);
        midpoint=max(min(maxindex1,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d,n_z,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J,e_vals,ReturnFnParamsAgeMatrix,2);
        daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind+N_d*n2aprime*N_j*zind;
        EVfine_e=reshape(EVinterp_d(daprimej(:)),[N_d*n2long,N_a,N_j,N_z]);
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j,N_z]);
        [Vhatii,maxindexL2]=max(entireRHS_ii,[],1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind;
        Policy(1,:,:,:,e_c)=d_ind;
        Policy(2,:,:,:,e_c)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy(3,:,:,:,e_c)=shiftdim(ceil(maxindexL2/N_d),-1);
        L2offset=ceil(maxindexL2/N_d);
        linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
        linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
        inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
        Policy(4,:,:,:,e_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

        %% Vunderbar
        linidx_e=double(reshape(maxindexL2,[1,N_a*N_j*N_z]))+N_d*n2long*(0:N_a*N_j*N_z-1);
        EV_at_policy_e=reshape(EVfine_e(linidx_e),[N_a,N_j,N_z]);
        Vtemp=shiftdim(Vhatii,1)+reshape(beta_J-beta0beta_J,[1,N_j,1]).*EV_at_policy_e;
        V(:,:,e_c)=reshape(Vtemp,[N_a*N_j,N_z]);
        Vhat(:,:,e_c)=reshape(Vhatii,[N_a*N_j,N_z]); % snapshot pre-Vunderbar
    end

elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:);
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);
        EVinterp_d_z=EVinterp_d(:,:,:,:,z_c);

        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,1,:,1,e_c,:);

            ReturnMatrix_ze=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, special_n_z,special_n_e, N_j, d_gridvals, a_grid, a_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix,1);
            % fastOLG: ReturnMatrix is [d,aprime,a,j]

            %% beta0beta-step
            entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z;
            [~,maxindex1]=max(entireRHS_ze,[],2);
            midpoint=max(min(maxindex1,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d,special_n_z,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_vals,e_vals,ReturnFnParamsAgeMatrix,2);
            daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind;
            EVfine_ze=reshape(EVinterp_d_z(daprimej(:)),[N_d*n2long,N_a,N_j]);
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(daprimej(:)),[N_d*n2long,N_a,N_j]);
            [Vhatii,maxindexL2]=max(entireRHS_ii,[],1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aBind+N_d*N_a*jBind;
            Policy(1,:,:,z_c,e_c)=d_ind;
            Policy(2,:,:,z_c,e_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy(3,:,:,z_c,e_c)=shiftdim(ceil(maxindexL2/N_d),-1);
            L2offset=ceil(maxindexL2/N_d);
            linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
            linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind;
            isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
            Policy(4,:,:,z_c,e_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

            %% Vunderbar
            linidx_ze=double(reshape(maxindexL2,[1,N_a*N_j]))+N_d*n2long*(0:N_a*N_j-1);
            EV_at_policy_ze=reshape(EVfine_ze(linidx_ze),[N_a,N_j]);
            Vtemp=shiftdim(Vhatii,1)+reshape(beta_J-beta0beta_J,[1,N_j]).*EV_at_policy_ze;
            V(:,z_c,e_c)=reshape(Vtemp,[N_a*N_j,1]);
            Vhat(:,z_c,e_c)=reshape(Vhatii,[N_a*N_j,1]); % snapshot pre-Vunderbar
        end
    end
end


%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1);
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust;
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1);


end
