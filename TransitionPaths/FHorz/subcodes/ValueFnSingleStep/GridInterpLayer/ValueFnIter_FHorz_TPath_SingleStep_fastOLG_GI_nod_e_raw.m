function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI_nod_e_raw(V,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J,e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z,e), rather than standard (a,z,e,j)
% V is (a,j)-by-z-by-e

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

z_gridvals_J=shiftdim(z_gridvals_J,-2); % [1,1,N_j,N_z,l_z]
e_gridvals_J=reshape(e_gridvals_J,[1,1,N_j,1,N_e,length(n_e)]); % [1,1,N_j,1,N_e,l_e]

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

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV=DiscountFactorParamsVec.*EV;
DiscountedEVinterp=DiscountFactorParamsVec.*EVinterp;

if vfoptions.lowmemory==0

    Policy=zeros(2,N_a,N_j,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z, n_e, N_j, a_grid, a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [aprime,a,j,z]
    
    entireRHS=ReturnMatrix+DiscountedEV; % [aprime,a,j,z]

    % Calc the max and it's index
    [~,maxindex]=max(entireRHS,[],1);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a-by-N_j-by-n_z-by-n_e
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a-by-N_j-by-n_z-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn,n_z,n_e,N_j,aprime_grid(aprimeindexes),a_grid,z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimejz(:)),[n2long,N_a,N_j,N_z,N_e]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V=reshape(Vtempii,[N_a*N_j,N_z,N_e]);
    Policy(1,:,:,:,:)=shiftdim(squeeze(midpoint),-1); % midpoint
    Policy(2,:,:,:,:)=shiftdim(maxindexL2,-1); % aprimeL2ind
    
elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    Policy=zeros(2,N_a,N_j,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime
    
    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,:,1,e_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG

        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z,special_n_e, N_j, a_grid, a_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix_z is [aprime,a,j,z]

        entireRHS_e=ReturnMatrix_e+DiscountedEV; % [aprime,a,j,z]

        % Calc the max and it's index
        [~,maxindex]=max(entireRHS_e,[],1);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a-by-N_j
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a-by-N_j
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z,special_n_e, N_j,aprime_grid(aprimeindexes), a_grid, z_gridvals_J, e_vals,ReturnFnParamsAgeMatrix,2);
        aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimejz(:)),[n2long,N_a,N_j,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,e_c)=reshape(Vtempii,[N_a*N_j,N_z]);
        Policy(1,:,:,:,e_c)=shiftdim(squeeze(midpoint),-1); % midpoint
        Policy(2,:,:,:,e_c)=shiftdim(maxindexL2,-1); % aprimeL2ind
    end

elseif vfoptions.lowmemory==2

    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    Policy=zeros(2,N_a,N_j,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime
    
    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,z_c);
        
        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,:,1,e_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG

            ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, special_n_z,special_n_e, N_j, a_grid, a_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix,1);
            % fastOLG: ReturnMatrix_z is [aprime,a,j]

            entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z; % [aprime,a,j]

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_ze,[],1);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-N_j
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a-by-N_j
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, special_n_z,special_n_e, N_j,aprime_grid(aprimeindexes),a_grid, z_vals,e_vals, ReturnFnParamsAgeMatrix,2);
            aprimej=aprimeindexes+n2aprime*jind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(aprimej(:)),[n2long,N_a,N_j]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,z_c,e_c)=reshape(Vtempii,[N_a*N_j,1]);
            Policy(1,:,:,z_c,e_c)=shiftdim(squeeze(midpoint),-1); % midpoint
            Policy(2,:,:,z_c,e_c)=shiftdim(maxindexL2,-1); % aprimeL2ind
        end
    end
end

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:,:)=Policy(1,:,:,:,:)-adjust; % lower grid point
Policy(2,:,:,:,:)=adjust.*Policy(2,:,:,:,:)+(1-adjust).*(Policy(2,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% Leave the first dimension as is
% Policy=squeeze(Policy(1,:,:,:,:)+N_a*(Policy(2,:,:,:,:)-1));


%% fastOLG with z & e, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_z,N_e]);
% Note that in fastOLG, we do not separate d from aprime in Policy


end
