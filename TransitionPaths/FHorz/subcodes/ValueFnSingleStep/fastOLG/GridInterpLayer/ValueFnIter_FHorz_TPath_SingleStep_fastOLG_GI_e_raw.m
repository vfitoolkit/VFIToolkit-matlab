function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z-by-e

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Policy=zeros(3,N_a,N_j,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

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
DiscountedEV=repelem(shiftdim(DiscountedEV,-1),N_d,1,1,1); % [d,aprime,1,j,z]

DiscountedEVinterp=DiscountFactorParamsVec.*EVinterp;
DiscountedEVinterp=repelem(shiftdim(DiscountedEVinterp,-1),N_d,1,1,1); % [d,aprime,1,j,z]

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2e(ReturnFn, n_d, n_z, n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [d,aprime,a,j,z,e]

    entireRHS=ReturnMatrix+DiscountedEV; %  [d,aprime,a,j,z,e]

    % First, we want aprime conditional on (d,1,a,j)
    [~,maxindex1]=max(entireRHS,[],2);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex1,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_z-by-n_e
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a-by-N_j-by-n_z-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2e(ReturnFn,n_d,n_z,n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind+N_d*n2aprime*N_j*zind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j,N_z,N_e]);
    [V,maxindexL2]=max(entireRHS_ii,[],1);
    V=reshape(V,[N_a*N_j,N_z,N_e]);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind+N_d*N_a*N_j*N_z*eBind; % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_z-by-n_e
    Policy(1,:,:,:,:)=d_ind; % d
    Policy(2,:,:,:,:)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
    Policy(3,:,:,:,:)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG

        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2e(ReturnFn, n_d, n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J,e_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix is [d,aprime,a,j]

        entireRHS_e=ReturnMatrix_e+DiscountedEV; %(d,aprime)-by-(a,j)

        % First, we want aprime conditional on (d,1,a,j)
        [~,maxindex1]=max(entireRHS_e,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex1,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_z
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-N_j-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2e(ReturnFn,n_d,n_z,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J,e_vals,ReturnFnParamsAgeMatrix,2);
        daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind+N_d*n2aprime*N_j*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j,N_z]);
        [Vtemp,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,e_c)=reshape(Vtemp,[N_a*N_j,N_z]);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind; % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_z
        Policy(1,:,:,:,e_c)=d_ind; % d
        Policy(2,:,:,:,e_c)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,:,e_c)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
    end
elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);

        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG

            ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2e(ReturnFn, n_d, special_n_z,special_n_e, N_j, d_gridvals, a_grid, a_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix,1);
            % fastOLG: ReturnMatrix is [d,aprime,a,j]

            entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z; %(d,aprime)-by-(a,j)

            % First, we want aprime conditional on (d,1,a,j)
            [~,maxindex1]=max(entireRHS_ze,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex1,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a-by-N_j
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-N_j
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_Par2e(ReturnFn,n_d,special_n_z,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_vals,e_vals,ReturnFnParamsAgeMatrix,2);
            daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(daprimej(:)),[N_d*n2long,N_a,N_j]);
            [Vtemp,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,z_c,e_c)=reshape(Vtemp,[N_a*N_j,1]);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aBind+N_d*N_a*jBind; % midpoint is n_d-by-1-by-n_a-by-N_j
            Policy(1,:,:,z_c,e_c)=d_ind; % d
            Policy(2,:,:,z_c,e_c)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,:,z_c,e_c)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        end
    end
end



%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% Leave the first dimension as is
% Policy=squeeze(Policy(1,:,:,:,:)+N_d*(Policy(2,:,:,:,:)-1)+N_d*N_a*(Policy(3,:,:,:,:)-1));


%% fastOLG with z & e, so need output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_z,N_e]);
% Note that in fastOLG, we do not separate d from aprime in Policy


end
