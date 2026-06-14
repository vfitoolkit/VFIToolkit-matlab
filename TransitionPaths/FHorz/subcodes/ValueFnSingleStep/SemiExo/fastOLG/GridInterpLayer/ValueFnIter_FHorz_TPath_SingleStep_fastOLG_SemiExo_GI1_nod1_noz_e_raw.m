function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_GI1_nod1_noz_e_raw(V,n_d2,n_a,n_semiz,n_e,N_j, d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z-by-e

n_d=n_d2;
N_d2=prod(n_d2);
N_d=N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);
n_bothz=n_semiz; % These are the return function arguments
N_bothz=N_semiz;
N_e=prod(n_e);

Policy=zeros(4,N_a,N_j,N_bothz,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z (d, midpoint, aprimeL2ind, L2flag)

d_gridvals=d2_gridvals;
bothz_gridvals_J=shiftdim(semiz_gridvals_J,-3); % [1,1,1,N_j,N_semiz,l_semiz]
pi_semiz_J=permute(pi_semiz_J,[4,2,1,3]); % (j,semiz',semiz,d2)
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]);


%% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_bothz-1),-3);
aBind=gpuArray(0:1:N_a-1);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
zBind=shiftdim(gpuArray(0:1:N_bothz-1),-2);
eBind=shiftdim(gpuArray(0:1:N_e-1),-3);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_bothz (note: N_aprime is just equal to N_a)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(1:end-N_a,:,:),3); zeros(N_a,N_bothz,'gpuArray')]; % I use zeros in j=N_j so that can just use the transition probabilities to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_bothz]);

% Expectations over the semi-exogenous state depend on d2: compute them for each d2 and stack over d2
EV=zeros(N_a,1,N_j,N_bothz,N_d2,'gpuArray');
for d2_c=1:N_d2
    pi_bothz=pi_semiz_J(:,:,:,d2_c); % (j,semiz',semiz)
    EV_d2=EVpre.*shiftdim(pi_bothz,-2);
    EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV(:,1,:,:,d2_c)=reshape(sum(EV_d2,4),[N_a,1,N_j,N_bothz]); % (aprime,1,j,bothz)
end

% Interpolate EV over aprime_grid (interp1 operates along the first dimension)
EVinterp=interp1(a_grid,EV,aprime_grid); % (n2aprime,1,N_j,N_bothz,N_d2)

DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV;
DiscountedEV=permute(DiscountedEV,[5,1,2,3,4]); % [N_d2,N_aprime,1,N_j,N_bothz] (d2 indexes the first dimension)

DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,N_j]).*EVinterp;
DiscountedEVinterp=permute(DiscountedEVinterp,[5,1,2,3,4]); % [N_d2,n2aprime,1,N_j,N_bothz]

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_bothz, n_e, N_j, d_gridvals, a_grid, a_grid, bothz_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % fastOLG: ReturnMatrix is [d,aprime,a,j,z,e]

    entireRHS=ReturnMatrix+DiscountedEV; %  [d,aprime,a,j,z,e]

    % First, we want aprime conditional on (d,1,a,j)
    [~,maxindex1]=max(entireRHS,[],2);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex1,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_bothz-by-n_e
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a-by-N_j-by-n_bothz-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d,n_bothz,n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid, bothz_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,2);
    daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind+N_d*n2aprime*N_j*zind;
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j,N_bothz,N_e]);
    [V,maxindexL2]=max(entireRHS_ii,[],1);
    V=reshape(V,[N_a*N_j,N_bothz,N_e]);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind+N_d*N_a*N_j*N_bothz*eBind; % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_bothz-by-n_e
    Policy(1,:,:,:,:)=d_ind; % d
    Policy(2,:,:,:,:)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
    Policy(3,:,:,:,:)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    L2offset=ceil(maxindexL2/N_d);
    linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind+N_d*n2long*N_a*N_j*N_bothz*eBind;
    linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind+N_d*n2long*N_a*N_j*N_bothz*eBind;
    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
    inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
    Policy(4,:,:,:,:)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_bothz,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % bothz_gridvals_J has shape (j,prod(n_bothz),l_z) for fastOLG

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_bothz, special_n_e, N_j, d_gridvals, a_grid, a_grid, bothz_gridvals_J,e_vals, ReturnFnParamsAgeMatrix,1);
        % fastOLG: ReturnMatrix is [d,aprime,a,j]

        entireRHS_e=ReturnMatrix_e+DiscountedEV; %(d,aprime)-by-(a,j)

        % First, we want aprime conditional on (d,1,a,j)
        [~,maxindex1]=max(entireRHS_e,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex1,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_bothz
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-N_j-by-n_bothz
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn,n_d,n_bothz,special_n_e,N_j,d_gridvals,aprime_grid(aprimeindexes),a_grid,bothz_gridvals_J,e_vals,ReturnFnParamsAgeMatrix,2);
        daprimej=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*jind+N_d*n2aprime*N_j*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(daprimej(:)),[N_d*n2long,N_a,N_j,N_bothz]);
        [Vtemp,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,e_c)=reshape(Vtemp,[N_a*N_j,N_bothz]);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aBind+N_d*N_a*jBind+N_d*N_a*N_j*zBind; % midpoint is n_d-by-1-by-n_a-by-N_j-by-n_bothz
        Policy(1,:,:,:,e_c)=d_ind; % d
        Policy(2,:,:,:,e_c)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,:,e_c)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset=ceil(maxindexL2/N_d);
        linidx_lower=d_ind                  +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
        linidx_upper=d_ind+N_d*(n2long-1)   +N_d*n2long*aBind+N_d*n2long*N_a*jBind+N_d*n2long*N_a*N_j*zBind;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
        inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
        Policy(4,:,:,:,e_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
    end
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end



%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)


%% fastOLG with z & e, so need output to take certain shapes
% V=reshape(V,[N_a*N_j,N_bothz,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_bothz,N_e]);


end
