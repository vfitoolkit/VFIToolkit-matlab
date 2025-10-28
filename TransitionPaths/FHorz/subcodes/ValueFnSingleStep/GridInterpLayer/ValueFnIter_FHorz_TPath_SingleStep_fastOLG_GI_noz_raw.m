function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI_noz_raw(V,n_d,n_a,N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions)
% fastOLG just means parallelize over "age" (j)

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(3,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for d  & aprime (layer 1 and layer 2)


% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
% n2aprime=length(aprime_grid);

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=V(:,2:end);
% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

DiscountedEV=repelem(DiscountFactorParamsVec.*EV,N_d,1,1); % [N_d*N_aprime,1,N_j]
DiscountedEVinterp=repelem(DiscountFactorParamsVec.*EVinterp,N_d,1,1); % [N_d*N_aprime,1,N_j]

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_noz_Par2(ReturnFn, n_d, N_j, d_grid, a_grid, a_grid, ReturnFnParamsAgeMatrix,2);

entireRHS=ReturnMatrix+DiscountedEV; %(d,aprime)-by-(a,j)

% First, we want aprime conditional on (d,1,a)
[~,maxindex1]=max(entireRHS,[],2);

% Just keep the 'midpoint' vesion of maxindex1 [as GI]
midpoints_jj(:,1,level1ii)=maxindex1;

% Turn this into the 'midpoint'
midpoint=max(min(maxindex1,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
% midpoint is n_d-1-by-n_a
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
% aprime possibilities are n_d-by-n2long-by-n_a
ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
daprimez=(1:1:N_d)'+N_d*(aprimeindexes-1);
entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d*n2long,N_a]);
[Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
V(:,jj)=shiftdim(Vtempii,1);
d_ind=rem(maxindexL2-1,N_d)+1;
allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
Policy(1,:,jj)=d_ind; % d
Policy(2,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
Policy(3,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=Policy(1,:,:)+N_d*(Policy(2,:,:)-1)+N_d*N_a*(Policy(3,:,:)-1);


end
