function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% fastOLG just means parallelize over "age" (j)
N_a=prod(n_a);

% V=zeros(N_a,N_j,'gpuArray'); % V is over (a,j)
Policy=zeros(2,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime (layer 1 and layer 2)

%%

% Preallocate
midpoints_jj=zeros(1,N_a,N_j,'gpuArray');

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);


%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

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

DiscountedEV=DiscountFactorParamsVec.*EV;
DiscountedEVinterp=DiscountFactorParamsVec.*EVinterp;

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_noz_Par2(ReturnFn, N_j, a_grid, a_grid(level1ii), ReturnFnParamsAgeMatrix,1);
% [N_aprime,level1n,N_j]

entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % (aprime,a and j), autofills a dimension for expectation term

% Calc the max and it's index
[~,maxindex1]=max(entireRHS_ii,[],1);

% Just keep the 'midpoint' vesion of maxindex1 [as GI]
midpoints_jj(1,level1ii,:)=maxindex1;

% Attempt for improved version
maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    if maxgap(ii)>0
        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
        % loweredge is 1-by-1-by-N_j
        aprimeindexes=loweredge+(0:1:maxgap(ii))'; % ' due to no d
        % aprimeindexes is 1-by-maxgap(ii)+1-by-N_j
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_noz_Par2(ReturnFn,N_j,a_grid(aprimeindexes),a_grid(level1ii(ii)+1:level1ii(ii+1)-1),ReturnFnParamsAgeMatrix,2);
        aprimej=repelem(aprimeindexes,1,level1iidiff(ii),1)+N_a*jind; % the current aprimeii(ii):aprimeii(ii+1)
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimej(:)),[maxgap(ii)+1,level1iidiff(ii),N_j]);
       [~,maxindex]=max(entireRHS_ii,[],1);
        midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
    else
        loweredge=maxindex1(1,ii,:);       
        midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
    end
end

% Turn this into the 'midpoint'
midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
% midpoint is 1-by-n_a-by-N_j
aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
% aprime possibilities are n2long-by-n_a-by-N_j
ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_noz_Par2(ReturnFn,N_j,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsAgeMatrix,2);
aprimej=aprimeindexes+n2aprime*jind;
entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimej(:)),[n2long,N_a,N_j]);
[Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
V=shiftdim(Vtempii,1);
Policy(1,:,:)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
Policy(2,:,:)=shiftdim(maxindexL2,-1); % aprimeL2ind

% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:)=Policy(1,:,:)-adjust; % lower grid point
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=squeeze(Policy(1,:,:)+N_a*(Policy(2,:,:)-1));



end