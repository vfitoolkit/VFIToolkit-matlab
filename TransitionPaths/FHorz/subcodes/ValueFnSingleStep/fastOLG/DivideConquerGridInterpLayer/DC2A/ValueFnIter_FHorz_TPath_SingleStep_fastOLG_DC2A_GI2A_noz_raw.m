function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_GI2A_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions)
% fastOLG just means parallelize over "age" (j)
% DC2A_GI2A (no z): divide-and-conquer in a1 (iterate a2), grid interpolation on a1

N_d=prod(n_d);
N_a=prod(n_a);

% V is over (a,j) (no z); Policy first dim is (d, a1prime midpoint, a2prime, a1prime L2)
Policy=zeros(4,N_a,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%% Split endogenous state into a1 (DC) and a2 (iterate)
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity (DC on a1 only)
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));

% Grid interpolation (GI on a1)
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% precompute index arrays
% ReturnMatrix_ii Level=1/3 shape: [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_j] (a2prime in dim 3, j in dim 6)
% ReturnMatrix_ii Level=2 shape after reshape: [N_d*n2long*N_a2, N_a, N_j]
% DiscountedEV shape: [1, N_a1, N_a2, 1, 1, N_j]; DiscountedEVinterp shape: [1, N_a1fine, N_a2, 1, 1, N_j]
a2Bind=shiftdim(gpuArray(0:1:N_a2-1),-1); % [1,1,N_a2] -> a2prime offset in dim 3
a2ind =shiftdim(gpuArray(0:1:N_a2-1),-1); % [1,1,N_a2]
jind  =shiftdim(gpuArray(0:1:N_j-1),-4);  % [1,1,1,1,1,N_j] -> age offset in dim 6

% Final Policy linear indexers (target shape [1, N_a, N_j]):
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1); % [1,N_a]
jBind =shiftdim(gpuArray(0:1:N_j-1),-1); % [1,1,N_j]

midpoints_jj=zeros(N_d,1,N_a2,N_a1,N_a2,N_j,'gpuArray');

%% First, create the big 'next period (of transition path) expected value fn.

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.EVpre==0
    EV=zeros(N_a,N_j,'gpuArray');
    EV(:,1:N_j-1)=V(:,2:end);
elseif vfoptions.EVpre==1
    EV=reshape(V,[N_a,N_j]); % input V is of size [N_a,N_j]
end
% Reshape to (a1,a2,1,1,j) for DC2A-style broadcast against ReturnMatrix dims (a1prime,a2prime,a1,a2,j)
EV=reshape(EV,[N_a1,N_a2,1,1,N_j]);
% Interpolate over a1prime_grid (along dim 1)
EVinterp=interp1(a1_grid,EV,a1prime_grid);

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV; % [N_a1, N_a2, 1, 1, N_j]
DiscountedEV=shiftdim(DiscountedEV,-1); % [1, N_a1, N_a2, 1, 1, N_j]; will autoexpand d in dim 1
DiscountedEVinterp=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EVinterp; % [N_a1fine, N_a2, 1, 1, N_j]
DiscountedEVinterp=shiftdim(DiscountedEVinterp,-1); % [1, N_a1fine, N_a2, 1, 1, N_j]

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_noz(ReturnFn, n_d, N_j, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, ReturnFnParamsAgeMatrix,1);
% shape: [N_d, N_a1, N_a2, level1n, N_a2, N_j]

entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

% First, we want a1prime conditional on (d,1,a2prime,a1=level1ii,a2,j)
[~,maxindex1]=max(entireRHS_ii,[],2);
% maxindex1 shape: [N_d, 1, N_a2, level1n, N_a2, N_j]

midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-N_j
        a1primeindexes=loweredge+(0:1:maxgap(ii));
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_noz(ReturnFn, n_d, N_j, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, ReturnFnParamsAgeMatrix,3);
        % Linear-index DiscountedEV: stride 1 for a1prime (dim 2), N_a1 for a2prime (dim 3), N_a for j (dim 6)
        aprimej=a1primeindexes+N_a1*a2Bind+N_a*jind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimej(:)),[N_d,(maxgap(ii)+1),N_a2,1,N_a2,N_j]);
        [~,maxindex]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
    else
        loweredge=maxindex1(:,1,:,ii,:,:);
        midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
    end
end

midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
% midpoint is n_d-by-1-by-n_a2-by-n_a1-by-n_a2-by-N_j
a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
% aprime possibilities are n_d-by-n2long-by-n_a2-by-n_a1-by-n_a2-by-N_j
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_noz(ReturnFn,n_d,N_j,d_gridvals,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, ReturnFnParamsAgeMatrix,2);
% shape after reshape: [N_d*n2long*N_a2, N_a, N_j]
aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*jind;
entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[N_d*n2long*N_a2,N_a,N_j]);
[Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
V=reshape(Vtempii,[N_a,N_j]);
maxindexL2d=rem(maxindexL2-1,N_d)+1;
maxindexL2a=ceil(maxindexL2/N_d);
maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
maxindexL2a2=ceil(maxindexL2a/n2long);
Policy(1,:,:)=maxindexL2d; % d
% midpoints_jj is [N_d, 1, N_a2, N_a1, N_a2, N_j]; strides: d=1, a2prime=N_d, a1=N_d*N_a2, a2=N_d*N_a2*N_a1, j=N_d*N_a2*N_a
midindex=maxindexL2d+N_d*(maxindexL2a2-1)+N_d*N_a2*a12ind+N_d*N_a2*N_a*jBind;
Policy(2,:,:)=midpoints_jj(midindex); % a1prime midpoint
Policy(3,:,:)=maxindexL2a2; % a2prime
Policy(4,:,:)=maxindexL2a1; % a1primeL2ind

% L2 flag
linidx_lower = maxindexL2d                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind;
linidx_upper = maxindexL2d + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*a12ind + N_d*n2long*N_a2*N_a*jBind;
isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
PolicyL2flag(1,:,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);


%% Currently Policy(2,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(4,:,:)=adjust.*Policy(4,:,:)+(1-adjust).*(Policy(4,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy; PolicyL2flag];


end
