function [V,Policy2]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_noz_raw(Vnext,n_d,n_a, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step with divide-and-conquer and grid interpolation layer, with d, no z.
% Vnext is [N_a,1].
% There are no exogenous states to loop over, so vfoptions.lowmemory is irrelevant (every value is treated as 0).

N_d=prod(n_d);
N_a=prod(n_a);

% Preallocate
midpoints=zeros(N_d,1,N_a,'gpuArray');

aind=0:1:N_a-1; % already includes -1

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

Policy=zeros(4,N_a,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a (extra channel for PolicyL2flag pilot)

%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

EV=Vnext; % [N_a,1], no exogenous state so nothing to take expectations over
DiscountedEV=DiscountFactorParamsVec*shiftdim(EV,-1); % [1,aprime] — pre-discounted; broadcasts over d at every use site

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);
DiscountedEVinterp=DiscountFactorParamsVec*shiftdim(EVinterp,-1); % pre-discounted, only ever linear-indexed

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);
% (d,aprime,level1 a)

entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

% First, we want aprime conditional on (d,1,a)
[~,maxindex1]=max(entireRHS_ii,[],2);

% Just keep the 'midpoint' version of maxindex1 [as GI]
midpoints(:,1,level1ii)=maxindex1;

% Attempt for improved version
maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(:,1,ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
        % loweredge is n_d-by-1
        aprimeindexes=loweredge+(0:1:maxgap(ii));
        % aprime possibilities are n_d-by-maxgap(ii)+1
        % (reshape keeps a_grid(aprimeindexes) n_d-by-maxgap(ii)+1 when n_d=1, where indexing the column a_grid with a row would return a column)
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,3);
        % ReturnMatrix_ii is n_d-by-maxgap(ii)+1-by-length(curraindex)
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimeindexes(:)),[N_d,(maxgap(ii)+1)]); % autoexpands over curraindex in 3rd dim
        [~,maxindex]=max(entireRHS_ii,[],2);
        midpoints(:,1,curraindex)=maxindex+(loweredge-1);
    else
        loweredge=maxindex1(:,1,ii);
        midpoints(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex));
    end
end

% Turn this into the 'midpoint'
midpoints=max(min(midpoints,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
% midpoint is n_d-by-1-by-n_a
aprimeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
% aprime possibilities are n_d-by-n2long-by-n_a
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
% ReturnMatrix_ii is (n_d*n2long)-by-n_a
entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
[Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
V=shiftdim(Vtempii,1); % [N_a,1]
d_ind=rem(maxindexL2-1,N_d)+1;
allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
Policy(1,:)=d_ind; % d
Policy(2,:)=midpoints(allind); % midpoint
Policy(3,:)=ceil(maxindexL2/N_d); % aprimeL2ind
% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
L2offset = ceil(maxindexL2/N_d);
linidx_lower = d_ind                  + N_d*n2long*aind; % (d,L2=1,a) of the (n_d*n2long)-by-n_a ReturnMatrix_ii
linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind; % (d,L2=n2long,a)
isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
Policy(4,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:)=Policy(2,:)-adjust; % lower grid point
Policy(3,:)=adjust.*Policy(3,:)+(1-adjust).*(Policy(3,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%% Policy in transition paths
l_d=length(n_d);
Policy2=zeros(l_d+3,N_a); % +3 = midpoint, aprimeL2ind, L2flag (pilot)
% sort d variables
Policy2(1,:)=rem(Policy(1,:)-1,n_d(1))+1;
if l_d>1
    if l_d>2
        for ii=2:l_d-1
            Policy2(ii,:)=rem(ceil(Policy(1,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
        end
    end
    Policy2(l_d,:)=ceil(Policy(1,:)/prod(n_d(1:l_d-1)));
end
% rest are already in right shape
Policy2(l_d+1:end,:)=Policy(2:4,:);

end
