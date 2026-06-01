function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_GI1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(3,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray'); % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
% When ReturnFn is -Inf on one of the course grid points, we will allow fine index between that and the neighbouring course grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iteration

%%
if vfoptions.lowmemory>0
    error('vfoptions.lowmemory>0 not supported for ValueFnIter_FHorz_TPath_SingleStep_GI1_noz_raw')
end

aind=gpuArray(0:1:N_a-1); % already includes -1

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
% n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

%% j=N_j: terminal age has no continuation in TPath
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,1);
%Calc the max and it's index
[~,maxindex]=max(ReturnMatrix,[],2);

% Turn this into the 'midpoint'
midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
% midpoint is n_d-1-by-n_a
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
% aprime possibilities are n_d-by-n2long-by-n_a
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d, d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
[Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);

% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
d_ind = rem(maxindexL2-1,N_d)+1;
L2offset = ceil(maxindexL2/N_d);
linidx_lower = d_ind + N_d*n2long*aind;
linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
PolicyL2flag(1,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

V(:,N_j)=shiftdim(Vtempii,1);
allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
Policy(1,:,N_j)=d_ind; % d
Policy(2,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
Policy(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,1);
    % (d,aprime,a)

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1); % [d,aprime,a]

    %Calc the max and it's index
    [~,maxindex]=max(entireRHS,[],2);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-1-by-n_a
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    d_ind = rem(maxindexL2-1,N_d)+1;
    L2offset = ceil(maxindexL2/N_d);
    linidx_lower = d_ind + N_d*n2long*aind;
    linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,jj)=shiftdim(Vtempii,1);
    allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
    Policy(1,:,jj)=d_ind; % d
    Policy(2,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
    Policy(3,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
end

% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy; PolicyL2flag];


end
