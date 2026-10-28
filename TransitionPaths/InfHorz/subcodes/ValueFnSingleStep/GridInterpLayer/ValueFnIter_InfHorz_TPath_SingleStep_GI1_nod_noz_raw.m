function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_nod_noz_raw(Vnext,n_a, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step with grid interpolation layer, no d, no z.
% Vnext is [N_a,1].
% There are no exogenous states to loop over, so vfoptions.lowmemory is irrelevant (every value is treated as 0).

N_a=prod(n_a);

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

Policy=zeros(3,N_a,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a (extra channel for PolicyL2flag pilot)

%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

EV=Vnext; % [N_a,1], no exogenous state so nothing to take expectations over

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, [], a_grid, ReturnFnParamsVec,0);
% (aprime,a)

entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

%Calc the max and it's index
[~,maxindex]=max(entireRHS,[],1);

% Turn this into the 'midpoint'
midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
% midpoint is 1-by-n_a
aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
% aprime possibilities are n2long-by-n_a
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
% ReturnMatrix_ii is n2long-by-n_a
entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
[Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
V=shiftdim(Vtempii,1); % [N_a,1]
Policy(1,:)=midpoint; % midpoint
Policy(2,:)=maxindexL2; % aprimeL2ind
% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
aind = 0:1:N_a-1;
linidx_lower = 1      + n2long*aind; % (1,a) of the n2long-by-n_a ReturnMatrix_ii
linidx_upper = n2long + n2long*aind; % (n2long,a)
isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
Policy(3,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:)=Policy(1,:)-adjust; % lower grid point
Policy(2,:)=adjust.*Policy(2,:)+(1-adjust).*(Policy(2,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)


end
