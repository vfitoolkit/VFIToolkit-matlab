function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_DC1_GI1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.

N_a=prod(n_a);

Policy=zeros(3,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for midpoint, L2, L2flag

%%
% Preallocate
midpoints_jj=zeros(1,N_a,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));

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

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

%Calc the max and it's index
[~,maxindex]=max(ReturnMatrix_ii,[],1);

% Just keep the 'midpoint' version of maxindex1 [as GI]
midpoints_jj(1,level1ii)=maxindex;

for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
    [~,maxindex]=max(ReturnMatrix_ii,[],1);
    midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
end

% Turn this into the 'midpoint'
midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
% midpoint is 1-by-n_a
aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
% aprime possibilities are n2long-by-n_a
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
[Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
V(:,N_j)=shiftdim(Vtempii,1);
Policy(1,:,N_j)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
Policy(2,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
% L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
Policy(3,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

    EV=VKronNext_j;

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;
    %Calc the max and it's index
    [~,maxindex]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(1,level1ii)=maxindex;

    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,jj)=shiftdim(Vtempii,1);
    Policy(1,:,jj)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
    Policy(2,:,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
end


%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:)=Policy(1,:,:)-adjust; % lower grid point
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

end
