function [V, Policy]=ValueFnIter_FHorz_GulPesendorfer_GI1_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Gul-Pesendorfer with the grid interpolation layer. The choice set is the fine grid, so the
% most-tempting term is the max of v over the FINE grid, found by the same two-stage scheme as
% the main max but around v's OWN coarse argmax (otherwise the chosen fine point could be more
% tempting than the coarse max of v, making the self-control cost negative). The main max is
% the standard GI two-stage on the tempted objective u+v+beta*EV, with the L2 -Inf flag based
% on u+v (the temptation fn is -Inf exactly where the return fn is, by user contract, but the
% sum makes the flag robust either way). The most-tempting term is subtracted after the max.

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(3,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)
Policy(3,:,:)=2; % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
% When ReturnFn is -Inf on one of the course grid points, we will allow fine index between that and the neighbouring course grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iteration

%%
% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
% n2aprime=length(aprime_grid);

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
    TemptationMatrix=CreateReturnFnMatrix_Disc_noz(TemptationFn, 0, n_a, 0, a_grid, TemptationFnParamsVec,0);
    %Calc the max and it's index
    [~,maxindex]=max(ReturnMatrix+TemptationMatrix,[],1);

    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
    [~,maxindexT]=max(TemptationMatrix,[],1);
    midpointT=max(min(maxindexT,n_a-1),2);
    aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexesT),a_grid,TemptationFnParamsVec);
    MostTempting=max(TemptationMatrix_Tii,[],1);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a-by-n_bothz-by-n_e
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexes),a_grid,TemptationFnParamsVec);
    Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
    [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);

    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (Ftemp_ii(1,    :) == -Inf);
    isInfUpper    = (Ftemp_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,N_j)=shiftdim(Vtempii-MostTempting,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoint),-1); % midpoint
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
    TemptationMatrix=CreateReturnFnMatrix_Disc_noz(TemptationFn, 0, n_a, 0, a_grid, TemptationFnParamsVec,0);
    entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [~,maxindex]=max(entireRHS,[],1);

    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
    [~,maxindexT]=max(TemptationMatrix,[],1);
    midpointT=max(min(maxindexT,n_a-1),2);
    aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexesT),a_grid,TemptationFnParamsVec);
    MostTempting=max(TemptationMatrix_Tii,[],1);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a-by-n_z
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexes),a_grid,TemptationFnParamsVec);
    Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
    entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (Ftemp_ii(1,    :) == -Inf);
    isInfUpper    = (Ftemp_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,N_j)=shiftdim(Vtempii-MostTempting,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoint),-1); % midpoint
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=V(:,jj+1);

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
    TemptationMatrix=CreateReturnFnMatrix_Disc_noz(TemptationFn, 0, n_a, 0, a_grid, TemptationFnParamsVec,0);
    entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [~,maxindex]=max(entireRHS,[],1);

    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
    [~,maxindexT]=max(TemptationMatrix,[],1);
    midpointT=max(min(maxindexT,n_a-1),2);
    aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short)';
    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexesT),a_grid,TemptationFnParamsVec);
    MostTempting=max(TemptationMatrix_Tii,[],1);

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a-by-n_z
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexes),a_grid,TemptationFnParamsVec);
    Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
    entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (Ftemp_ii(1,    :) == -Inf);
    isInfUpper    = (Ftemp_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,jj)=shiftdim(Vtempii-MostTempting,1);
    Policy(1,:,jj)=shiftdim(squeeze(midpoint),-1); % midpoint
    Policy(2,:,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind
end


%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 1:nshort+2 up from this.
adjust=(Policy(2,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:)=Policy(1,:,:)-adjust; % lower grid point
Policy(2,:,:)=Policy(2,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

end
