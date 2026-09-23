function [V,Policy]=ValueFnIter_FHorz_GulPesendorfer_DC1_GI1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Gul-Pesendorfer with divide-and-conquer plus the grid interpolation layer. The DC pass runs
% on the tempted objective u+v+beta*EV and produces the midpoints for the GI refinement; the
% most-tempting term is the max of v over the FINE grid (the choice set under GI), found by
% its own two-stage scheme: v's coarse argmax is computed EXACTLY over the full aprime grid
% (full-column temptation matrices one a-slab at a time -- never a window max, and no
% monotonicity assumption on v), then refined on the fine points around it. The most-tempting
% term is subtracted from V after the max.
% divide-and-conquer for length(n_a)==1

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(3,N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z
Policy(3,:,:)=2; % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
% When ReturnFn is -Inf on one of the course grid points, we will allow fine index between that and the neighbouring course grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iteration

%%
% Preallocate
midpoints_jj=zeros(1,N_a,'gpuArray');
midpointsT_jj=zeros(1,N_a,'gpuArray'); % v's own coarse argmax (for the most-tempting term)

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));

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

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn, a_grid, a_grid(level1ii), TemptationFnParamsVec);

    %Calc the max and it's index
    [~,maxindex]=max(ReturnMatrix_ii+TemptationMatrix_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(1,level1ii)=maxindex;

    % v's own coarse argmax at the level-1 stations (full aprime grid)
    [~,maxindexT1]=max(TemptationMatrix_ii,[],1);
    midpointsT_jj(1,level1ii)=maxindexT1;

    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        % v's coarse argmax over the FULL aprime grid for these a (never just the window)
        TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn, a_grid, a_grid(curraindex), TemptationFnParamsVec);
        [~,maxindexT]=max(TemptationMatrix_full,[],1);
        midpointsT_jj(1,curraindex)=maxindexT;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), TemptationFnParamsVec);
        [~,maxindex]=max(ReturnMatrix_ii+TemptationMatrix_ii,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexes),a_grid,TemptationFnParamsVec);
    Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
    [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);
    % Most-tempting term: fine refinement of v around v's own coarse argmax
    midpointsT_jj=max(min(midpointsT_jj,n_a-1),2);
    aprimeindexesT=(midpointsT_jj+(midpointsT_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexesT),a_grid,TemptationFnParamsVec);
    MostTempting=max(TemptationMatrix_Tii,[],1);
    V(:,N_j)=shiftdim(Vtempii-MostTempting,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (Ftemp_ii(1,     :) == -Inf);
    isInfUpper    = (Ftemp_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]); % Using V_Jplus1

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn, a_grid, a_grid(level1ii), TemptationFnParamsVec);
    entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*EV;
    %Calc the max and it's index
    [~,maxindex]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(1,level1ii)=maxindex;

    % v's own coarse argmax at the level-1 stations (full aprime grid)
    [~,maxindexT1]=max(TemptationMatrix_ii,[],1);
    midpointsT_jj(1,level1ii)=maxindexT1;

    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        % v's coarse argmax over the FULL aprime grid for these a (never just the window)
        TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn, a_grid, a_grid(curraindex), TemptationFnParamsVec);
        [~,maxindexT]=max(TemptationMatrix_full,[],1);
        midpointsT_jj(1,curraindex)=maxindexT;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), TemptationFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*EV(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexes),a_grid,TemptationFnParamsVec);
    Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
    % aprime=aprimeindexes;
    entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    % Most-tempting term: fine refinement of v around v's own coarse argmax
    midpointsT_jj=max(min(midpointsT_jj,n_a-1),2);
    aprimeindexesT=(midpointsT_jj+(midpointsT_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexesT),a_grid,TemptationFnParamsVec);
    MostTempting=max(TemptationMatrix_Tii,[],1);
    V(:,N_j)=shiftdim(Vtempii-MostTempting,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (Ftemp_ii(1,     :) == -Inf);
    isInfUpper    = (Ftemp_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
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

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn, a_grid, a_grid(level1ii), TemptationFnParamsVec);
    entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*EV;
    %Calc the max and it's index
    [~,maxindex]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(1,level1ii)=maxindex;

    % v's own coarse argmax at the level-1 stations (full aprime grid)
    [~,maxindexT1]=max(TemptationMatrix_ii,[],1);
    midpointsT_jj(1,level1ii)=maxindexT1;

    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        % v's coarse argmax over the FULL aprime grid for these a (never just the window)
        TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn, a_grid, a_grid(curraindex), TemptationFnParamsVec);
        [~,maxindexT]=max(TemptationMatrix_full,[],1);
        midpointsT_jj(1,curraindex)=maxindexT;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), TemptationFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountFactorParamsVec*EV(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexes),a_grid,TemptationFnParamsVec);
    Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
    % aprime=aprimeindexes;
    entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    % Most-tempting term: fine refinement of v around v's own coarse argmax
    midpointsT_jj=max(min(midpointsT_jj,n_a-1),2);
    aprimeindexesT=(midpointsT_jj+(midpointsT_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_nod_noz(TemptationFn,aprime_grid(aprimeindexesT),a_grid,TemptationFnParamsVec);
    MostTempting=max(TemptationMatrix_Tii,[],1);
    V(:,jj)=shiftdim(Vtempii-MostTempting,1);
    Policy(1,:,jj)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
    Policy(2,:,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (Ftemp_ii(1,     :) == -Inf);
    isInfUpper    = (Ftemp_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
end



% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:)=Policy(1,:,:)-adjust; % lower grid point
Policy(2,:,:)=Policy(2,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

end
