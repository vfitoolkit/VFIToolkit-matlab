function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_twolayer_raw(Vnext,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable AND the grid interpolation layer, done in TWO LAYERS: a
% coarse max over the window per d, then the interpolation layer around the winner per d. This is
% vfoptions.localsearchtwolayer=1; the one-pass alternative is
% ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_raw.
%
% BOTH LAYERS ARE CONDITIONAL ON d, exactly as the window is, and for the same reason: the reference
% has to exist for every d and not only the one this period's dstar picked, or the next sweep has no
% centre for the others. So layer 1 maxes over the coarse window separately for each d, and ITS
% ANSWER IS THE NEW REFERENCE -- a coarse index per (d,a,z), taken straight, with no fine answer to
% compress. Layer 2 then polishes, again per d, and only after that is the max over d taken.
%
% THE SAME WINDOW AS THE ONE-PASS RAW. The coarse argmax is CLAMPED to the window's interior,
% [loweredge+1, loweredge+2n-1], so layer 2's one coarse cell either side reaches exactly from coarse
% loweredge to coarse loweredge+2n -- the same fine points the one-pass raw searches. Both raws
% restrict aprime to identically the same set per d and differ only in how they search it.
%
% See the no-d version's header for the cost comparison, the n=1 coincidence, and the encoding note.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if vfoptions.lowmemory>0
    error('vfoptions.lowmemory is not used with vfoptions.localsearch=1 (the local window is already small in memory)')
end

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_z]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);
nC=2*n+1; % the coarse window

Policy=zeros(4,N_a,N_z,'gpuArray'); % d, lower coarse point, L2, L2flag

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% Expected value, and its interpolation onto the fine grid (both exactly as in the GI raw)
EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % [N_a,1,N_z]

EVinterp=interp1(a_grid,EV,aprime_grid);

%% The window, in COARSE grid points, one per (d,a,z)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_z], a coarse index

%% LAYER 1: the coarse max over the window, per d
% On the COARSE EV, not EVinterp: layer 1 never needs the fine grid.
coarseindexes=reshape(loweredge,[N_d,1,N_a,N_z])+(0:1:2*n); % [N_d,nC,N_a,N_z], coarse indices
% Level 3 returns [N_d,N_aprime,N_a,N_z] without folding d and aprime together, which is what the
% per-d max below wants
ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(coarseindexes), a_grid, z_gridvals, ReturnFnParamsVec,3);
coarsez=coarseindexes+N_a*shiftdim((0:1:N_z-1),-2);
entireRHS1=ReturnMatrix1+DiscountFactorParamsVec*EV(coarsez);
[~,offset1]=max(entireRHS1,[],2);                      % [N_d,1,N_a,N_z]
coarseoffset=reshape(offset1,[N_d,N_a,N_z])-1;         % 0 to 2n, steps above loweredge, per d

%% The new reference, the SAME object as the input: a COARSE aprime index per (d,a,z)
% THE LAYER 1 ANSWER, taken straight, per d. No fine answer to compress and no rounding step to get
% wrong. It is the UNCLAMPED argmax: if the coarse optimum for this d sits on a window end then that
% is where this d's policy is heading, and the reference should say so even though the clamp stops
% layer 2 centring there.
aprimeReferencePolicyNew=loweredge+coarseoffset;

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint, per d
% The clamp keeps layer 2 inside the window and keeps every fine index in range: midpoint>=2 puts
% the lowest searched point at (1+n2short)*(midpoint-2)+1>=1, and midpoint<=N_a-1 puts the highest at
% (1+n2short)*midpoint+1<=n2aprime.
midpoint=loweredge+min(max(coarseoffset,1),2*n-1);     % [N_d,N_a,N_z]
aprimeindexes=reshape(midpoint+(midpoint-1)*n2short,[N_d,1,N_a,N_z])+(-n2short-1:1:1+n2short); % [N_d,n2long,N_a,N_z]
ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_gridvals, ReturnFnParamsVec,3);
aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-2);
% Discounting applied AFTER the gather, the same order the GI raw uses, so that the two can be held
% to a bit-for-bit identity where they search the same points
entireRHS2=ReturnMatrix2+DiscountFactorParamsVec*EVinterp(aprimez);
[Vd,offsetL2]=max(entireRHS2,[],2);                    % [N_d,1,N_a,N_z]
L2d=reshape(offsetL2,[N_d,N_a,N_z]);                   % 1 to n2long, per d

%% Then the max over d
[V,dstar]=max(reshape(Vd,[N_d,N_a,N_z]),[],1);         % [1,N_a,N_z]
V=reshape(V,[N_a,N_z]);
dstar=reshape(dstar,[N_a,N_z]);

% Everything below is read AT THE CHOSEN d, which is the thing to get wrong here
lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1);        % [N_a,N_z], into an [N_d,N_a,N_z] array
L2star=L2d(lin);
midstar=midpoint(lin);

%% L2 flag, for the cell the chosen d landed in
% Layer 2's window is structurally the standard GI raw's -- n2long points, one coarse cell either
% side of a midpoint -- so this is that raw's rule, with the two cell endpoints at window positions
% 1 and n2long in the chosen d's row of ReturnMatrix2. Computed from L2star BEFORE the re-spelling
% below, again as in the standard GI raw.
base=dstar+N_d*n2long*(0:1:N_a-1)'+N_d*n2long*N_a*(0:1:N_z-1); % window position 1, at the chosen d
linidx_lower=base;
linidx_upper=base+N_d*(n2long-1);
isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
inLowerStrict=(L2star>=2)         & (L2star<=n2short+1);
inUpperStrict=(L2star>=n2short+3) & (L2star<=n2long-1);
Policy(4,:,:)=shiftdim(2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper),-1);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
% The standard GI raw's lines. L2star below n2short+2 means the answer is in the cell BELOW the
% midpoint, so the lower coarse point is midpoint-1 and L2 counts up from there; otherwise the cell
% is the one above and L2 counts from the midpoint itself. L2 ends up in [1,n2short+2].
adjust=(L2star<1+n2short+1);
Policy(1,:,:)=shiftdim(dstar,-1);
Policy(2,:,:)=shiftdim(midstar-adjust,-1);
Policy(3,:,:)=shiftdim(adjust.*L2star+(1-adjust).*(L2star-n2short-1),-1);

% NOTE ON THE ENCODING, which differs from the one-pass raw: L2 reaches n2short+2 here, so the
% window's top fine point can come out as either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2).
% Those are the same fine point. Anything comparing these raws, and the ratchet's edge test in
% Step1, must use the DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.

end
