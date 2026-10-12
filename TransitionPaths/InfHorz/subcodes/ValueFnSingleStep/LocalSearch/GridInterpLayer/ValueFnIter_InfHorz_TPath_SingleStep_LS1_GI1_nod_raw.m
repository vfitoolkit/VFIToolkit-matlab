function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_nod_raw(Vnext,n_a,n_z, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with the grid interpolation layer, in TWO LAYERS: a coarse max over the window, then
% the interpolation layer around the winner. There is no one-pass alternative any more -- a raw that
% searched every fine point of the window in a single max was built, timed against this one and
% dropped (see below).
%
% THE WINDOW. The coarse argmax is CLAMPED to the window's interior, [loweredge+1, loweredge+2n-1],
% so that the interpolation layer's one coarse cell either side reaches exactly from coarse loweredge
% to coarse loweredge+2n. The union of layer-2 search sets over every possible layer-1 argmax is
% therefore EXACTLY the window: nothing inside it is unreachable and nothing outside it is reachable.
% That is what preserves the window guarantee |aprime-ref|<=2n coarse points, the ratchet's edge
% test, and the verification-sweep certification. Get the clamp wrong by one and the reachable set
% stops being the window.
%
% WHY TWO LAYERS AND NOT ONE. A one-pass search of the window costs 2*n*(1+ngridinterp)+1 fine
% points, growing at 2*(1+ngridinterp) per unit of nlocalsearch. This costs (2n+1) coarse plus a
% FIXED 2*ngridinterp+3 fine, growing at 2. Since nlocalsearch is ratcheted by the solver rather
% than set by the user, the one-pass cost fought the controller: it was most expensive exactly when
% the controller widened the window.
% MEASURED, n_a=1001, ngridinterp=5, CoreInfHorzTPathTests doPart(4)/(5):
%   with d (N_d=9): time tracked the point counts to two decimals at nlocalsearch=5 and 10, so the
%     kernel is memory-bandwidth bound there. Speed-up against the unrestricted sweep decayed
%     16.1x -> 2.6x one-pass as n went 1 -> 10, against 15.0x -> 8.9x here.
%   no d: latency bound, two-layer time flat at 0.14-0.16s while one-pass climbed 0.11 -> 0.22s.
%   Two layers lose only at nlocalsearch=1, by 0.02s with d and 0.05s without, and nlocalsearch=1 is
%   the ratchet's floor rather than its operating point.
%   Accuracy cost: the fine optimum was mislocated at 0 of 199980 state-periods without d and 0 of
%   2020 with d, and the L2 flag differed at 0 of 199980.
% Two layers are also structurally safe where one pass was not: the interior clamp puts the coarse
% argmax at most at N_a-1, so Policy's coarse channel can never reach N_a. The one-pass raw could
% emit N_a, which is out of range for the agent distribution -- see the spelling note at the bottom.
%
% Layer 2 cannot start until layer 1's reduction retires, which is one exposed latency per step. At
% small grids that is what makes this lose at nlocalsearch=1; at large grids it is negligible.
%
% aprimeReferencePolicy is a COARSE aprime index, [1,N_a,N_z], identical to every other localsearch
% raw with or without GI. Layer 1's answer IS the new reference, taken straight: it is already a
% coarse index, so there is no fine answer to compress and no rounding step to get wrong.
%
% nlocalsearch=0 is refused where the option is validated.

N_a=prod(n_a);
N_z=prod(n_z);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs; the seed sweeps are the ones that honour it.

n=vfoptions.nlocalsearch;

% Shaped like Policy on the way in, [N_a,N_z] internally
aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_z]);

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=2*n2short+3;            % the interpolation layer's search set, exactly as in the standard GI raw
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);
nC=2*n+1; % the coarse window

Policy=zeros(3,N_a,N_z,'gpuArray'); % lower coarse point, L2, L2flag

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% Expected value, and its interpolation onto the fine grid (both exactly as in the GI raw)
EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over z', leaving a singular second dimension

EVinterp=interp1(a_grid,EV,aprime_grid);

%% The window, in COARSE grid points
% The window SLIDES at the grid ends rather than being clamped, so it is always exactly 2n+1
% distinct COARSE points inside [1,N_a]. Same line as every other localsearch raw.
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_z], a coarse index

%% LAYER 1: the coarse max over the window
% On the COARSE EV, not EVinterp: layer 1 never needs the fine grid, so it gathers from an
% N_a-length array rather than the N_a+(N_a-1)*ngridinterp-length one.
coarseindexes=shiftdim(loweredge,-1)+(0:1:2*n)'; % [nC,N_a,N_z], coarse indices
ReturnMatrix1=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,a_grid(coarseindexes),a_grid,z_gridvals,ReturnFnParamsVec,2);
coarsez=coarseindexes+N_a*shiftdim((0:1:N_z-1),-1);
entireRHS1=ReturnMatrix1+DiscountFactorParamsVec*reshape(EV(coarsez(:)),[nC,N_a,N_z]);
[~,maxindex1]=max(entireRHS1,[],1);
% COARSEOFFSET IS 0-BASED, in every localsearch raw: it is the number of grid steps the window's
% argmax sits ABOVE loweredge, so it runs 0 to 2n and the grid index is loweredge+coarseoffset.
% One name and one base across all four raws deliberately -- a 1-based offset converted with
% loweredge+offset-1 reads almost identically, and getting the two confused shifts every window
% centre by one grid point without erroring, because the window absorbs it.
coarseoffset=reshape(maxindex1,[N_a,N_z])-1; % 0 to 2n, steps above loweredge

%% The new reference, the SAME object as the input: a COARSE aprime index per state, [1,N_a,N_z]
% THE LAYER 1 ANSWER, taken straight. This is the whole reason for having a coarse layer at all: the
% reference wants a coarse index and layer 1 produces one, so there is no fine answer to compress and
% no rounding step to get wrong. Note it is the UNCLAMPED argmax: if the coarse optimum sits on a
% window end then that is where the policy is heading, and the reference should say so even though
% the clamp stops layer 2 centring there.
aprimeReferencePolicyNew=reshape(loweredge+coarseoffset,[1,N_a,N_z]);

%% LAYER 2: the interpolation layer, one coarse cell either side of the clamped midpoint
% The clamp is what keeps layer 2 inside the window, and it also keeps every fine index in range:
% midpoint>=loweredge+1>=2 puts the lowest searched point at (1+n2short)*(midpoint-2)+1>=1, and
% midpoint<=loweredge+2n-1<=N_a-1 puts the highest at (1+n2short)*midpoint+1<=n2aprime.
midpoint=loweredge+min(max(coarseoffset,1),2*n-1); % [N_a,N_z]
aprimeindexes=shiftdim(midpoint+(midpoint-1)*n2short,-1)+(-n2short-1:1:1+n2short)'; % [n2long,N_a,N_z]
ReturnMatrix2=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals,ReturnFnParamsVec,2);
aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-1);
entireRHS2=ReturnMatrix2+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
[V,maxindexL2]=max(entireRHS2,[],1);
V=reshape(V,[N_a,N_z]);
% maxindexL2 is deliberately LEFT as [1,N_a,N_z]. Everything below it -- the flag's linear indexes,
% built from a 1-by-N_a aind and a 1-by-1-by-N_z zind, and the Policy channel slices -- is in that
% shape, which is the standard GI raw's convention. Reshaping it to [N_a,N_z] here makes the flag's
% & operands disagree in dimension 2 (N_a against N_z) rather than broadcasting.

%% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
% (1=all to lower, 2=usual, 3=all to upper.) Layer 2's window is structurally identical to the
% standard GI raw's -- n2long points, one coarse cell either side of a midpoint -- so these are that
% raw's lines unchanged, and the two cell endpoints are its window's two ends. Computed from
% maxindexL2 BEFORE the re-spelling below, again as in the standard GI raw.
aind=0:1:N_a-1;
zind=shiftdim(0:1:N_z-1,-1);
linidx_lower=1      +n2long*aind+n2long*N_a*zind;
linidx_upper=n2long +n2long*aind+n2long*N_a*zind;
isInfLower=(ReturnMatrix2(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix2(linidx_upper)==-Inf);
inLowerStrict=(maxindexL2>=2)         & (maxindexL2<=n2short+1);
inUpperStrict=(maxindexL2>=n2short+3) & (maxindexL2<=n2long-1);
Policy(3,:,:)=2+(inLowerStrict & isInfLower)-(inUpperStrict & isInfUpper);

%% Policy, re-spelled from (midpoint, offset within n2long) to (lower coarse point, L2)
% The standard GI raw's lines. maxindexL2 below n2short+2 means the answer is in the cell BELOW the
% midpoint, so the lower coarse point is midpoint-1 and L2 counts up from there; otherwise the cell
% is the one above and L2 counts from the midpoint itself. L2 ends up in [1,n2short+2].
Policy(1,:,:)=shiftdim(midpoint,-1); % midpoint is [N_a,N_z], so this is the one place a shiftdim is needed
Policy(2,:,:)=maxindexL2;            % already [1,N_a,N_z]
adjust=(Policy(2,:,:)<1+n2short+1);
Policy(1,:,:)=Policy(1,:,:)-adjust;
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1);

% NOTE ON THE ENCODING, inherited from the standard GI raw. L2 reaches n2short+2, so the window's
% top fine point can come out as either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2) depending
% on which side of the midpoint layer 2 landed. Those are the same fine point and neither spelling is
% canonical, so anything reading this Policy -- the ratchet's edge test in Step1 included -- must use
% the DECODED fine point (1+n2short)*(P1-1)+L2 and never the raw channels.
% The interior clamp also keeps the coarse channel at or below N_a-1, which the agent distribution
% requires because it always forms the upper grid point as P1+1.

end
