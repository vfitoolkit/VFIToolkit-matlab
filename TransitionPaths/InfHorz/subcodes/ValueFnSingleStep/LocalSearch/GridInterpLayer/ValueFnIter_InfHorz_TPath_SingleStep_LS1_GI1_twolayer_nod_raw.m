function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_twolayer_nod_raw(Vnext,n_a,n_z, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with the grid interpolation layer, done in TWO LAYERS: a coarse max over the window,
% then the interpolation layer around the winner. This is vfoptions.localsearchtwolayer=1; the
% one-pass alternative is ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_nod_raw.
%
% THE SAME WINDOW AS THE ONE-PASS RAW, which is what makes the two comparable. The coarse argmax is
% CLAMPED to the window's interior, [loweredge+1, loweredge+2n-1], so that the interpolation layer's
% one coarse cell either side reaches exactly from coarse loweredge to coarse loweredge+2n -- the
% same fine points the one-pass raw searches in a single max. So both raws restrict aprime to
% identically the same set and differ only in HOW they search it: the one-pass raw exactly, this one
% by coarse argmax and then polish. Every guarantee the window carries is therefore unchanged, and
% so is the ratchet's edge test and the verification-sweep certification.
%
% WHY IT MIGHT BE FASTER. The one-pass search set is 2*n*(1+ngridinterp)+1 fine points, growing at
% 2*(1+ngridinterp) per unit of nlocalsearch. Here it is (2n+1) coarse plus a FIXED 2*ngridinterp+3
% fine, growing at 2. The kernel is memory-bandwidth bound at large grids and the bytes touched are
% proportional to the search set, so the ratio of those counts is the thing to beat. It crosses over
% at n=1+3/(2*ngridinterp), which is below 2 for any sensible ngridinterp. Against that, layer 2
% cannot start until layer 1's reduction retires, which is one exposed latency per step that does
% not overlap with anything. Which wins is measured in the test bank, not asserted here.
%
% AT n=1 THE TWO RAWS AGREE EXACTLY. The clamp forces the midpoint to loweredge+1, so layer 2's
% 2*ngridinterp+3 points are the one-pass raw's 2*1*(1+ngridinterp)+1 points, in the same order, so
% the max is over a bit-identical array. V and the decoded fine policy are identical; only the
% channel spelling and the reference differ (see below).
%
% aprimeReferencePolicy is a COARSE aprime index, [1,N_a,N_z], identical to every other localsearch
% raw with or without GI.
%
% nlocalsearch=0 is refused where the option is validated.

N_a=prod(n_a);
N_z=prod(n_z);

if vfoptions.lowmemory>0
    error('vfoptions.lowmemory is not used with vfoptions.localsearch=1 (the local window is already small in memory)')
end

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

% NOTE ON THE ENCODING, which differs from the one-pass raw. That raw always names a fine point as
% (cell's lower coarse point, L2 in [1,1+n2short]) and never emits L2=n2short+2. This one inherits
% the standard GI raw's spelling, where L2 reaches n2short+2, so the window's top fine point can come
% out as either (loweredge+2n, 1) or (loweredge+2n-1, n2short+2) depending on which side of the
% midpoint layer 2 landed. Those are the same fine point. Anything comparing these raws, and the
% ratchet's edge test in Step1, must therefore use the DECODED fine point
% (1+n2short)*(P1-1)+L2 and never the raw channels.

end
