function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_nod_raw(Vnext,n_a,n_z, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with the grid interpolation layer. aprime is restricted to the 2*nlocalsearch+1
% coarse grid points around aprimeReferencePolicy, with ngridinterp points between each
% consecutive pair, exactly as standard GI puts them between each pair of coarse points.
%
% There is NO COARSE LAYER. Standard GI takes a full max over all N_a coarse points, turns the
% winner into a midpoint, and searches the fine points one coarse step either side. Here the
% window already says where to look, so the fine points spanning it ARE the search, in one max
% over 2*n*(1+ngridinterp)+1 points. At nlocalsearch=1 that is 2*ngridinterp+3 points, which is
% exactly what standard GI's second layer costs on its own: the entire coarse pass disappears.
%
% aprimeReferencePolicy is a COARSE aprime index, [1,N_a,N_z], identical to the no-GI case.
%
% nlocalsearch=0 is refused where the option is validated: it would give a single fine point and
% the interpolation layer would do nothing at all.

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
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);
nLS=2*n*(1+n2short)+1; % 2n+1 coarse points, n2short between each consecutive pair

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

%% The window, in fine grid points
% The window SLIDES at the grid ends rather than being clamped, so it is always exactly 2n+1
% distinct COARSE points inside [1,N_a]. Same line as the no-GI raw.
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_z], a coarse index
% Coarse point i sits at fine index 1+(i-1)*(1+n2short), which is how the GI raw writes
% midpoint+(midpoint-1)*n2short. Take every fine point from coarse loweredge to coarse loweredge+2n.
aprimeindexes=shiftdim(loweredge+(loweredge-1)*n2short,-1)+(0:1:2*n*(1+n2short))'; % [nLS,N_a,N_z]

ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals,ReturnFnParamsVec,2);
aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-1);
entireRHS=ReturnMatrix+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[nLS,N_a,N_z]);

[V,maxindex]=max(entireRHS,[],1);
V=reshape(V,[N_a,N_z]);

%% Policy, in the GI convention: lower coarse point, then L2 counting up from it
% maxindex counts fine points from coarse point loweredge, so k is how many fine steps above it the
% optimum sits. There are 1+n2short fine steps per coarse step, so dividing splits k into whole
% coarse steps (q) and a remainder inside the cell.
k=reshape(maxindex,[N_a,N_z])-1;
q=floor(k/(1+n2short));
r=k-q*(1+n2short); % 0 on the lower coarse point, up to n2short just below the upper one
Policy(1,:,:)=shiftdim(loweredge+q,-1); % lower coarse grid point
Policy(2,:,:)=shiftdim(r+1,-1);         % from 1 (on the lower point) to 1+n2short

% NOTE ON THE ENCODING. L2 here runs 1 to 1+n2short and NEVER reaches the n2short+2 that the GI raw
% can emit. That is not a different answer: (P1,L2=n2short+2) and (P1+1,L2=1) are the same fine
% point, and this names it the second way, always. The GI raw has no canonical rule either -- which
% of the two names it gives depends on which side of its midpoint the second-layer max fell -- so no
% encoding could reproduce it channel for channel. Compare the decoded fine point,
% (1+n2short)*(P1-1)+L2, not the raw channels.

%% L2 flag, to later avoid putting weight on an aprime where the ReturnFn is -Inf
% (1=all to lower, 2=usual, 3=all to upper.) The rule is the GI raw's: if the optimum is STRICTLY
% inside a cell and an endpoint of that cell is -Inf, force all the weight to the other endpoint.
% The GI raw can read those endpoints off the two ends of its window because its window is exactly
% the two cells either side of the midpoint. This window is 2n cells, so the endpoints have to be
% located: the cell the optimum landed in is [loweredge+q, loweredge+q+1], whose two coarse points
% sit at window positions q*(1+n2short)+1 and (q+1)*(1+n2short)+1.
% At n=1 this reduces to the GI raw line for line, which is checked in the test bank.
% Broader than the GI raw in one place: it only ever tests the FAR endpoint of the cell, because its
% near endpoint is the midpoint and the midpoint is the coarse argmax and so feasible. There is no
% coarse argmax here, so both endpoints of the cell are tested. If both were -Inf the two terms
% cancel and the flag stays 2, which is the same degenerate case the GI raw has.
% q reaches 2n when the optimum sits exactly on the TOP coarse point of the window. There is no cell
% above that point, so its upper endpoint would index past the end of ReturnMatrix. The flag is 2
% there whatever the lookup says, because k=2n*(1+n2short) forces r=0 and so strictlyinside is false,
% but the lookup is evaluated for every state before the mask is applied and so has to be in range.
% Clamping q by one cell does that and changes nothing where the value is used: r>=1 already implies
% q<=2n-1, since k=q*(1+n2short)+r cannot exceed 2n*(1+n2short).
qflag=min(q,2*n-1);
cellbase=nLS*(0:1:N_a-1)'+nLS*N_a*(0:1:N_z-1); % [N_a,N_z]
linidx_lower=qflag*(1+n2short)+1+cellbase;
linidx_upper=linidx_lower+(1+n2short);
isInfLower=(ReturnMatrix(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix(linidx_upper)==-Inf);
strictlyinside=(r>=1);
Policy(3,:,:)=shiftdim(2+(strictlyinside & isInfLower)-(strictlyinside & isInfUpper),-1);

end
