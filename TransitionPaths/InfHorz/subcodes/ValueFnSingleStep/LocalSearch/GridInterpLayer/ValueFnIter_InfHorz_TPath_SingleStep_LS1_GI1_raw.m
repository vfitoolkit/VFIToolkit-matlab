function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_raw(Vnext,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable AND the grid interpolation layer. For each d, aprime is
% restricted to the 2*nlocalsearch+1 coarse points around aprimeReferencePolicy(d,a,z), with
% ngridinterp points between each consecutive pair, and d is searched in full. One max over all of
% those fine points per d, then one over d. No coarse layer, as in the no-d GI version.
%
% The reference is a COARSE aprime index per (d,a,z), [N_d,N_a,N_z] -- see the header of
% ValueFnIter_InfHorz_TPath_SingleStep_LS1_raw for why it has to condition on d, and why that is
% not the same thing as being shaped like Policy.

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
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);
nLS=2*n*(1+n2short)+1; % 2n+1 coarse points, n2short between each consecutive pair

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

%% The window, in fine grid points, one per (d,a,z)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_d,N_a,N_z], a coarse index
% Coarse point i sits at fine index 1+(i-1)*(1+n2short), which is how the GI raw writes
% midpoint+(midpoint-1)*n2short. Take every fine point from coarse loweredge to coarse loweredge+2n.
aprimeindexes=reshape(loweredge+(loweredge-1)*n2short,[N_d,1,N_a,N_z])+(0:1:2*n*(1+n2short)); % [N_d,nLS,N_a,N_z]

% Level 3 returns [N_d,N_aprime,N_a,N_z] without folding d and aprime together, which is what the
% two maxima below want
ReturnMatrix=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_gridvals, ReturnFnParamsVec,3);
aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-2);
% Discounting applied AFTER the gather, the same order the GI raw uses, so that the two can be held
% to a bit-for-bit identity where they search the same points
entireRHS=ReturnMatrix+DiscountFactorParamsVec*EVinterp(aprimez);

%% Two maxima: the best fine point for each d, which is the new reference, then the best d
[Vd,offsetd]=max(entireRHS,[],2);                   % [N_d,1,N_a,N_z]
kd=reshape(offsetd,[N_d,N_a,N_z])-1;                % fine steps above coarse loweredge, per d
qd=floor(kd/(1+n2short));                           % whole coarse steps
rd=kd-qd*(1+n2short);                               % remainder inside the cell, 0 to n2short

%% The new reference, the SAME object as the input: a COARSE aprime index per (d,a,z)
% The per-d answer is a fine point, compressed to the nearest coarse one. It cannot run off the top:
% loweredge+qd=N_a only happens at qd=2n, which forces rd=0.
aprimeReferencePolicyNew=loweredge+qd+round(rd/(1+n2short));

[V,dstar]=max(reshape(Vd,[N_d,N_a,N_z]),[],1);      % [1,N_a,N_z]
V=reshape(V,[N_a,N_z]);
dstar=reshape(dstar,[N_a,N_z]);

% Everything below is read AT THE CHOSEN d, which is the thing to get wrong here
lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1);     % [N_a,N_z], into an [N_d,N_a,N_z] array
qstar=qd(lin);
rstar=rd(lin);
lowstar=loweredge(lin);

% THE TOP OF THE GRID. The agent distribution code always forms the UPPER grid point as P1+1 (see
% TransitionPath_InfHorz_substeps_Step2_AdjustPolicy), so the toolkit's GI convention is P1<=N_a-1,
% with the top grid point spelled (N_a-1, L2=ngridinterp+2) rather than (N_a, L2=1). The standard GI
% raw gets this free from clamping its midpoint to [2,N_a-1]. Here loweredge+q reaches N_a exactly
% when the window has slid to the top of the grid (loweredge=N_a-2n) and the optimum is the window's
% top coarse point (q=2n, which forces r=0) -- a household saving to the top of the grid. That one
% case is re-spelled onto the cell below, which is the SAME fine point.
% Left unspelled it is an out-of-range index in the sparse() that builds the distribution transition,
% and the error surfaces far away in AgentDist_InfHorz_TPath_SingleStep_nProbs_raw. The value fn
% tests compare the DECODED fine point, which is correct either way, so they cannot see this: it is
% a convention on the SPELLING, and only the agent distribution depends on it.
attop=(lowstar+qstar==N_a);
Policy(1,:,:)=shiftdim(dstar,-1);
Policy(2,:,:)=shiftdim(lowstar+qstar-attop,-1);        % lower coarse grid point, never N_a
Policy(3,:,:)=shiftdim(rstar+1+attop*(1+n2short),-1);  % L2, 1 to 1+n2short, or n2short+2 at the very top

% NOTE ON THE ENCODING, as in the no-d GI version: L2 runs 1 to 1+n2short, reaching the n2short+2
% the GI raw can emit only in the top-of-grid case just above. (P1,n2short+2) and (P1+1,1) are the same fine point and this names
% it the second way, always. Compare the decoded fine point, (1+n2short)*(P1-1)+L2.

%% L2 flag, for the cell the chosen d landed in
% Same rule and same reduction to the GI raw at nlocalsearch=1 as the no-d GI version, with the cell
% endpoints now located in the chosen d's row of ReturnMatrix. qd=2n means the optimum sits exactly
% on the top coarse point of the window, where there is no cell above and rd=0 makes the flag 2
% regardless, so qflag is clamped by one cell to keep the lookup in range.
qflagstar=min(qstar,2*n-1);
base=dstar+N_d*nLS*(0:1:N_a-1)'+N_d*nLS*N_a*(0:1:N_z-1); % window position 1, at the chosen d
linidx_lower=base+N_d*qflagstar*(1+n2short);
linidx_upper=linidx_lower+N_d*(1+n2short);
isInfLower=(ReturnMatrix(linidx_lower)==-Inf);
isInfUpper=(ReturnMatrix(linidx_upper)==-Inf);
strictlyinside=(rstar>=1);
Policy(4,:,:)=shiftdim(2+(strictlyinside & isInfLower)-(strictlyinside & isInfUpper),-1);

end
