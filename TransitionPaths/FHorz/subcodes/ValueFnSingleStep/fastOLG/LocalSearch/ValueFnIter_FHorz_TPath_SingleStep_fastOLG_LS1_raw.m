function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, fastOLG (all ages at once), with z, with a decision variable.
% fastOLG is done as (a,j,z): V is (a,j)-by-z, Policy is (1,a,j,z).
% This is EXACTLY slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LS1_raw, with every age in one
% kernel: aprime is restricted to a window of 2*nlocalsearch+1 grid points around aprimeReferencePolicy,
% d is searched in FULL, the reference is CONDITIONAL ON d, and the max is taken in two stages (over
% aprime per d, which is the new reference, then over d). See that raw for the reasoning.
%
% THE REFERENCE is [N_d,N_a,N_j,N_z], one window per (d,a,j,z).
% j=N_j: EV is zero there via the zero slice appended to pi_z_J, as in the unrestricted fastOLG raw.
%
% Policy comes out in the SAME Kron form as ValueFnIter_FHorz_TPath_SingleStep_fastOLG_raw, the folded
% (d,aprime) index with d fastest, so the dispatcher's UnKron applies unchanged.
% Tie-breaking differs from the folded raw (lowest d among the per-d maxima, rather than linear order);
% two names for one value, V is unaffected.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

% ONLY THE vfoptions.EVpre==0 PATH. Local search is reachable only from a transition path, where
% EVpre=0 (EVpre=1 is the Matched Expectations Path, which never sets localsearch).
if vfoptions.EVpre~=0
    error('vfoptions.localsearch requires vfoptions.EVpre=0 (ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_raw)')
end

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_j,N_z]);

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);

%% First, create the big 'next period (of transition path) expected value fn.
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
EVpre=zeros(N_a,1,N_j,N_z);
EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z)

DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV; % [N_a,1,N_j,N_z]

%% The window, one per (d,a,j,z)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n);                % [N_d,N_a,N_j,N_z]
aprimeindexes=reshape(loweredge,[N_d,1,N_a,N_j,N_z])+(0:1:2*n);       % [N_d,2n+1,N_a,N_j,N_z], d fastest

%% Return matrix on the window only
% Level 3 returns [N_d,N_aprime,N_a,N_j,N_z] without folding d and aprime together (aprime not shifted)
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_z, N_j, d_gridvals, a_grid(aprimeindexes), a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,3);

aprimejz=aprimeindexes+N_a*jind+N_a*N_j*zind; % linear index into DiscountedEV, which is [N_a,1,N_j,N_z]
entireRHS=ReturnMatrix+DiscountedEV(aprimejz); % [N_d,2n+1,N_a,N_j,N_z]

%% Two maxima: over aprime per d (the new reference), then over d
[Vd,offset1]=max(entireRHS,[],2);                                  % [N_d,1,N_a,N_j,N_z]
% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(offset1,[N_d,N_a,N_j,N_z])-1;                 % 0 to 2n, steps above loweredge, per d
aprimeReferencePolicyNew=loweredge+coarseoffset;                   % [N_d,N_a,N_j,N_z]

[Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j,N_z]),[],1);             % [1,N_a,N_j,N_z]
V=reshape(Vtemp,[N_a*N_j,N_z]);
dstar=reshape(dstar,[N_a,N_j,N_z]);

% aprime at the chosen d. The lookup must use dstar and not the state index.
aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)+N_d*N_a*N_j*shiftdim((0:1:N_z-1),-1)); % [N_a,N_j,N_z]

%% Output shape for policy, the same as the unrestricted raw
% The folded (d,aprime) index, d fastest, as the unrestricted raw's repelem(DiscountedEV,N_d,...) layout
Policy=shiftdim(dstar+N_d*(aprimestar-1),-1); % [1,N_a,N_j,N_z]

end
