function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_nod_e_raw(V,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J,e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, fastOLG (all ages at once), with z and e, no d.
% fastOLG is done as (a,j,z,e): V is (a,j)-by-z-by-e, Policy is (1,a,j,z,e).
% This is EXACTLY slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_raw, with every age in
% one kernel and e added: aprime is restricted to a window of 2*nlocalsearch+1 grid points around
% aprimeReferencePolicy, the window SLIDES at the grid ends, and coarseoffset is 0-BASED. See that raw
% for the reasoning.
%
% THE REFERENCE is [1,N_a,N_j,N_z,N_e], one window per (a,j,z,e).
% j=N_j: EV is zero there via the zero slice appended to pi_z_J, exactly as in the unrestricted raw
% ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_e_raw, whose EV lines are copied unchanged.
% ONLY THE EVpre=0 PATH, which is all that unrestricted raw has: local search is only reachable from a
% transition path, where EVpre=0.

N_a=prod(n_a);
N_z=prod(n_z);
pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[1,N_a,N_j,N_z,N_e]);

z_gridvals_J=shiftdim(z_gridvals_J,-2); % [1,1,N_j,N_z,l_z]
e_gridvals_J=reshape(e_gridvals_J,[1,1,N_j,1,N_e,length(n_e)]); % [1,1,N_j,1,N_e,l_e]

jind=shiftdim(gpuArray(0:1:N_j-1),-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-2);

%% First, create the big 'next period (of transition path) expected value fn.
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a

DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV; % [N_a,1,N_j,N_z]

%% The window, one per (a,j,z,e)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [1,N_a,N_j,N_z,N_e]
aprimeindexes=loweredge+(0:1:2*n)';                     % [2n+1,N_a,N_j,N_z,N_e]

%% Return matrix on the window only
% Level 2 takes aprime as [N_aprime,N_a,N_j,N_z,N_e] and returns that shape
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, n_e, N_j, a_grid(aprimeindexes), a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);

aprimejz=aprimeindexes+N_a*jind+N_a*N_j*zind; % linear index into DiscountedEV, which is [N_a,1,N_j,N_z]; e autofills
entireRHS=ReturnMatrix+DiscountedEV(aprimejz); % [2n+1,N_a,N_j,N_z,N_e]

[Vtemp,maxindex]=max(entireRHS,[],1);
V=reshape(Vtemp,[N_a*N_j,N_z,N_e]);

% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=maxindex-1; % [1,N_a,N_j,N_z,N_e], 0 to 2n, steps above loweredge

%% Output shape for policy, the same as the unrestricted raw: [1,N_a,N_j,N_z,N_e]
Policy=loweredge+coarseoffset;
% With no d the new reference is just the chosen aprime. Returned all the same, so that a caller
% never has to branch on whether the model has a d.
aprimeReferencePolicyNew=Policy;

end
