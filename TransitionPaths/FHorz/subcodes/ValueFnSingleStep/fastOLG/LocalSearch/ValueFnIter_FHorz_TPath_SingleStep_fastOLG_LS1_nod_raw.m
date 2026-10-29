function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, fastOLG (all ages at once), with z, no d.
% fastOLG is done as (a,j,z): V is (a,j)-by-z, Policy is (1,a,j,z).
% This is EXACTLY slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_raw, with every age in
% one kernel instead of a loop over j: aprime is restricted to a window of 2*nlocalsearch+1 grid points
% around aprimeReferencePolicy, the window SLIDES at the grid ends, and coarseoffset is 0-BASED. See that
% raw (and the InfHorz original it points to) for the reasoning.
%
% THE REFERENCE is [1,N_a,N_j,N_z], the same (a,j,z) order as Policy. One window per (a,j,z).
%
% THE TERMINAL AGE is handled as in the unrestricted fastOLG raw: a zero slice is appended to pi_z_J,
% so EV is zero at j=N_j and R+0 is R exactly. At the cap (2n+1=N_a) this is bit-identical to
% ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_raw.

N_a=prod(n_a);
N_z=prod(n_z);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs.

% ONLY THE vfoptions.EVpre==0 PATH. Local search is reachable only from a transition path, where
% EVpre=0 (EVpre=1 is the Matched Expectations Path, which never sets localsearch).
if vfoptions.EVpre~=0
    error('vfoptions.localsearch requires vfoptions.EVpre=0 (ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_nod_raw)')
end

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_j,N_z]);

z_gridvals_J=shiftdim(z_gridvals_J,-2); % [1,1,N_j,N_z,l_z]

jind=shiftdim(gpuArray(0:1:N_j-1),-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-2);

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

%% The window, for every (a,j,z) at once
% Same line as every other localsearch raw. Requires N_a>=2n+1, which is checked where nlocalsearch
% is validated.
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_j,N_z]
aprimeindexes=shiftdim(loweredge,-1)+(0:1:2*n)';       % [2n+1,N_a,N_j,N_z]

%% Return matrix on the window only
% Level 2 of the no-d creator is [N_aprime,N_a,N_j,N_z] with N_aprime=size(aprime_grid,1), unfolded
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_z, N_j, a_grid(aprimeindexes), a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);

aprimejz=aprimeindexes+N_a*jind+N_a*N_j*zind; % linear index into DiscountedEV, which is [N_a,1,N_j,N_z]
entireRHS=ReturnMatrix+DiscountedEV(aprimejz); % [2n+1,N_a,N_j,N_z]

[Vtemp,maxindex]=max(entireRHS,[],1);
V=reshape(Vtemp,[N_a*N_j,N_z]);

% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(maxindex,[N_a,N_j,N_z])-1; % 0 to 2n, steps above loweredge
aprimeindexstar=loweredge+coarseoffset;

% With no d the new reference is just the chosen aprime. Returned all the same, so that a caller
% never has to branch on whether the model has a d.
aprimeReferencePolicyNew=shiftdim(aprimeindexstar,-1); % [1,N_a,N_j,N_z]

%% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(aprimeindexstar,-1); % [1,N_a,N_j,N_z]

end
