function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, finite horizon, fastOLG (all ages at once), no d, no z. The V input is next period value
% fn (across all ages), the V output is this period.
% The fastOLG version of slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_raw: aprime is
% restricted to a window of 2*nlocalsearch+1 grid points around aprimeReferencePolicy, the window SLIDES
% at the grid ends, and coarseoffset is 0-BASED. See that raw (and the InfHorz original) for the reasoning.
%
% THE REFERENCE is [1,N_a,N_j], one window per (a,j), the same trailing dims as Policy.
% Policy comes out as the unrestricted raw's, [1,N_a,N_j], the aprime index.
%
% EV is built EXACTLY as ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_noz_raw does: the j=N_j slice is
% left as zeros, so the terminal age needs no special case. That raw has no EVpre branch; local search is
% only reachable from a transition path, where EVpre=0 anyway.

N_a=prod(n_a);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_a,N_j]);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=V(:,2:end); % Leave N_j as zeros

%% The window, for every (a,j) at once
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [N_a,N_j]
aprimeindexes=reshape(loweredge,[1,N_a,N_j])+(0:1:2*n)'; % [2n+1,N_a,N_j]

%% Return matrix on the window only
% Level 2 of the nod creator takes aprime as [N_aprime,N_a,N_j] and returns that shape
ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn, N_j, a_grid(aprimeindexes), a_grid, ReturnFnParamsAgeMatrix,2);

aprimej=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-1); % linear index into EV, which is [N_a,1,N_j]
% beta_j*EV(aprime,j), the same products as the unrestricted raw's reshape(DiscountFactor_J,[1,1,N_j]).*EV
entireRHS=ReturnMatrix+reshape(DiscountFactor_J,[1,1,N_j]).*EV(aprimej); % [2n+1,N_a,N_j]

[Vtemp,maxindex]=max(entireRHS,[],1);
V=reshape(Vtemp,[N_a,N_j]);

% COARSEOFFSET IS 0-BASED, in every localsearch raw: the grid index is loweredge+coarseoffset
coarseoffset=reshape(maxindex,[N_a,N_j])-1; % 0 to 2n, steps above loweredge
aprimeindexstar=loweredge+coarseoffset;

%% Output shape for policy, the same as the unrestricted raw
Policy=reshape(aprimeindexstar,[1,N_a,N_j]);
% With no d the new reference is just the chosen aprime
aprimeReferencePolicyNew=reshape(aprimeindexstar,[1,N_a,N_j]);

end
