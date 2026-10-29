function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames)
% THE SEED SWEEP, finite horizon, fastOLG (all ages at once), with a d, no z, without the grid
% interpolation layer. An UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d,
% [N_d,N_a,N_j], which is what vfoptions.localsearch needs as its reference.
% The fastOLG version of slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw; see it for
% why this exists.
%
% It has to agree with ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_raw: same creator, same search
% set, same arithmetic, so V is bit-identical. That raw takes one max over the folded (d,aprime); this
% reshapes the SAME folded return matrix to [N_d,N_a,N_a,N_j] (the fold is d fastest, so this is exact)
% and maxes over aprime first, then over d. TIE-BREAKING IS THE ONE DIFFERENCE.
%
% The input list is EXACTLY the unrestricted raw's, which takes no vfoptions: that raw has no lowmemory
% branch and no EVpre branch, so there is nothing to honour here. Its EV leaves the j=N_j slice as zeros.

N_d=prod(n_d);
N_a=prod(n_a);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

EV=zeros(N_a,1,N_j,'gpuArray');
EV(:,1,1:N_j-1)=V(:,2:end);

% The unrestricted raw repelems this over d; here it broadcasts over d in dim 1 instead, same values
DiscountedEV=shiftdim(reshape(DiscountFactor_J,[1,1,N_j]).*EV,-1); % [1,N_aprime,1,N_j]

ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_noz(ReturnFn, n_d, n_a, N_j, d_gridvals, a_grid, a_grid, ReturnFnParamsAgeMatrix);
% (d,aprime) folded with d fastest, so unfold it
ReturnMatrix=reshape(ReturnMatrix,[N_d,N_a,N_a,N_j]); % (d,aprime,a,j)

entireRHS=ReturnMatrix+DiscountedEV;

% Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
[Vd,offset1]=max(entireRHS,[],2);                   % [N_d,1,N_a,N_j]
% No window here, so the argmax is already a grid index in [1,N_a]
aprimeReferencePolicyNew=reshape(offset1,[N_d,N_a,N_j]);

[Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j]),[],1);  % [1,N_a,N_j]
V=reshape(Vtemp,[N_a,N_j]);
dstar=reshape(dstar,[N_a,N_j]);

% aprime at the chosen d. The lookup must use dstar and not the state index.
aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)); % [N_a,N_j]

%% Output shape for policy, the same as the folded raw
% The folded (d,aprime) index, d fastest
Policy=reshape(dstar+N_d*(aprimestar-1),[1,N_a,N_j]);

end
