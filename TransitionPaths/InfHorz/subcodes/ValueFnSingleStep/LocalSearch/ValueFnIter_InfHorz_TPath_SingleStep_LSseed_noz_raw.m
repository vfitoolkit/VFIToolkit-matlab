function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LSseed_noz_raw(Vnext,n_d,n_a, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, with a d, no z, no e, without the grid interpolation layer. An UNRESTRICTED sweep that also returns
% the aprime argmax FOR EACH d, [N_d,N_a], which is what vfoptions.localsearch needs as its reference.
% ValueFnIter_InfHorz_TPath_SingleStep_LSseed_raw without the z dimension; see its header for why this exists, why it
% uses the UNFOLDED return matrix, and the one difference from the folded raw (tie-breaking).
% It has to agree with ValueFnIter_InfHorz_TPath_SingleStep_noz_raw. No exogenous state, so no lowmemory branch (as
% in that raw).
% Vnext is [N_a,1].

N_d=prod(n_d);
N_a=prod(n_a);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% No shocks, so the expectation is just next period value fn; unfolded, (d,aprime,a), it broadcasts over d and a
DiscountedEV=DiscountFactorParamsVec*shiftdim(Vnext,-1); % [1,N_a]

% Level 1 (Refine=1) keeps d and aprime in separate dimensions, (d,aprime,a)
ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,1); % [N_d,N_a,N_a]
entireRHS=ReturnMatrix+DiscountedEV;

% Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
[Vd,offset1]=max(entireRHS,[],2);                    % [N_d,1,N_a]

% The new reference: the aprime argmax per (d,a), over the WHOLE grid, so already a grid index in [1,N_a]
aprimeReferencePolicyNew=reshape(offset1,[N_d,N_a]);

% Then the max over d
[V,dstar]=max(reshape(Vd,[N_d,N_a]),[],1);           % [1,N_a]
V=reshape(V,[N_a,1]);

% aprime at the chosen d. The lookup must use dstar and not the state index, which is the one thing
% to get wrong here.
aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)); % [1,N_a]

%% Policy, in the same form and with the same interpretation as the folded raw's Policy2
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],reshape(dstar+N_d*(aprimestar-1),[],1))',[length(n_d)+length(n_a),N_a]);

end
