function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_noz_raw(Vnext,n_a, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search, no d, no z, no e: aprime is restricted to a window of 2*nlocalsearch+1 grid points around
% aprimeReferencePolicy. ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_raw without the z dimension; see its header for
% the method, the sliding window and the 0-based coarseoffset. The reference is [1,N_a].
% Vnext is [N_a,1].

N_a=prod(n_a);

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[1,N_a]);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% The window, one per a, sliding at the grid ends (as in every localsearch raw)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n); % [1,N_a]
aprimeindexes=loweredge+(0:1:2*n)';                    % [2n+1,N_a]

% No shocks, so the expectation is just next period value fn
DiscountedEV=DiscountFactorParamsVec*Vnext; % [N_a,1]

%% Return matrix on the window only
ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(aprimeindexes), a_grid, ReturnFnParamsVec); % [2n+1,N_a]

entireRHS=ReturnMatrix+DiscountedEV(aprimeindexes);

[V,maxindex]=max(entireRHS,[],1);
V=reshape(V,[N_a,1]);

aprimeindexstar=loweredge+maxindex-1; % [1,N_a]

%% The new reference, the SAME object as the input: [1,N_a]
aprimeReferencePolicyNew=aprimeindexstar;

%% Policy in transition paths
Policy=reshape(ind2sub_vec_homemade(n_a,aprimeindexstar(:))',[length(n_a),N_a]);

end
