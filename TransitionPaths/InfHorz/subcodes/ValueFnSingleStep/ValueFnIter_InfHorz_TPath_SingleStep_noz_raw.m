function [V,Policy2]=ValueFnIter_InfHorz_TPath_SingleStep_noz_raw(Vnext,n_d,n_a, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, with d, no z, no e.
% Vnext is [N_a,1]. No exogenous state to loop over, so vfoptions.lowmemory is ignored (everything at once).

N_d=prod(n_d);
N_a=prod(n_a);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% No shocks, so the expectation is just next period value fn
EV=Vnext; % [N_a,1], aprime-by-1
DiscountedEV=DiscountFactorParamsVec*EV;

ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0); % [N_d*N_a,N_a], (d,aprime)-by-a

entireRHS=ReturnMatrix+repelem(DiscountedEV,N_d,1); % (d,aprime)-by-a (DiscountedEV autoexpands over a)

% Calc the max and it's index
[V,Policy]=max(entireRHS,[],1);
V=reshape(V,[N_a,1]); % otherwise it was outputting as [1,N_a]

%% Policy in transition paths
Policy2=reshape(ind2sub_vec_homemade([n_d,n_a],Policy(:))',[length(n_d)+length(n_a),N_a]);

end
