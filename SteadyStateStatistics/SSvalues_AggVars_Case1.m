function SSvalues_AggVars=SSvalues_AggVars_Case1(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters,SSvalueParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p_val, Parallel)
% Evaluates the aggregate value (weighted sum/integral) for each element of SSvaluesFn

SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames);

% Have used this setup as for some functions in the toolkit it is more
% convenient if they can call the vector version directly.
SSvalues_AggVars=SSvalues_AggVars_Case1_vec(StationaryDist, PolicyIndexes, SSvaluesFn, SSvalueParamsVec, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p_val, Parallel);

end