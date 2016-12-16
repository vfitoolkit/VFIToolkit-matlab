function SSvalues_AggVars=SSvalues_AggVars_Case2(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters,SSvalueParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p_val,Parallel)
% Evaluates the aggregate value (weighted sum/integral) for each element of SSvaluesFn


% Includes check for cases in which no parameters are actually required
if (isempty(SSvalueParamNames) || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
    SSvalueParamsVec=[];
else
    SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames);
end

% Have used this setup as for some functions in the toolkit it is more
% convenient if they can call the vector version directly.
SSvalues_AggVars=SSvalues_AggVars_Case2_vec(StationaryDist, PolicyIndexes, SSvaluesFn, SSvalueParamsVec, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p_val, Parallel);

end

