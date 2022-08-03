function PricePath=TransitionPath_Case1_EndoType(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions)

if transpathoptions.GEnewprice~=2
    if transpathoptions.parallel==2
        N_d=prod(n_d);
        if N_d==0
            % Not yet implemented
            error('TransitionPath_Case1 with vfoptions.endotype is not yet implemented')
%             PricePath=TransitionPath_Case1_shooting_no_d(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
        else
            PricePath=TransitionPath_Case1_shooting_endotype(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
        end
    else
        error('VFI Toolkit does not offer par1 for transition path. Would be too slow to be useful.')
    end
end

if transpathoptions.GEnewprice==2
    warning('Have not yet implemented transpathoptions.GEnewprice==2 for infinite horizon transition paths (2 is to treat path as a fixed-point problem) ')
end

end