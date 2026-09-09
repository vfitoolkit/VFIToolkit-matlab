function [p_eqm_vec,GeneralEqmConditions]=StationaryGeneralEqm_subcode_fminalgo5_PType(GeneralEqmConditionsFnOpt,GEparamsvec0,Parameters,GeneralEqmEqns,GEPriceParamNames,GEpriceindexesB,heteroagentoptions,N_i,GEpriceindexes)

% Update based on rules in heteroagentoptions.fminalgo5.howtoupdate
% Set up the howtoupdate rules in the format needed
heteroagentoptions=setupGEnewprice3_shooting(heteroagentoptions,GeneralEqmEqns,GEPriceParamNames,N_i,GEpriceindexes');

% Get initial prices, p
p_old_unconstrained=GEparamsvec0'; % row orientation, to match the add/factor/keepold rule vectors (rows)
[p_old,~]=ParameterConstraints_TransformParamsToOriginal(p_old_unconstrained,GEpriceindexesB,GEPriceParamNames,heteroagentoptions);

% Given current prices solve the model to get the general equilibrium conditions as a structure
itercounter=1;
p_change=Inf;
GeneralEqmConditions=Inf;
while (any(p_change>heteroagentoptions.toleranceGEprices) || any(GeneralEqmConditions>heteroagentoptions.toleranceGEcondns)) && itercounter<heteroagentoptions.maxiter

    % Note: need the unconstrained as input here
    p_i=GeneralEqmConditionsFnOpt(p_old_unconstrained); % using heteroagentoptions.outputGEform=1, so this is a vector (note the transpose)
    % p_i contains the GE eqn values; these are computed from the model and so are already in terms of the original (constrained) parameters

    GeneralEqmConditionsVec=p_i; % Need later to look at convergence

    % Update prices based on GEstruct following the howtoupdate rules
    p_i=p_i(heteroagentoptions.fminalgo5.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
    I_makescutoff=(abs(p_i)>heteroagentoptions.updateaccuracycutoff);
    p_i=I_makescutoff.*p_i;

    % The additional-factor ramp (howtoupdate columns 5, 6 and 7, set via
    % heteroagentoptions.fminalgo5.additionalfactor=[f_add,t1_add,t2_add]). rampweight is 0 up to
    % iteration t1_add and 1 from iteration t2_add on, linear in between, so factor_iter is factor up
    % to t1_add, runs to f_add*factor by t2_add, and stays there. Written as 1+(f_add-1)*w rather
    % than min(...,f_add) so that f_add<1, damping the step down over iterations, works as well as
    % f_add>1. t2_add>t1_add is enforced in setupGEnewprice3_shooting, so the denominator is never
    % zero. itercounter is 1 on the first pass, the same as on a transition path.
    % Same arithmetic as in updatePricePathNew_TPath_tt and the non-PType fminalgo5 subcode.
    rampweight=min(max((itercounter-heteroagentoptions.fminalgo5.t1_add)./(heteroagentoptions.fminalgo5.t2_add-heteroagentoptions.fminalgo5.t1_add),0),1);
    factor_iter=heteroagentoptions.fminalgo5.factor.*(1+(heteroagentoptions.fminalgo5.f_add-1).*rampweight);
    p_new=(p_old.*heteroagentoptions.fminalgo5.keepold)+heteroagentoptions.fminalgo5.add.*factor_iter.*p_i-(1-heteroagentoptions.fminalgo5.add).*factor_iter.*p_i;

    % Calculate GeneralEqmConditions which measures convergence
    if heteroagentoptions.multiGEcriterion==0
        GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
    elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market
        GeneralEqmConditions=sqrt(sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2)));
    end

    % Put new prices into Parameters
    for ii=1:length(GEPriceParamNames)
        Parameters.(GEPriceParamNames{ii})=p_new(GEpriceindexes(ii,1):GEpriceindexes(ii,2)); % ptype-dependent prices span multiple indexes
    end

    p_change=abs(p_new-p_old); % note, this is a vector
    % p_percentchange=max(abs(p_new-p)./abs(p));
    % p_percentchange(p==0)=abs(p_new(p==0)); %-p(p==0)); but this is just zero anyway

    % Update p for next iteration
    p_old=p_new;
    p_old_unconstrained=ParameterConstraints_TransformParamsToUnconstrained(p_old,GEpriceindexesB,GEPriceParamNames,heteroagentoptions,0); % final input 0 as constraints are already in vector form
    itercounter=itercounter+1; % increment iteration counter
end

if itercounter>heteroagentoptions.maxiter
    warning('HeteroAgentStationaryEqm stopped due to reaching maximum number of iterations (you can control using heteroagentoptions.maxiter)')
end

p_eqm_vec=p_old_unconstrained; % Output (will be untransformed later, note this equal to p_new, just with the transform)

end
