function [p_eqm_vec,GeneralEqmConditions]=StationaryGeneralEqm_subcode_fminalgo5(GeneralEqmConditionsFnOpt,GEparamsvec0,Parameters,GeneralEqmEqns,GEPriceParamNames,nGEParams,heteroagentoptions)

% Update based on rules in heteroagentoptions.fminalgo5.howtoupdate
% Set up the howtoupdate rules in the format needed
heteroagentoptions=setupGEnewprice3_shooting(heteroagentoptions,GeneralEqmEqns,GEPriceParamNames);

% Get initial prices, p
p_old_unconstrained=GEparamsvec0;
[p_old,~]=ParameterConstraints_TransformParamsToOriginal(p_old_unconstrained,0:1:nGEParams,GEPriceParamNames,heteroagentoptions);

% Given current prices solve the model to get the general equilibrium conditions as a structure
itercounter=0;
p_change=Inf;
GeneralEqmConditions=Inf;
while (any(p_change>heteroagentoptions.toleranceGEprices) || GeneralEqmConditions>heteroagentoptions.toleranceGEcondns) && itercounter<heteroagentoptions.maxiter


    % Note: need the unconstrained as input here
    p_i=GeneralEqmConditionsFnOpt(p_old_unconstrained); % using heteroagentoptions.outputGEform=1, so this is a vector (note the transpose)
    % p_i contains the GE eqn values; these are computed from the model and so are already in terms of the original (constrained) parameters

    GeneralEqmConditionsVec=p_i; % Need later to look at convergence

    % Update prices based on GEstruct following the howtoupdate rules
    p_i=p_i(heteroagentoptions.fminalgo5.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
    I_makescutoff=(abs(p_i)>heteroagentoptions.updateaccuracycutoff);
    p_i=I_makescutoff.*p_i;

    p_new=(p_old.*heteroagentoptions.fminalgo5.keepold)+heteroagentoptions.fminalgo5.add.*heteroagentoptions.fminalgo5.factor.*p_i-(1-heteroagentoptions.fminalgo5.add).*heteroagentoptions.fminalgo5.factor.*p_i;

    % Calculate GeneralEqmConditions which measures convergence
    if heteroagentoptions.multiGEcriterion==0
        GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
    elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market
        GeneralEqmConditions=sqrt(sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2)));
    end

    % Put new prices into Parameters
    for ii=1:length(GEPriceParamNames)
        Parameters.(GEPriceParamNames{ii})=p_new(ii);
    end

    p_change=abs(p_new-p_old); % note, this is a vector
    % p_percentchange=max(abs(p_new-p)./abs(p));
    % p_percentchange(p==0)=abs(p_new(p==0)); %-p(p==0)); but this is just zero anyway

    % Update p for next iteration
    p_old=p_new;
    p_old_unconstrained=ParameterConstraints_TransformParamsToUnconstrained(p_old,0:1:nGEParams,GEPriceParamNames,heteroagentoptions,0); % final input 0 as constraints are already in vector form
    itercounter=itercounter+1; % increment iteration counter
end

if itercounter>=heteroagentoptions.maxiter
    warning('HeteroAgentStationaryEqm stopped due to reaching maximum number of iterations (you can control using heteroagentoptions.maxiter)')
end

p_eqm_vec=p_old_unconstrained; % Output (will be untransformed later, note this equal to p_new, just with the transform)

end