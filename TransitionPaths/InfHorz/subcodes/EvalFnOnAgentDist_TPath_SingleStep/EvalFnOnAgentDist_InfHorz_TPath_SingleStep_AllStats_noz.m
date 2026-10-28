function AllStats=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AllStats_noz(AgentDist, PolicyValues, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, FnsToEvaluateNames, n_a, a_gridvals, simoptions)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
% No z (and no e): AgentDist is [N_a,1], PolicyValues is [l_d+l_aprime,N_a]
% For internal use only

N_a=prod(n_a);

l_daprime=size(PolicyValues,1);

AgentDist=AgentDist(:); % Needs to be a vector
PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a]),[2,1]); %[N_a,l_d+l_a]

AllStats=struct();

%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it).
if isfield(simoptions,'conditionalrestrictions')

    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(length(CondlRestnFnNames),1);
    RestrictionStruct=struct();

    l_a=length(n_a);

    % For each conditional restriction, create a 'restricted stationary distribution'
    for rr=1:length(CondlRestnFnNames)
        CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
        % Get parameter names for Conditional Restriction functions
        temp=getAnonymousFnInputNames(CondlRestnFn);
        if length(temp)>(l_daprime+l_a)
            CondlRestnFnParamNames={temp{l_daprime+l_a+1:end}}; % the first inputs will always be (d,aprime,a)
        else
            CondlRestnFnParamNames={};
        end
        CondlRestnFnParamsCell=CreateCellFromParams(Parameters,CondlRestnFnParamNames);

        RestrictionValues=logical(EvalFnOnAgentDist_Grid(CondlRestnFn, CondlRestnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]));
        RestrictionValues=reshape(RestrictionValues,[N_a,1]);

        RestrictedAgentDist=AgentDist;
        RestrictedAgentDist(~RestrictionValues)=0; % zero mass on all points that do not meet the restriction

        % Need to keep two things, the restrictedsamplemass and the RestrictedStationaryDistVec (normalized to have mass of 1)
        restrictedsamplemass(rr)=sum(RestrictedAgentDist);
        RestrictionStruct(rr).RestrictedAgentDist=RestrictedAgentDist/restrictedsamplemass(rr);

        AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr); % Seems likely this would be something user might want
    end
end


%%

for ff=1:length(FnsToEvaluateNames)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
    Values=EvalFnOnAgentDist_Grid(FnsToEvaluateCell{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]);
    Values=reshape(Values,[N_a,1]);

    AllStats.(FnsToEvaluateNames{ff})=StatsFromWeightedGrid(Values,AgentDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

    %% If there are any conditional restrictions then deal with these
    % Evaluate AllStats, but conditional on the restriction being one.
    if useCondlRest==1
        % Evaluate the conditional restrictions:
        % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
        for rr=1:length(CondlRestnFnNames)
            if restrictedsamplemass(rr)>0
                AllStats.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff})=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedAgentDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
            end
        end
    end

end


end
