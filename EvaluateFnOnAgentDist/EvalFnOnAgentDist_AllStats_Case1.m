function AllStats=EvalFnOnAgentDist_AllStats_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions)
% Returns a wide variety of statistics
%
% simoptions optional inputs

%%
if ~exist('simoptions','var')
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.npoints=100;
    simoptions.nquantiles=20;
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being non-zero.
else
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100;
    end
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=20;
    end
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    if ~isfield(simoptions,'whichstats')
        simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
    end
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being non-zero.
end


if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

AllStats=struct();

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluate_copy=FnsToEvaluate; % keep a copy in case needed for conditional restrictions
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%% I want to do some things now, so that they can be used in setting up conditional restrictions
StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
if simoptions.parallel==2

    StationaryDistVec=gpuArray(StationaryDistVec);
    PolicyIndexes=gpuArray(PolicyIndexes);

    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
end

%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it). 
if isfield(simoptions,'conditionalrestrictions')
    if simoptions.parallel~=2
        error('simoptions.conditionalrestrictions can only be used with GPU')
    end
    
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(length(CondlRestnFnNames),1);
    RestrictionStruct=struct();

    % For each conditional restriction, create a 'restricted stationary distribution'
    for rr=1:length(CondlRestnFnNames)
        % Get parameter names for Conditional Restriction functions
        temp=getAnonymousFnInputNames(simoptions.conditionalrestrictions.(CondlRestnFnNames{rr}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            CondlRestnFnParamNames={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            CondlRestnFnParamNames={};
        end
        % Get parameter values for Conditional Restriction functions
        if isempty(CondlRestnFnParamNames) % check for '={}'
            CondlRestnFnParamsVec=[];
        else
            CondlRestnFnParamsVec=CreateVectorFromParams(Parameters,CondlRestnFnParamNames);
        end
        % Store the actual functions
        CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});

        RestrictionValues=logical(EvalFnOnAgentDist_Grid_Case1(CondlRestnFn, CondlRestnFnParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel));
        RestrictionValues=reshape(RestrictionValues,[N_a*N_z,1]);

        RestrictedStationaryDistVec=StationaryDistVec;
        RestrictedStationaryDistVec(~RestrictionValues)=0; % zero mass on all points that do not meet the restriction
        
        % Need to keep two things, the restrictedsamplemass and the RestrictedStationaryDistVec (normalized to have mass of 1)
        restrictedsamplemass(rr)=sum(StationaryDistVec(RestrictionValues));
        RestrictionStruct(rr).RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass(rr);

        if restrictedsamplemass(rr)==0
            warning('One of the conditional restrictions evaluates to a zero mass')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr); % Just return this and hopefully it is clear to the user
        else
            AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr); % Seems likely this would be something user might want
        end

    end
end

%%

if simoptions.parallel==2

    for ff=1:length(FnsToEvalNames)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
        end
        
        Values=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{ff}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel);
        Values=reshape(Values,[N_a*N_z,1]);

        AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDistVec,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

        %% If there are any conditional restrictions then deal with these
        % Evaluate AllStats, but conditional on the restriction being one.
        if useCondlRest==1
            % Evaluate the conditinal restrictions: 
            % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
            for rr=1:length(CondlRestnFnNames)
                if restrictedsamplemass(rr)>0
                    AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedStationaryDistVec,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
                end
            end
        end

    end

else 
    %% On CPU (simoptions.parallel~=2)
    StationaryDistVec=gather(StationaryDistVec);
    PolicyIndexes=gather(PolicyIndexes);

    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    for ff=1:length(FnsToEvalNames)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'FnsToEvaluateParamNames={}'
            Values=zeros(N_a*N_z,1);
            if l_d==0
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
            else % l_d>0
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
            end
        else
            Values=zeros(N_a*N_z,1);
            if l_d==0
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
            else % l_d>0
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
            end
        end
                
        AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDistVec,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

    end
end



end