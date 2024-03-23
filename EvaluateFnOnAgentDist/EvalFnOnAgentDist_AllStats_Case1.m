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

%%
StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

if simoptions.parallel==2
    StationaryDistVec=gpuArray(StationaryDistVec);
    PolicyIndexes=gpuArray(PolicyIndexes);
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
        
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
    end
    
else
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

%% If there are any conditional restrictions then send these off to be done
% Evaluate AllStats, but conditional on the restriction being non-zero.
%
% Code works by evaluating the the restriction and imposing this on the
% distribution (and renormalizing it) and then just sending this off to
% EvalFnOnAgendDist_AllStats_Case1() again. Some of the results are then
% modified so that there is both, e.g., 'mean' and 'total'.
if isfield(simoptions,'conditionalrestrictions')
    % First couple of lines get the conditional restrictions and convert
    % them to a names and cell
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);
    for ff=1:length(CondlRestnFnNames)
        temp=getAnonymousFnInputNames(simoptions.conditionalrestrictions.(CondlRestnFnNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            CondlRestnFnParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            CondlRestnFnParamNames(ff).Names={};
        end
        CondlRestnFns{ff}=simoptions.conditionalrestrictions.(CondlRestnFnNames{ff});
    end
    simoptions=rmfield(simoptions,'conditionalrestrictions'); % Have to delete this before resend it to EvalFnOnAgentDist_AllStats_Case1()
    
    % Note that some things have already been created above, so we don't need
    % to recreate them to evaluated the restrictions.
    
    if simoptions.parallel==2
        % Evaluate the conditinal restrictions
        for ff=1:length(CondlRestnFnNames)
            % Includes check for cases in which no parameters are actually required
            if isempty(CondlRestnFnParamNames(ff).Names) % check for '={}'
                CondlRestnFnParamsVec=[];
            else
                CondlRestnFnParamsVec=CreateVectorFromParams(Parameters,CondlRestnFnParamNames(ff).Names);
            end
            
            Values=EvalFnOnAgentDist_Grid_Case1(CondlRestnFns{ff}, CondlRestnFnParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel);
            Values=reshape(Values,[N_a*N_z,1]);

            RestrictedStationaryDistVec=StationaryDistVec;
            RestrictedStationaryDistVec(Values==0)=0; % Drop all those that don't meet the restriction
            restrictedsamplemass=sum(RestrictedStationaryDistVec);
            RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass; % Normalize to mass one

            if restrictedsamplemass==0
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{ff},' has a restricted sample that is of zero mass \n'])
                AllStats.(CondlRestnFnNames{ff}).RestrictedSampleMass=restrictedsamplemass; % Just return this and hopefully it is clear to the user
            else
                AllStats.(CondlRestnFnNames{ff})=EvalFnOnAgentDist_AllStats_Case1(RestrictedStationaryDistVec, PolicyIndexes, FnsToEvaluate_copy, Parameters, [], n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions);
                
                % Create some renormalizations where relevant (just the mean)
                for ii=1:length(FnsToEvaluate) %Note FnsToEvaluate alread created above
                    AllStats.(CondlRestnFnNames{ff}).(FnsToEvalNames{ii}).Total=restrictedsamplemass*AllStats.(CondlRestnFnNames{ff}).(FnsToEvalNames{ii}).Mean;
                end
                AllStats.(CondlRestnFnNames{ff}).RestrictedSampleMass=restrictedsamplemass; % Seems likely this would be something user might want
            end
        end
    else % simoptions.parallel~=2
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
            
            RestrictedStationaryDistVec=StationaryDistVec;
            RestrictedStationaryDistVec(Values==0)=0; % Drop all those that don't meet the restriction
            restrictedsamplemass=sum(RestrictedStationaryDistVec);
            RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass; % Normalize to mass one

            if restrictedsamplemass==0
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{ff},' has a restricted sample that is of zero mass \n'])
                AllStats.(CondlRestnFnNames{ff}).RestrictedSampleMass=restrictedsamplemass; % Just return this and hopefully it is clear to the user
            else
                AllStats.(CondlRestnFnNames{ff})=EvalFnOnAgentDist_AllStats_Case1(RestrictedStationaryDistVec, PolicyIndexes, FnsToEvaluate_copy, Parameters, [], n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions);
                
                % Create some renormalizations where relevant (just the mean)
                for ii=1:length(FnsToEvaluate) %Note FnsToEvaluate alread created above
                    AllStats.(CondlRestnFnNames{ff}).(FnsToEvalNames{ii}).Total=restrictedsamplemass*AllStats.(CondlRestnFnNames{ff}).(FnsToEvalNames{ii}).Mean;
                end
                AllStats.(CondlRestnFnNames{ff}).RestrictedSampleMass=restrictedsamplemass; % Seems likely this would be something user might want
            end
        end
    end
end




end