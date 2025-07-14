function AggVars=EvalFnOnAgentDist_AggVars_Case1_withV(V,StationaryDist, Policy, FnsToEvaluate, FnsToEvalNames, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, simoptions)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
%
% Parallel, simoptions and EntryExitParamNames are optional inputs, only needed when using endogenous entry

if n_d(1)==0
    l_d=0;
    N_d=0;
else
    l_d=length(n_d);
    N_d=prod(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);


%% Output as structure
FnsToEvaluateStruct=1;

% Note: No SDP nor entry/exit is considered

%%
if Parallel==2 || Parallel==4
    Parallel=2;
    StationaryDist=gpuArray(StationaryDist);
    Policy=gpuArray(Policy);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    z_grid=gpuArray(z_grid);
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names)  % check for 'FnsToEvaluateParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
        end
        Values=EvalFnOnAgentDist_Grid_Case1_withV(V,FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        % When evaluating value function (which may sometimes give -Inf
        % values) on StationaryDistVec (which at those points will be
        % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
        temp=Values.*StationaryDistVec;
        AggVars(i)=sum(temp(~isnan(temp)));
    end
    
else
    
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(Policy,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    
    StationaryDistVec=gather(StationaryDistVec);
    
    AggVars=zeros(length(FnsToEvaluate),1);
    
    if l_d>0
        
        for i=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},V(j1,j2));
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},V(j1,j2),FnToEvaluateParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            end
        end
    
    else %l_d=0
        
        for i=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},V(j1,j2));
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},V(j1,j2),FnToEvaluateParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            end
        end
    end
    
end

%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    AggVars2=AggVars;
    clear AggVars
    AggVars=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        AggVars.(FnsToEvalNames{ff}).Mean=AggVars2(ff);
    end
end

%% If there are any conditional restrictions then send these off to be done
% Evaluate AggVars, but conditional on the restriction being non-zero.
%
% Code works by evaluating the the restriction and imposing this on the
% distribution (and renormalizing it) and then just sending this off to
% EvalFnOnAgendDist_AggVars_Case1() again. Some of the results are then
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
        for kk=1:length(CondlRestnFnNames)
            % Includes check for cases in which no parameters are actually required
            if isempty(CondlRestnFnParamNames(kk).Names) % check for '={}'
                CondlRestnFnParamsVec=[];
            else
                CondlRestnFnParamsVec=CreateVectorFromParams(Parameters,CondlRestnFnParamNames(kk).Names);
            end
            
            Values=EvalFnOnAgentDist_Grid_Case1(CondlRestnFns{kk}, CondlRestnFnParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel);
            Values=reshape(Values,[N_a*N_z,1]);
            
            RestrictedStationaryDistVec=StationaryDistVec;
            RestrictedStationaryDistVec(Values==0)=0; % Drop all those that don't meet the restriction
            restrictedsamplemass=sum(RestrictedStationaryDistVec);
            RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass; % Normalize to mass one

            if restrictedsamplemass==0
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{kk},' has a restricted sample that is of zero mass \n'])
                AggVars.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Just return this and hopefully it is clear to the user
            else
                AggVars.(CondlRestnFnNames{kk})=EvalFnOnAgentDist_AggVars_Case1_withV(V,RestrictedStationaryDistVec, Policy, FnsToEvaluate, FnsToEvalNames, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,simoptions);
                
                % Create some renormalizations where relevant (just the mean)
                for ii=1:length(FnsToEvaluate) %Note FnsToEvaluate alread created above
                    AggVars.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Total=restrictedsamplemass*AggVars.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Mean;
                end
                AggVars.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Seems likely this would be something user might want
            end
        end
    else % simoptions.parallel~=2
        for kk=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(kk).Names) % check for 'FnsToEvaluateParamNames={}'
                Values=zeros(N_a*N_z,1);
                if l_d==0
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                    end
                else % l_d>0
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                    end
                end
            else
                Values=zeros(N_a*N_z,1);
                if l_d==0
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(kk).Names));
                    Values=zeros(N_a*N_z,1);
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                    end
                else % l_d>0
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(kk).Names));
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                    end
                end
            end
            
            RestrictedStationaryDistVec=StationaryDistVec;
            RestrictedStationaryDistVec(Values==0)=0; % Drop all those that don't meet the restriction
            restrictedsamplemass=sum(RestrictedStationaryDistVec);
            RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass; % Normalize to mass one

            if restrictedsamplemass==0
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{kk},' has a restricted sample that is of zero mass \n'])
                AggVars.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Just return this and hopefully it is clear to the user
            else
                AggVars.(CondlRestnFnNames{kk})=EvalFnOnAgentDist_AggVars_Case1_withV(V,RestrictedStationaryDistVec, Policy, FnsToEvaluate, FnsToEvalNames, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid,Parallel, simoptions);
                
                % Create some renormalizations where relevant (just the mean)
                for ii=1:length(FnsToEvaluate) %Note FnsToEvaluate alread created above
                    AggVars.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Total=restrictedsamplemass*AggVars.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Mean;
                end
                AggVars.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Seems likely this would be something user might want
            end
        end
    end
end



end
