function AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions, EntryExitParamNames, PolicyWhenExiting)
% vfoptions or simoptions can be used as the input
%
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
%
% EntryExitParamNames and PolicyWhenExiting are optional inputs, only needed when using endogenous entry and endogenous exit.

%%
if ~isfield(simoptions,'parallel')
    simoptions.parallel=1+(gpuDeviceCount>0);
end
if ~isfield(simoptions, 'alreadygridvals')
    simoptions.alreadygridvals=0;
end
if ~isfield(simoptions, 'gridinterplayer')
    simoptions.gridinterplayer=0;
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);


if N_d==0 && isscalar(n_a) && simoptions.gridinterplayer==0
    l_daprime=1;
else
    l_daprime=size(Policy,1);
    if simoptions.gridinterplayer==1
        l_daprime=l_daprime-1;
    end
end
a_gridvals=CreateGridvals(n_a,a_grid,1);
% Switch to z_gridvals
if simoptions.alreadygridvals==0
    if simoptions.parallel<2
        z_gridvals=z_grid; % On cpu, only basics are allowed. No e.
    else
        [z_gridvals, ~, simoptions]=ExogShockSetup(n_z,z_grid,[],Parameters,simoptions,1);
    end
elseif simoptions.alreadygridvals==1
    z_gridvals=z_grid;
end

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluate_copy=FnsToEvaluate; % keep a copy in case needed for conditional restrictions
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        FnsToEvalNames=simoptions.AggVarNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end


%% Check for functions that use value function and send these off to a subversion called EvalFnOnAgentDist_AggVars_Case1_withV()
if isfield(simoptions,'eval_valuefn')
    AggVarsExtra=struct();
    for ff=1:length(FnsToEvalNames)
        if ~isempty(FnsToEvaluateParamNames(ff).Names)
            if strcmp(FnsToEvaluateParamNames(ff).Names{1},simoptions.eval_valuefnname{1})
                % This function to evaluate has value function as an input
                % Send it to subfunction, and remove it from FnsToEvaluate
                tempFnsToEvaluateParamNames.Names={};
                if length(FnsToEvaluateParamNames(ff).Names)>1
                    tempFnsToEvaluateParamNames(ff).Names=FnsToEvaluateParamNames(ff).Names{2:end};
                end
                AggVarsExtra=EvalFnOnAgentDist_AggVars_Case1_withV(simoptions.eval_valuefn,StationaryDist, Policy, {FnsToEvaluate{ff}}, {FnsToEvalNames{ff}}, Parameters, tempFnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions.parallel, simoptions);
                FnsToEvaluate2=FnsToEvaluate; FnsToEvaluateParamNames2=FnsToEvaluateParamNames;
                clear FnsToEvaluateParamNames
                FnsToEvaluate={}; % Note: there may be no other FnsToEvaluate
                for gg=1:length(FnsToEvalNames)
                    if gg<ff
                        FnsToEvaluateParamNames(gg).Names=FnsToEvaluateParamNames2(gg).Names;
                        FnsToEvaluate{gg}=FnsToEvaluate2{gg};
                    elseif gg>ff
                        FnsToEvaluateParamNames(gg-1).Names=FnsToEvaluateParamNames2(gg).Names;
                        FnsToEvaluate{gg-1}=FnsToEvaluate2{gg};
                    end
                end
            end
        end
    end
    if isempty(FnsToEvaluate) % There are no FnsToEvaluate that do not have a dependence on the value function
        AggVars=AggVarsExtra;
        return
    end
end



%% Deal with Entry and/or Exit if approprate
if isstruct(StationaryDist)
    % Note: if you want the agent mass of the stationary distribution you have to call it 'agentmass'
    if ~isfield(simoptions,'endogenousexit')
        simoptions.endogenousexit=0;
    end
    if simoptions.endogenousexit~=2
        AggVars=EvalFnOnAgentDist_AggVars_InfHorz_Mass(StationaryDist.pdf,StationaryDist.mass, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions.parallel, simoptions);
    elseif simoptions.endogenousexit==2
        exitprobabilities=CreateVectorFromParams(Parameters, simoptions.exitprobabilities);
        exitprobs=[1-sum(exitprobabilities),exitprobabilities];
        AggVars=EvalFnOnAgentDist_AggVars_InfHorz_Mass_MixExit(StationaryDist.pdf,StationaryDist.mass, Policy, PolicyWhenExiting, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions.parallel, exitprobs);
    end
    
    if FnsToEvaluateStruct==1
        % Change the output into a structure
        AggVars2=AggVars;
        clear AggVars
        AggVars=struct();
        %     AggVarNames=fieldnames(FnsToEvaluate);
        for ff=1:length(FnsToEvalNames)
            AggVars.(FnsToEvalNames{ff}).Aggregate=AggVars2(ff);
        end
    end
    
    return
end


%%
if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist);
    Policy=gpuArray(Policy);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    a_gridvals=gpuArray(a_gridvals);
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid,simoptions);
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]
    
    for ff=1:length(FnsToEvaluate)
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);        
        Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
        Values=reshape(Values,[N_a*N_z,1]);
        % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
        temp=Values.*StationaryDistVec;
        AggVars(ff)=sum(temp(~isnan(temp)));
    end
    
else % CPU
    
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(Policy,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=num2cell(a_gridvals);
    z_gridvals=num2cell(z_gridvals);

    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    StationaryDistVec=gather(StationaryDistVec);
    
    AggVars=zeros(length(FnsToEvaluate),1);
    
    if l_d>0
        
        for ff=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(ff)=sum(temp(~isnan(temp)));
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(ff)=sum(temp(~isnan(temp)));
            end
        end
    
    else % l_d=0
        
        for ff=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(ff)=sum(temp(~isnan(temp)));
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                AggVars(ff)=sum(temp(~isnan(temp)));
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

if isfield(simoptions,'eval_valuefn')
    % There are some thing in AggVarsExtra that depended on the value function, put them into AggVars
    extraNames=fieldnames(AggVarsExtra);
    for ff=1:length(extraNames)
        AggVars.(extraNames{ff})=AggVarsExtra.(extraNames{ff});
    end
end



end
