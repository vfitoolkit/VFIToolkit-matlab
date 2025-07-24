function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_InfHorz_Mass(StationaryDistmass, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, simoptions)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
% Internal fn, not intended for use by user.

if ~isfield(simoptions,'endogenousexit')
    simoptions.endogenousexit=0;
else
    if simoptions.endogenousexit==1
        if ~isfield(simoptions,'keeppolicyonexit')
            simoptions.keeppolicyonexit=0;
        end
    end
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

l_daprime=size(Policy,1);
if simoptions.gridinterplayer==1
    l_daprime=l_daprime-1;
end

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%%
if Parallel==2
    StationaryDistmass=gpuArray(StationaryDistmass);
    Policy=gpuArray(Policy);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    l_daprime=size(Policy,1);
    a_gridvals=CreateGridvals(n_a,gpuArray(a_grid),1);
    z_gridvals=CreateGridvals(n_z,gpuArray(z_grid),1);
    
    N_a=prod(n_a);
    N_z=prod(n_z);
        
    % When there is endogenous exit, add exit to the policy (to avoid what
    % would otherwise be zeros) and instead multiply the exiting by the
    % stationary dist to eliminate the 'decisions' there.
    ExitPolicy=zeros(N_a*N_z,1,'gpuArray');
    if simoptions.endogenousexit==1
        if simoptions.keeppolicyonexit==0
            if n_d(1)==0
                l_d=0;
            else
                l_d=length(n_d);
            end
            % Add one to PolicyIndexes
            Policy=Policy+ones(l_d+l_a,1).*(1-shiftdim(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),-1));
            % And use ExitPolicy to later replace these with nan
            ExitPolicy=logical(1-reshape(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),[N_a*N_z,1]));
        end
    end
    RemoveExits=nan(N_a*N_z,1);
    RemoveExits(logical(~ExitPolicy))=1;
    
    ValuesOnGrid=zeros(N_a*N_z,length(FnsToEvaluate),'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid);
    PolicyValues=reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]);
    % permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    % PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    PolicyValuesPermute=permute(PolicyValues,[2,3,1]); %[n_a,n_s,l_d+l_a]
    
    for ff=1:length(FnsToEvaluate)
        if isempty(FnsToEvaluateParamNames(ff).Names)
            FnToEvaluateParamsCell=cell(0);
        else
            if strcmp(FnsToEvaluateParamNames(ff).Names{1},'agentmass')
                if length(FnsToEvaluateParamNames(ff).Names)==1
                    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
                else
                    FnToEvaluateParamsCell=cell(1,length(FnsToEvaluateParamNames(ff).Names));
                    FnToEvaluateParamsCell(1)={StationaryDistmass};
                    FnToEvaluateParamsCell(2:end)=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names(2:end));
                end
            else
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
            end
        end
        Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);        Values=reshape(Values,[N_a*N_z,1]);
        Values=Values.*RemoveExits;
        ValuesOnGrid(:,ff)=Values;
    end
else
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    % When there is endogenous exit, add exit to the policy (to avoid what
    % would otherwise be zeros) and instead multiply the exiting by the
    % stationary dist to eliminate the 'decisions' there.
    ExitPolicy=zeros(N_a*N_z,1);
    if simoptions.endogenousexit==1
        if simoptions.keeppolicyonexit==0
            % Add one to PolicyIndexes
            Policy=Policy+ones(l_d+l_a,1).*(1-shiftdim(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),-1));
            % And use ExitPolicy to later replace these with nan
            ExitPolicy=1-reshape(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),[N_a*N_z,1]);
        end
    end
    RemoveExits=nan(N_a*N_z,1);
    RemoveExits(logical(~ExitPolicy))=1;

    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(Policy,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    ValuesOnGrid=zeros(N_a*N_z,length(FnsToEvaluate));
    
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
                ValuesOnGrid(:,ff)=Values; 
            else
                if strcmp(FnsToEvaluateParamNames(ff).Names{1},'agentmass')
                    if length(FnsToEvaluateParamNames(ff).Names)==1
                        FnToEvaluateParamsVec=StationaryDistmass;
                    else
                        FnToEvaluateParamsVec=[StationaryDistmass,CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names(2:end))];
                    end
                else
                    FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
                end
                FnToEvaluateParamsCell=num2cell(FnToEvaluateParamsVec);               
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                Values=Values.*RemoveExits;
                ValuesOnGrid(:,ff)=Values;
            end
        end
    
    else %l_d=0
        
        for ff=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                ValuesOnGrid(:,ff)=Values;
            else
                if strcmp(FnsToEvaluateParamNames(ff).Names{1},'agentmass')
                    if length(FnsToEvaluateParamNames(ff).Names)==1
                        FnToEvaluateParamsVec=StationaryDistmass;
                    else
                        FnToEvaluateParamsVec=[StationaryDistmass,CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names(2:end))];
                    end
                else
                    FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
                end
                FnToEvaluateParamsCell=num2cell(FnToEvaluateParamsVec);
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                Values=Values.*RemoveExits;
                ValuesOnGrid(:,ff)=Values;
            end
        end
    end
end






%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    ValuesOnGrid2=ValuesOnGrid'; % Note the transpose
    clear ProbDensityFns
    ValuesOnGrid=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        ValuesOnGrid.(AggVarNames{ff})=reshape(ValuesOnGrid2(ff,:),[n_a,n_z]);
    end
else
    % Change the ordering and size so that ValuesOnGrid has same kind of
    % shape as StationaryDist, except first dimension indexes the 'FnsToEvaluate'.
    ValuesOnGrid=ValuesOnGrid';
    ValuesOnGrid=reshape(ValuesOnGrid,[length(FnsToEvaluate),n_a,n_z]);
end

end
