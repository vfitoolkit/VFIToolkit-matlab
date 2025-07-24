function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_Case1(Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions, EntryExitParamNames,StationaryDist)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
%
% StationaryDist, Parallel, simoptions and EntryExitParamNames are optional inputs, only needed when using endogenous entry

%%
Parallel=1+(gpuDeviceCount>0);
if ~exist('simoptions','var')
    simoptions.gridinterplayer=0;
elseif ~isfield(simoptions,'gridinterplayer')
    simoptions.gridinterplayer=0;    
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

l_daprime=size(Policy,1);
if simoptions.gridinterplayer==1
    l_daprime=l_daprime-1;
end
a_gridvals=CreateGridvals(n_a,a_grid,1);
[z_gridvals,~,simoptions]=ExogShockSetup(n_z,z_grid,[],Parameters,simoptions,1);

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
if exist('StationaryDist','var')
    if isstruct(StationaryDist)
        % Even though Mass is unimportant, still need to deal with 'exit' in Policy
        ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_InfHorz_Mass(StationaryDist.mass, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,simoptions);
        return
    end
end

ValuesOnGrid=struct();

if Parallel==2
    Policy=gpuArray(Policy);
    l_daprime=size(Policy,1);

    PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,gpuArray(d_grid),gpuArray(a_grid),simoptions,1);
    if l_z==0
        PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a]),[2,1]); %[N_a,l_d+l_a]
    else
        PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]
    end

    for ff=1:length(FnsToEvaluate)
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
        Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
        Values=reshape(Values,[N_a*N_z,1]);
        ValuesOnGrid.(AggVarNames{ff})=reshape(Values,[n_a,n_z]);
    end
else
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(Policy,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
        
    if l_d>0
        
        for ff=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'FnsToEvaluateParamNames(i).Names={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                ValuesOnGrid.(AggVarNames{ff})=reshape(Values,[n_a,n_z]);
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                ValuesOnGrid.(AggVarNames{ff})=reshape(Values,[n_a,n_z]);
            end
        end
    
    else %l_d=0
        
        for ff=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'FnsToEvaluateParamNames(i).Names={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                ValuesOnGrid.(AggVarNames{ff})=reshape(Values,[n_a,n_z]);
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                ValuesOnGrid.(AggVarNames{ff})=reshape(Values,[n_a,n_z]);
            end
        end
    end
end

%%
if FnsToEvaluateStruct==0
    % Change the output into a structure
    ValuesOnGrid2=ValuesOnGrid;
    clear ValuesOnGrid
    ValuesOnGrid=zeros(n_a,n_z,length(FnsToEvaluate));
    for ff=1:length(AggVarNames)
        ValuesOnGrid(:,:,ff)=ValuesOnGrid2.(AggVarNames{ff});
    end
end


end
