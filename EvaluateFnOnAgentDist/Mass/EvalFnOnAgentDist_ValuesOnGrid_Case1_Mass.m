function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_Case1_Mass(StationaryDistpdf,StationaryDistmass, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,simoptions)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate

eval('fieldexists=1;simoptions.endogenousexit;','fieldexists=0;')
if fieldexists==0
    simoptions.endogenousexit=0;
else
    if simoptions.endogenousexit==1
        eval('fieldexists=1;simoptions.keeppolicyonexit;','fieldexists=0;')
        if fieldexists==0
            simoptions.keeppolicyonexit=0;
        end
    end
end


if Parallel==2 || Parallel==4
    Parallel=2;
    StationaryDistmass=gpuArray(StationaryDistmass);
    PolicyIndexes=gpuArray(PolicyIndexes);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    z_grid=gpuArray(z_grid);
    
    % l_d not needed with Parallel=2 implementation
    l_a=length(n_a);
    l_z=length(n_z);
    
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
            PolicyIndexes=PolicyIndexes+ones(l_d+l_a,1).*(1-shiftdim(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),-1));
            % And use ExitPolicy to later replace these with nan
            ExitPolicy=logical(1-reshape(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),[N_a*N_z,1]));
        end
    end
    RemoveExits=nan(N_a*N_z,1);
    RemoveExits(logical(~ExitPolicy))=1;
    
    ValuesOnGrid=zeros(N_a*N_z,length(FnsToEvaluate),'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names)  % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsCell=StationaryDistmass;
        else
            FnToEvaluateParamsCell=[StationaryDistmass,gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names))];
        end
        Values=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{i}, FnToEvaluateParamsCell,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
%         Values(logical(ExitPolicy))='nan';
        Values=Values.*RemoveExits;
        ValuesOnGrid(:,i)=Values;
    end
else
    if n_d(1)==0
        l_d=0;
    else
        l_d=length(n_d);
    end
    l_a=length(n_a);
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    % When there is endogenous exit, add exit to the policy (to avoid what
    % would otherwise be zeros) and instead multiply the exiting by the
    % stationary dist to eliminate the 'decisions' there.
    ExitPolicy=zeros(N_a*N_z,1);
    if simoptions.endogenousexit==1
        if simoptions.keeppolicyonexit==0
            % Add one to PolicyIndexes
            PolicyIndexes=PolicyIndexes+ones(l_d+l_a,1).*(1-shiftdim(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),-1));
            % And use ExitPolicy to later replace these with nan
            ExitPolicy=1-reshape(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),[N_a*N_z,1]);
        end
    end
    RemoveExits=nan(N_a*N_z,1);
    RemoveExits(logical(~ExitPolicy))=1;

    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    ValuesOnGrid=zeros(N_a*N_z,length(FnsToEvaluate));
    
    if l_d>0
        
        for i=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass);
                end
                ValuesOnGrid(:,i)=Values.*StationaryDistpdfVec;
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass,FnToEvaluateParamsCell{:});
                end
%                 Values(logical(ExitPolicy))='nan';
                Values=Values.*RemoveExits;
                ValuesOnGrid(:,i)=Values;
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
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass);
                end
                ValuesOnGrid(:,i)=Values.*StationaryDistpdfVec;
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass,FnToEvaluateParamsCell{:});
                end
%                 Values(logical(ExitPolicy))='nan';
                Values=Values.*RemoveExits;
                ValuesOnGrid(:,i)=Values;
            end
        end
    end
end

% Change the ordering and size so that ValuesOnGrid has same kind of
% shape as StationaryDist, except first dimension indexes the
% 'FnsToEvaluate'.
ValuesOnGrid=ValuesOnGrid';
ValuesOnGrid=reshape(ValuesOnGrid,[length(FnsToEvaluate),n_a,n_z]);


end
