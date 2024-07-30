function AggVars=EvalFnOnAgentDist_AggVars_Case1_Mass(StationaryDistpdf,StationaryDistmass, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,simoptions)
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
    StationaryDistpdf=gpuArray(StationaryDistpdf);
    StationaryDistmass=gpuArray(StationaryDistmass);
    PolicyIndexes=gpuArray(PolicyIndexes);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    l_daprime=size(PolicyIndexes,1);
    a_gridvals=CreateGridVals(n_a,gpuArray(a_grid),1);
    z_gridvals=CreateGridVals(n_z,gpuArray(z_grid),1);
    
    % l_d not needed with Parallel=2 implementation
    l_a=length(n_a);
    l_z=length(n_z);
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    StationaryDistpdfVec=reshape(StationaryDistpdf,[N_a*N_z,1]);

    % When there is endogenous exit, add exit to the policy (to avoid what
    % would otherwise be zeros) and instead multiply the exiting by the
    % stationary dist to eliminate the 'decisions' there.
    if simoptions.endogenousexit==1
        if simoptions.keeppolicyonexit==0
            if n_d(1)==0
                l_d=0;
            else
                l_d=length(n_d);
            end
            % Add one to PolicyIndexes
            PolicyIndexes=PolicyIndexes+ones(l_d+l_a,1).*(1-shiftdim(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),-1));
            % And make the corresponding StationaryDistpdfVec entries zero,
            % so the values are anyway ignored.
            ExitPolicy=logical(1-reshape(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),[N_a*N_z,1]));
            StationaryDistpdfVec(ExitPolicy)=0;
        end
    end
    
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for ff=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
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
        Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
        Values=reshape(Values,[N_a*N_z,1]);
        % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec 
        % (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
        temp=Values.*StationaryDistpdfVec;
        AggVars(ff)=sum(temp(~isnan(temp)));
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
    
    StationaryDistpdfVec=reshape(StationaryDistpdf,[N_a*N_z,1]);
    
    StationaryDistpdfVec=gather(StationaryDistpdfVec);
    StationaryDistmass=gather(StationaryDistmass);

    % When there is endogenous exit, add exit to the policy (to avoid what
    % would otherwise be zeros) and instead multiply the exiting by the
    % stationary dist to eliminate the 'decisions' there.
    if simoptions.endogenousexit==1
        if simoptions.keeppolicyonexit==0
            % Add one to PolicyIndexes which correspond to exit
            PolicyIndexes=PolicyIndexes+ones(l_d+l_a,1).*(1-shiftdim(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),-1));
            % And make the corresponding StationaryDistpdfVec entries zero,
            % so the values are anyway ignored.
            ExitPolicy=1-reshape(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),[N_a*N_z,1]);
            StationaryDistpdfVec(logical(ExitPolicy))=0;
        end
    end

    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    AggVars=zeros(length(FnsToEvaluate),1);
    
    if l_d>0
        
        for ff=1:length(FnsToEvaluate)
            if isempty(FnsToEvaluateParamNames(ff).Names)
                FnToEvaluateParamsVec={};
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
                FnToEvaluateParamsVec=num2cell(FnToEvaluateParamsVec);
            end
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsVec{:});
            end
            % When evaluating value function (which may sometimes give -Inf
            % values) on StationaryDistVec (which at those points will be
            % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistpdfVec;
            AggVars(ff)=sum(temp(~isnan(temp)));
        end
    
    else %l_d=0
        
        for ff=1:length(FnsToEvaluate)
            if isempty(FnsToEvaluateParamNames(ff).Names)
                FnToEvaluateParamsVec={};
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
                FnToEvaluateParamsVec=num2cell(FnToEvaluateParamsVec);
            end
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsVec{:});
            end
            % When evaluating value function (which may sometimes give -Inf
            % values) on StationaryDistVec (which at those points will be
            % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistpdfVec;
            AggVars(ff)=sum(temp(~isnan(temp)));
        end
    end
    
end

AggVars=AggVars*StationaryDistmass;

end
