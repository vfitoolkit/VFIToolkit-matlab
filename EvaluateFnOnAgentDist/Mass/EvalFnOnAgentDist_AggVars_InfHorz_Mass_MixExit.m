function AggVars=EvalFnOnAgentDist_AggVars_InfHorz_Mass_MixExit(StationaryDistpdf,StationaryDistmass, PolicyIndexes, PolicyIndexesWhenExiting, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,exitprobs)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate

% exitprobs=simoptions.exitprobabilities;

if ~isfield(FnsToEvaluateParamNames,'ExitStatus')
    FnsToEvaluateParamNames(1).ExitStatus=[1,1,1,1];
end

if Parallel==2
    StationaryDistpdf=gpuArray(StationaryDistpdf);
    StationaryDistmass=gpuArray(StationaryDistmass);
    PolicyIndexes=gpuArray(PolicyIndexes);
    PolicyIndexesWhenExiting=gpuArray(PolicyIndexesWhenExiting);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    l_daprime=size(PolicyIndexes,1);
    a_gridvals=CreateGridVals(n_a,gpuArray(a_grid),1);
    z_gridvals=CreateGridVals(n_z,gpuArray(z_grid),1);
    
    % l_d not needed with Parallel=2 implementation
    l_a=length(n_a);
    l_z=length(n_z);
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    StationaryDistpdfVec=reshape(StationaryDistpdf,[N_a*N_z,1]);

    ExitPolicy=logical(1-reshape(gpuArray(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:})),[N_a*N_z,1]));
    
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid);
    PolicyValuesWhenExiting=PolicyInd2Val_Case1(PolicyIndexesWhenExiting,n_d,n_a,n_z,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    PolicyValuesPermuteWhenExiting=permute(PolicyValuesWhenExiting,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for ff=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names)
            FnToEvaluateParamsCell={StationaryDistmass};
        else
            FnToEvaluateParamsCell=cell(1,1+length(FnsToEvaluateParamNames(ff).Names));
            FnToEvaluateParamsCell(1)={StationaryDistmass};
            FnToEvaluateParamsCell(2:end)=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
        end
        
        if ~isempty(FnsToEvaluateParamNames(ff).ExitStatus)
            ExitStatus=FnsToEvaluateParamNames(ff).ExitStatus;
            calcNotExit=1-prod(1-FnsToEvaluateParamNames(ff).ExitStatus(1:2)); % check if either of the first two elements of ExitStatus is 1
            calcExit=1-prod(1-FnsToEvaluateParamNames(ff).ExitStatus(3:4)); % check if either of the third or fourth elements of ExitStatus is 1
        else
            ExitStatus=[1,1,1,1]; % Default
            calcNotExit=1;
            calcExit=1;
        end
        
        if calcNotExit==1
            Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
            Values=reshape(Values,[N_a*N_z,1]);
        end
        if calcExit==1
            ValuesWhenExiting=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteWhenExiting,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
            ValuesWhenExiting=reshape(ValuesWhenExiting,[N_a*N_z,1]);
        end
        
        % When evaluating value function (which may sometimes give -Inf
        % values) on StationaryDistVec (which at those points will be
        % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
        if ExitStatus(1)==1
            temp=exitprobs(1)*Values.*StationaryDistpdfVec;
        else
            temp=zeros(N_a*N_z,1);
        end
        if ExitStatus(2)==1
            temp=temp+exitprobs(2)*(1-ExitPolicy).*Values.*StationaryDistpdfVec;
        end
        if ExitStatus(3)==1
            temp=temp+exitprobs(2)*ExitPolicy.*ValuesWhenExiting.*StationaryDistpdfVec;
        end
        if ExitStatus(4)==1
            temp=temp+exitprobs(3)*ValuesWhenExiting.*StationaryDistpdfVec;
        end
        % Following commented out line is just doing the same as the above four if statements but in a single line. Is residual code from earlier version.
%         temp=exitprobs(1)*Values.*StationaryDistpdfVec+exitprobs(2)*((1-ExitPolicy).*Values+ExitPolicy.*ValuesWhenExiting).*StationaryDistpdfVec+exitprobs(3)*ValuesWhenExiting.*StationaryDistpdfVec;
        AggVars(ff)=sum(temp(~isnan(temp)));
    end
    
else
    if n_d(1)==0
        l_d=0;
    else
        l_d=length(n_d);
    end
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    StationaryDistpdfVec=reshape(StationaryDistpdf,[N_a*N_z,1]);
    
    StationaryDistpdfVec=gather(StationaryDistpdfVec);
    StationaryDistmass=gather(StationaryDistmass);

    ExitPolicy=gather(1-reshape(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),[N_a*N_z,1]));
    
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    [d_gridvalsWhenExiting, aprime_gridvalsWhenExiting]=CreateGridvals_Policy(PolicyIndexesWhenExiting,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    AggVars=zeros(length(FnsToEvaluate),1);
    
    if l_d>0
        
        for ff=1:length(FnsToEvaluate)
            if ~isempty(FnsToEvaluateParamNames(ff).ExitStatus)
                ExitStatus=FnsToEvaluateParamNames(ff).ExitStatus;
                calcNotExit=1-prod(1-FnsToEvaluateParamNames(ff).ExitStatus(1:2)); % check if either of the first two elements of ExitStatus is 1
                calcExit=1-prod(1-FnsToEvaluateParamNames(ff).ExitStatus(3:4)); % check if either of the third or fourth elements of ExitStatus is 1
            else
                ExitStatus=[1,1,1,1]; % Default
                calcNotExit=1;
                calcExit=1;
            end
            
            if calcNotExit==1
                Values=zeros(N_a*N_z,1);
            end
            if calcExit==1
                ValuesWhenExiting=zeros(N_a*N_z,1);
            end
            
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'SSvalueParamNames={}'

                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    if calcNotExit==1
                        Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass);
                    end
                    if calcExit==1
                        ValuesWhenExiting(ii)=FnsToEvaluate{ff}(d_gridvalsWhenExiting{j1+(j2-1)*N_a,:},aprime_gridvalsWhenExiting{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass);
                    end
                end
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));

                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    if calcNotExit==1
                        Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass,FnToEvaluateParamsCell{:});
                    end
                    if calcExit==1
                        ValuesWhenExiting(ii)=FnsToEvaluate{ff}(d_gridvalsWhenExiting{j1+(j2-1)*N_a,:},aprime_gridvalsWhenExiting{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass,FnToEvaluateParamsCell{:});
                    end
                end
                        
            end
            if ExitStatus(1)==1
                temp=exitprobs(1)*Values.*StationaryDistpdfVec;
            else
                temp=zeros(N_a*N_z,1);
            end
            if ExitStatus(2)==1
                temp=temp+exitprobs(2)*(1-ExitPolicy).*Values.*StationaryDistpdfVec;
            end
            if ExitStatus(3)==1
                temp=temp+exitprobs(2)*ExitPolicy.*ValuesWhenExiting.*StationaryDistpdfVec;
            end
            if ExitStatus(4)==1
                temp=temp+exitprobs(3)*ValuesWhenExiting.*StationaryDistpdfVec;
            end
            AggVars(ff)=sum(temp(~isnan(temp)));
        end
    
    else %l_d=0
        
        for ff=1:length(FnsToEvaluate)
            if ~isempty(FnsToEvaluateParamNames(ff).ExitStatus)
                ExitStatus=FnsToEvaluateParamNames(ff).ExitStatus;
                calcNotExit=1-prod(1-FnsToEvaluateParamNames(ff).ExitStatus(1:2)); % check if either of the first two elements of ExitStatus is 1
                calcExit=1-prod(1-FnsToEvaluateParamNames(ff).ExitStatus(3:4)); % check if either of the third or fourth elements of ExitStatus is 1
            else
                ExitStatus=[1,1,1,1]; % Default
                calcNotExit=1;
                calcExit=1;
            end
            
            if calcNotExit==1
                Values=zeros(N_a*N_z,1);
            end
            if calcExit==1
                ValuesWhenExiting=zeros(N_a*N_z,1);
            end
            
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'SSvalueParamNames={}'
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    if calcNotExit==1
                        Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass);
                    end
                    if calcExit==1
                        ValuesWhenExiting(ii)=FnsToEvaluate{ff}(aprime_gridvalsWhenExiting{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass);
                    end
                end
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
   
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    if calcNotExit==1
                        Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass,FnToEvaluateParamsCell{:});
                    end
                    if calcExit==1
                        ValuesWhenExiting(ii)=FnsToEvaluate{ff}(aprime_gridvalsWhenExiting{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass,FnToEvaluateParamsCell{:});
                    end
                end
            end
            
            if ExitStatus(1)==1
                temp=exitprobs(1)*Values.*StationaryDistpdfVec;
            else
                temp=zeros(N_a*N_z,1);
            end
            if ExitStatus(2)==1
                temp=temp+exitprobs(2)*(1-ExitPolicy).*Values.*StationaryDistpdfVec;
            end
            if ExitStatus(3)==1
                temp=temp+exitprobs(2)*ExitPolicy.*ValuesWhenExiting.*StationaryDistpdfVec;
            end
            if ExitStatus(4)==1
                temp=temp+exitprobs(3)*ValuesWhenExiting.*StationaryDistpdfVec;
            end
            AggVars(ff)=sum(temp(~isnan(temp)));
        end
    end
    
end

AggVars=AggVars*StationaryDistmass;

end
