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
    Parallel=2;
    StationaryDistpdf=gpuArray(StationaryDistpdf);
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
        % When evaluating value function (which may sometimes give -Inf
        % values) on StationaryDistVec (which at those points will be
        % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
        temp=Values.*StationaryDistpdfVec;
        AggVars(i)=sum(temp(~isnan(temp)));
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
        
        for i=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     d_val=d_gridvals{j1+(j2-1)*N_a,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass);
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistpdfVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     d_val=d_gridvals{j1+(j2-1)*N_a,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val,StationaryDistmass, SSvalueParamsVec);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass,FnToEvaluateParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistpdfVec;
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
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(aprime_val,a_val,z_val);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass);
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistpdfVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(aprime_val,a_val,z_val,SSvalueParamsVec);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},StationaryDistmass,FnToEvaluateParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistpdfVec;
                AggVars(i)=sum(temp(~isnan(temp)));
            end
        end
    end
    
end

AggVars=AggVars*StationaryDistmass;

end
