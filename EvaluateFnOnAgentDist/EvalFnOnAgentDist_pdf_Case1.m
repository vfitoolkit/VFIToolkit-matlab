function ProbDensityFns=EvalFnOnAgentDist_pdf_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,simoptions,EntryExitParamNames)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate

% simoptions and EntryExitParamNames are optional inputs, only needed when using endogenous entry

if isstruct(StationaryDist)
    % Even though Mass is unimportant, still need to deal with 'exit' in PolicyIndexes.
    ProbDensityFns=EvalFnOnAgentDist_pdf_Case1_Mass(StationaryDist.pdf,StationaryDist.mass, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,simoptions);
    return
end

if Parallel==2 || Parallel==4
    Parallel=2;
    StationaryDist=gpuArray(StationaryDist);
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
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

    ProbDensityFns=zeros(N_a*N_z,length(FnsToEvaluate),'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names)  % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsCell=[];
        else
            FnToEvaluateParamsCell=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
        end
        Values=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{i}, FnToEvaluateParamsCell,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        ProbDensityFns(:,i)=Values.*StationaryDistVec;
    end
    
else
    if n_d(1)==0
        l_d=0;
    else
        l_d=length(n_d);
    end
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    
    StationaryDistVec=gather(StationaryDistVec);
    
    ProbDensityFns=zeros(N_a*N_z,length(FnsToEvaluate));
    
    if l_d>0
        
        for i=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                ProbDensityFns(:,i)=Values.*StationaryDistVec;
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                ProbDensityFns(:,i)=Values.*StationaryDistVec;
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
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                ProbDensityFns(:,i)=Values.*StationaryDistVec;
            else
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
                ProbDensityFns(:,i)=Values.*StationaryDistVec;
            end
        end
    end
end

% Normalize to 1 (to make it a pdf)
for i=1:length(FnsToEvaluate)
    ProbDensityFns(:,i)=ProbDensityFns(:,i)/sum(ProbDensityFns(:,i));
end

% When evaluating value function (which may sometimes give -Inf
% values) on StationaryDistVec (which at those points will be
% 0) we get 'NaN'. Just eliminate those.
ProbDensityFns(isnan(ProbDensityFns))=0;

% Change the ordering and size so that ProbDensityFns has same kind of
% shape as StationaryDist, except first dimension indexes the
% 'FnsToEvaluate'.
ProbDensityFns=ProbDensityFns';
ProbDensityFns=reshape(ProbDensityFns,[length(FnsToEvaluate),n_a,n_z]);

end
