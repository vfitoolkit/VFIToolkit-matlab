function AggVars=EvalFnOnAgentDist_AggVars_InfHorz_Mass(StationaryDistpdf,StationaryDistmass, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,simoptions)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate

if ~isfield(simoptions,'endogenousexit')
    simoptions.endogenousexit=0;
else
    if simoptions.endogenousexit==1
        if ~isfield(simoptions,'keeppolicyonexit')
            simoptions.keeppolicyonexit=0;
        end
    end
end



if Parallel==2
    StationaryDistpdf=gpuArray(StationaryDistpdf);
    StationaryDistmass=gpuArray(StationaryDistmass);
    PolicyIndexes=gpuArray(PolicyIndexes);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    l_daprime=size(PolicyIndexes,1)-2*simoptions.gridinterplayer; % gridinterplayer=1 carries an extra L2 index and L2flag
    a_gridvals=CreateGridvals(n_a,gpuArray(a_grid),1);
    z_gridvals=CreateGridvals(n_z,gpuArray(z_grid),1);

    % l_d not needed with Parallel=2 implementation
    l_a=length(n_a);

    N_a=prod(n_a);
    N_z=prod(n_z);

    StationaryDistpdfVec=reshape(StationaryDistpdf,[N_a*N_z,1]);

    % When there is endogenous exit, add exit to the policy (to avoid what
    % would otherwise be zeros) and instead multiply the exiting by the
    % stationary dist to eliminate the 'decisions' there.
    if simoptions.endogenousexit==1
        if simoptions.keeppolicyonexit==0
            % Policy is zero wherever the agent exits (the deliberate sentinel the value fn
            % iteration puts there). Make those indices valid so PolicyInd2Val can run, and zero
            % their mass so the values are ignored anyway.
            %
            % Keyed on the sentinel itself, NOT on 1-CondlProbOfSurvival. Survival is a
            % PROBABILITY and need not be binary: StationaryDist_InfHorz_Iteration_EntryExit_raw
            % already uses it as a transition weight and handles exit separately via
            % (optaprime>0). With a fractional survival the old form broke twice over -- adding
            % 1-p to the policy gives a non-integer index, and logical(1-p) is true for ANY p<1,
            % which zeroed the mass at every state rather than just the exiting ones.
            exitstates=reshape(any(PolicyIndexes==0,1),[N_a*N_z,1]);
            PolicyIndexes(PolicyIndexes==0)=1;
            StationaryDistpdfVec(exitstates)=0;
        end
    end

    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

    PolicyValues=PolicyInd2Val_InfHorz(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,simoptions);
    PolicyValues=reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1]); %[N_a,N_z,l_d+l_a]

    for ff=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names)
            FnToEvaluateParamsCell=cell(0);
        else
            if strcmp(FnsToEvaluateParamNames(ff).Names{1},'agentmass')
                if isscalar(FnsToEvaluateParamNames(ff).Names)
                    FnToEvaluateParamsCell={StationaryDistmass};
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
            % Policy is zero wherever the agent exits (the deliberate sentinel). Make those indices
            % valid and zero their mass, since the values are ignored anyway. Keyed on the sentinel,
            % NOT on 1-CondlProbOfSurvival: survival is a PROBABILITY and need not be binary, and a
            % fractional 1-p both gives a non-integer index and makes logical(1-p) true everywhere.
            exitstates=reshape(any(PolicyIndexes==0,1),[N_a*N_z,1]);
            PolicyIndexes(PolicyIndexes==0)=1;
            StationaryDistpdfVec(exitstates)=0;
        end
    end

    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,simoptions,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);

    AggVars=zeros(length(FnsToEvaluate),1);

    if l_d>0

        for ff=1:length(FnsToEvaluate)
            if isempty(FnsToEvaluateParamNames(ff).Names)
                FnToEvaluateParamsVec={};
            else
                if strcmp(FnsToEvaluateParamNames(ff).Names{1},'agentmass')
                    if isscalar(FnsToEvaluateParamNames(ff).Names)
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
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsVec{:});
            end
            % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistpdfVec;
            AggVars(ff)=sum(temp(~isnan(temp)));
        end

    else %l_d=0

        for ff=1:length(FnsToEvaluate)
            if isempty(FnsToEvaluateParamNames(ff).Names)
                FnToEvaluateParamsVec={};
            else
                if strcmp(FnsToEvaluateParamNames(ff).Names{1},'agentmass')
                    if isscalar(FnsToEvaluateParamNames(ff).Names)
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
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsVec{:});
            end
            % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistpdfVec;
            AggVars(ff)=sum(temp(~isnan(temp)));
        end
    end

end

AggVars=AggVars*StationaryDistmass;

end
