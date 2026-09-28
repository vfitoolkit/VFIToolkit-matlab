function AggVars=EvalFnOnAgentDist_AggVars_InfHorz_Mass_MixExit(StationaryDistpdf,StationaryDistmass, PolicyIndexes, PolicyIndexesWhenExiting, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,exitprobs,simoptions)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
%
% This is the simoptions.endogenousexit=2 path (mixture of exogenous and endogenous exit), the
% twin of EvalFnOnAgentDist_AggVars_InfHorz_Mass(). Keep the two in step: where they differ it
% should be because of the exit mixture, not because one of them has drifted.

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
    l_daprime=size(PolicyIndexes,1)-2*simoptions.gridinterplayer; % gridinterplayer=1 carries an extra L2 index and L2flag
    a_gridvals=CreateGridvals(n_a,gpuArray(a_grid),1);
    z_gridvals=CreateGridvals(n_z,gpuArray(z_grid),1);

    % l_d not needed with Parallel=2 implementation
    l_a=length(n_a);

    N_a=prod(n_a);
    N_z=prod(n_z);

    StationaryDistpdfVec=reshape(StationaryDistpdf,[N_a*N_z,1]);

    % Probability of endogenous exit, NOT a flag. CondlProbOfSurvival is a probability and need
    % not be binary: logical(1-p) would be true for any p<1, which sent the entire mass down the
    % exiting leg. Legs 2 and 3 below sum to exitprobs(2) for any p, so mass is conserved either
    % way. Same fix as in EvalFnOnAgentDist_AggVars_InfHorz_Mass().
    ProbOfExit=1-reshape(gpuArray(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:})),[N_a*N_z,1]);

    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

    PolicyValues=PolicyInd2Val_InfHorz(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,simoptions);
    PolicyValues=reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1]); %[N_a,N_z,l_d+l_a]
    PolicyValuesWhenExiting=PolicyInd2Val_InfHorz(PolicyIndexesWhenExiting,n_d,n_a,n_z,d_grid,a_grid,simoptions);
    PolicyValuesWhenExiting=reshape(PolicyValuesWhenExiting,[size(PolicyValuesWhenExiting,1),N_a,N_z]);
    PolicyValuesPermuteWhenExiting=permute(PolicyValuesWhenExiting,[2,3,1]); %[N_a,N_z,l_d+l_a]

    for ff=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        % Note: agentmass is opt-in, by naming it as the first parameter, exactly as in
        % EvalFnOnAgentDist_AggVars_InfHorz_Mass(). It used to be prepended unconditionally here,
        % which meant the same model needed its FnsToEvaluate rewritten with an extra leading
        % argument just to switch between simoptions.endogenousexit=1 and =2.
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
            temp=temp+exitprobs(2)*(1-ProbOfExit).*Values.*StationaryDistpdfVec;
        end
        if ExitStatus(3)==1
            temp=temp+exitprobs(2)*ProbOfExit.*ValuesWhenExiting.*StationaryDistpdfVec;
        end
        if ExitStatus(4)==1
            temp=temp+exitprobs(3)*ValuesWhenExiting.*StationaryDistpdfVec;
        end
        % Following commented out line is just doing the same as the above four if statements but in a single line. Is residual code from earlier version.
%         temp=exitprobs(1)*Values.*StationaryDistpdfVec+exitprobs(2)*((1-ProbOfExit).*Values+ProbOfExit.*ValuesWhenExiting).*StationaryDistpdfVec+exitprobs(3)*ValuesWhenExiting.*StationaryDistpdfVec;
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

    % Probability of endogenous exit, NOT a flag; see the note in the Parallel==2 branch above.
    % This branch was already numeric (it never had the logical() the GPU branch did), so the
    % two branches now agree.
    ProbOfExit=gather(1-reshape(Parameters.(EntryExitParamNames.CondlProbOfSurvival{:}),[N_a*N_z,1]));

    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,simoptions,1, 2);
    [d_gridvalsWhenExiting, aprime_gridvalsWhenExiting]=CreateGridvals_Policy(PolicyIndexesWhenExiting,n_d,n_a,n_a,n_z,d_grid,a_grid,simoptions,1, 2);
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
            % Note: agentmass is opt-in, by naming it as the first parameter, exactly as in
            % EvalFnOnAgentDist_AggVars_InfHorz_Mass(). It used to be passed unconditionally here.
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

            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                if calcNotExit==1
                    Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsVec{:});
                end
                if calcExit==1
                    ValuesWhenExiting(ii)=FnsToEvaluate{ff}(d_gridvalsWhenExiting{j1+(j2-1)*N_a,:},aprime_gridvalsWhenExiting{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsVec{:});
                end
            end
            if ExitStatus(1)==1
                temp=exitprobs(1)*Values.*StationaryDistpdfVec;
            else
                temp=zeros(N_a*N_z,1);
            end
            if ExitStatus(2)==1
                temp=temp+exitprobs(2)*(1-ProbOfExit).*Values.*StationaryDistpdfVec;
            end
            if ExitStatus(3)==1
                temp=temp+exitprobs(2)*ProbOfExit.*ValuesWhenExiting.*StationaryDistpdfVec;
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
            % Note: agentmass is opt-in, by naming it as the first parameter, exactly as in
            % EvalFnOnAgentDist_AggVars_InfHorz_Mass(). It used to be passed unconditionally here.
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

            for ii=1:N_a*N_z
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                if calcNotExit==1
                    Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsVec{:});
                end
                if calcExit==1
                    ValuesWhenExiting(ii)=FnsToEvaluate{ff}(aprime_gridvalsWhenExiting{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsVec{:});
                end
            end

            if ExitStatus(1)==1
                temp=exitprobs(1)*Values.*StationaryDistpdfVec;
            else
                temp=zeros(N_a*N_z,1);
            end
            if ExitStatus(2)==1
                temp=temp+exitprobs(2)*(1-ProbOfExit).*Values.*StationaryDistpdfVec;
            end
            if ExitStatus(3)==1
                temp=temp+exitprobs(2)*ProbOfExit.*ValuesWhenExiting.*StationaryDistpdfVec;
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
