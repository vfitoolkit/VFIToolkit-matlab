function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_noz(StationaryDist,PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,N_j,d_grid,a_grid,Parallel,simoptions)

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
N_a=prod(n_a);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        AggVarNames=simoptions.AggVarNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end

%%
if Parallel==2
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
    
    StationaryDistVec=reshape(StationaryDist,[N_a,N_j]);
    
    PolicyValues=PolicyInd2Val_FHorz_Case1_noz(PolicyIndexes,n_d,n_a,N_j,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a)),1,1+l_a+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,l_d+l_a,N_j]
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*(l_d+l_a),N_j]);

    for ii=1:length(FnsToEvaluate)
        Values=nan(N_a,N_j,'gpuArray');
        for jj=1:N_j            
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
            end
            Values(:,jj)=reshape(EvalFnOnAgentDist_Grid_Case1_noz(FnsToEvaluate{ii}, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,l_d+l_a]),n_d,n_a,a_grid,Parallel),[N_a,1]);
        end
        AggVars(ii)=sum(sum(Values.*StationaryDistVec));
    end
    
else
    AggVars=zeros(length(FnsToEvaluate),1);

    a_gridvals=CreateGridvals(n_a,a_grid,2);

    StationaryDistVec=reshape(StationaryDist,[N_a*N_j,1]);
    
    sizePolicyIndexes=size(PolicyIndexes);
    if length(PolicyIndexes)>4 % If not in vectorized form
        PolicyIndexes=reshape(PolicyIndexes,[sizePolicyIndexes(1),N_a,N_j]);
    end
    
    for ii=1:length(FnsToEvaluate)
        Values=zeros(N_a,N_j);
        if l_d==0
            for jj=1:N_j
                [~, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,jj),n_d,n_a,n_a,0,d_grid,a_grid,1, 2);
                if ~isempty(FnsToEvaluateParamNames(ii).Names)
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                end
                for a_c=1:N_a
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names)
                        Values(a_c,jj)=FnsToEvaluate{ii}(aprime_gridvals{a_c,:},a_gridvals{a_c,:});
                    else
                        Values(a_c,jj)=FnsToEvaluate{ii}(aprime_gridvals{a_c,:},a_gridvals{a_c,:},FnToEvaluateParamsCell{:});
                    end
                end
            end
        else
            for jj=1:N_j

                [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,jj),n_d,n_a,n_a,0,d_grid,a_grid,1, 2);
                if ~isempty(FnsToEvaluateParamNames(ii).Names)
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                end
                for a_c=1:N_a
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names)
                        Values(a_c,jj)=FnsToEvaluate{ii}(d_gridvals{a_c,:},aprime_gridvals{a_c,:},a_gridvals{a_c,:});
                    else
                        Values(a_c,jj)=FnsToEvaluate{ii}(d_gridvals{a_c,:},aprime_gridvals{a_c,:},a_gridvals{a_c,:},FnToEvaluateParamsCell{:});
                    end
                end
            end
        end
        Values=reshape(Values,[N_a*N_j,1]);
        AggVars(ii)=sum(Values.*StationaryDistVec);
    end
    
end

%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    AggVars2=AggVars;
    clear AggVars
    AggVars=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        AggVars.(AggVarNames{ff}).Mean=AggVars2(ff);
    end
end


end
