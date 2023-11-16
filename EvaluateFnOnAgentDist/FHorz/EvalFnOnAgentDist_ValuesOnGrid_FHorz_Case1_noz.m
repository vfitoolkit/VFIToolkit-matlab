function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1_noz(PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, N_j, d_grid, a_grid, Parallel,simoptions)

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
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    elseif simoptions.keepoutputasmatrix==2
        FnsToEvaluateStruct=2;
    end
end

if ~exist('simoptions','var')
    l_aprime=l_a;
    aprime_grid=a_grid;
    n_aprime=n_a;
else
    % If using a specific asset type, then remove from aprime
    if isfield(simoptions,'experienceasset')
        l_aprime=l_a-1;
        aprime_grid=a_grid(1:sum(n_a(1:end-1)));
        n_aprime=n_a(1:end-1);
    elseif isfield(simoptions,'experienceassetu')
        l_aprime=l_a-1;
        aprime_grid=a_grid(1:sum(n_a(1:end-1)));
        n_aprime=n_a(1:end-1);
    elseif isfield(simoptions,'riskyasset')
        l_aprime=l_a-1;
        aprime_grid=a_grid(1:sum(n_a(1:end-1)));
        n_aprime=n_a(1:end-1);
    elseif isfield(simoptions,'residualasset')
        l_aprime=l_a-1;
        aprime_grid=a_grid(1:sum(n_a(1:end-1)));
        n_aprime=n_a(1:end-1);
    else % not using any specific asset type
        l_aprime=l_a;
        aprime_grid=a_grid;
        n_aprime=n_a;
    end
end

%%
if Parallel==2

    ValuesOnGrid=zeros(N_a,N_j,length(FnsToEvaluate),'gpuArray');

    PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);

    for ff=1:length(FnsToEvaluate)
        Values=nan(N_a,N_j,'gpuArray');
        for jj=1:N_j
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
            end

            Values(:,jj)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsVec,PolicyValues(:,:,jj),l_d+l_aprime,n_a,0,a_grid,[]);

            % Values(:,jj)=reshape(EvalFnOnAgentDist_Grid_Case1_noz(FnsToEvaluate{ff}, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,l_d+l_aprime]),n_d,n_a,a_grid,Parallel),[N_a,1]);
        end
        ValuesOnGrid(:,:,ff)=Values;
    end
else
    dbstack
    error('This command only works on gpu')
end

if FnsToEvaluateStruct==1
    ValuesOnGrid2=ValuesOnGrid;
    clear ValuesOnGrid
    ValuesOnGrid=struct();
    for ff=1:length(AggVarNames)
        ValuesOnGrid.(AggVarNames{ff})=reshape(ValuesOnGrid2(:,:,ff),[n_a,N_j]);
        % Change the ordering and size so that ProbDensityFns has same kind of shape as StationaryDist
    end
elseif FnsToEvaluateStruct==0
    % Change the ordering and size so that ProbDensityFns has same kind of
    % shape as StationaryDist, except first dimension indexes the
    % 'FnsToEvaluate'.
    ValuesOnGrid=permute(ValuesOnGrid,[3,1,2]);
    ValuesOnGrid=reshape(ValuesOnGrid,[length(FnsToEvaluate),n_a,N_j]);
elseif FnsToEvaluateStruct==2 % Just a rearranged version of FnsToEvaluateStruct=0 for use internally when length(FnsToEvaluate)==1
%     ValuesOnGrid=reshape(ValuesOnGrid,[N_a,N_j]);
    % The output is already in this shape anyway, so no need to actually reshape it at all
end

end
