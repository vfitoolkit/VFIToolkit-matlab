function CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCorr_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid,simoptions)
% Evaluates the cross-sectional correlation between every possible pair from FnsToEvaluate
% eg. if you give a FnsToEvaluate with three functions you will get three
% cross-sectional correlations; with two function you get one; with four
% you get 6.
%
% E.g., CrossSectionCorr(i,j) is the cross-sectional correlation between the i-th and j-th FnsToEvaluate


%%
if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
elseif isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
end
if Parallel~=2
    error('EvalFnOnAgentDist_CrossSectionCorr_Case1 can only be used with GPU')
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

l_daprime=size(PolicyIndexes,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);
z_gridvals=CreateGridvals(n_z,z_grid,1);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
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
StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

CrossSectionCorr=struct();

PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,simoptions);
permuteindexes=[1+(1:1:(l_a+l_z)),1];
PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]

for ff1=1:length(FnsToEvaluate)
    FnToEvaluateParamsCell1=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff1).Names);
    Values1=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff1}, FnToEvaluateParamsCell1,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
    Values1=reshape(Values1,[N_a*N_z,1]);

    for ff2=ff1:length(FnsToEvaluate)
        if ff1==ff2
            CrossSectionCorr.(AggVarNames{ff1}).(AggVarNames{ff2})=1;
        else
            FnToEvaluateParamsCell2=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff2).Names);
            Values2=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff2}, FnToEvaluateParamsCell2,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
            Values2=reshape(Values2,[N_a*N_z,1]);

            Mean1=sum(Values1.*StationaryDistVec);
            Mean2=sum(Values2.*StationaryDistVec);
            StdDev1=sqrt(sum(StationaryDistVec.*((Values1-Mean1.*ones(N_a*N_z,1)).^2)));
            StdDev2=sqrt(sum(StationaryDistVec.*((Values2-Mean2.*ones(N_a*N_z,1)).^2)));

            Numerator=sum((Values1-Mean1*ones(N_a*N_z,1,'gpuArray')).*(Values2-Mean2*ones(N_a*N_z,1,'gpuArray')).*StationaryDistVec);
            CrossSectionCorr.(AggVarNames{ff1}).(AggVarNames{ff2})=Numerator/(StdDev1*StdDev2);
        end
    end
end

for ff1=1:length(FnsToEvaluate)
    for ff2=1:ff1-1
        CrossSectionCorr.(AggVarNames{ff1}).(AggVarNames{ff2})=CrossSectionCorr.(AggVarNames{ff2}).(AggVarNames{ff1});
    end
end



end
