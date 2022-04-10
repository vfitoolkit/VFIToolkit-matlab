function CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCorr_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel)
% Evaluates the cross-sectional correlation between every possible pair from FnsToEvaluate
% eg. if you give a FnsToEvaluate with three functions you will get three
% cross-sectional correlations; with two function you get one; with four
% you get 6.
%
% E.g., CrossSectionCorr(i,j) is the cross-sectional correlation between the i-th and j-th FnsToEvaluate
%
% Parallel is an optional input

%%
if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
elseif isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
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

if Parallel==2    
    CrossSectionCorr=zeros(length(FnsToEvaluate),length(FnsToEvaluate),'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names)
            FnToEvaluateParamsVec1=[];
        else
            FnToEvaluateParamsVec1=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
        end
        
        for j=i:length(FnsToEvaluate)
            if i==j
                CrossSectionCorr(i,j)=1;
            else
                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(j).Names)
                    FnToEvaluateParamsVec2=[];
                else
                    FnToEvaluateParamsVec2=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(j).Names);
                end
                
                Values1=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{i}, FnToEvaluateParamsVec1,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
                Values1=reshape(Values1,[N_a*N_z,1]);
                Values2=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{j}, FnToEvaluateParamsVec2,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
                Values2=reshape(Values2,[N_a*N_z,1]);
                
                Mean1=sum(Values1.*StationaryDistVec);
                Mean2=sum(Values2.*StationaryDistVec);
                StdDev1=sqrt(sum(StationaryDistVec.*((Values1-Mean1.*ones(N_a*N_z,1)).^2)));
                StdDev2=sqrt(sum(StationaryDistVec.*((Values2-Mean2.*ones(N_a*N_z,1)).^2)));
                
                Numerator=sum((Values1-Mean1*ones(N_a*N_z,1,'gpuArray')).*(Values2-Mean2*ones(N_a*N_z,1,'gpuArray')).*StationaryDistVec);
                CrossSectionCorr(i,j)=Numerator/(StdDev1*StdDev2);
            end
        end
    end
    
else
    CrossSectionCorr=zeros(length(FnsToEvaluate),length(FnsToEvaluate));

    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    for i=1:length(FnsToEvaluate)
        for j=i:length(FnsToEvaluate)
            if i==j
                CrossSectionCorr(i,j)=1;
            else
                Values1=zeros(N_a*N_z,1);
                Values2=zeros(N_a*N_z,1);
                
                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(i).Names) && isempty(FnsToEvaluateParamNames(j).Names)
                    if l_d==0
                        for ii=1:N_a*N_z
                            j1=rem(ii-1,N_a)+1;
                            j2=ceil(ii/N_a);
                            Values1(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                            Values2(ii)=FnsToEvaluate{j}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                        end
                    else % l_d>0
                        for ii=1:N_a*N_z
                            j1=rem(ii-1,N_a)+1;
                            j2=ceil(ii/N_a);
                            Values1(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                            Values2(ii)=FnsToEvaluate{j}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                        end
                    end
                    
                elseif isempty(FnsToEvaluateParamNames(i).Names) % i is empty, but j is not
                    FnToEvaluateParamsCell2=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(j).Names));
                     if l_d==0
                        for ii=1:N_a*N_z
                            j1=rem(ii-1,N_a)+1;
                            j2=ceil(ii/N_a);
                            Values1(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                            Values2(ii)=FnsToEvaluate{j}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell2{:});
                        end
                    else % l_d>0
                        for ii=1:N_a*N_z
                            j1=rem(ii-1,N_a)+1;
                            j2=ceil(ii/N_a);
                            Values1(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                            Values2(ii)=FnsToEvaluate{j}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell2{:});
                        end
                     end
                     
                elseif isempty(FnsToEvaluateParamNames(j).Names) % j is empty, but i is not
                    FnToEvaluateParamsCell1=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                    if l_d==0
                        for ii=1:N_a*N_z
                            j1=rem(ii-1,N_a)+1;
                            j2=ceil(ii/N_a);
                            Values1(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell1{:});
                            Values2(ii)=FnsToEvaluate{j}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                        end
                    else % l_d>0
                        for ii=1:N_a*N_z
                            j1=rem(ii-1,N_a)+1;
                            j2=ceil(ii/N_a);
                            Values1(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell1{:});
                            Values2(ii)=FnsToEvaluate{j}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                        end
                    end
                    
                else % Neither is empty
                    FnToEvaluateParamsCell1=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                    FnToEvaluateParamsCell2=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(j).Names));
                    if l_d==0
                        for ii=1:N_a*N_z
                            j1=rem(ii-1,N_a)+1;
                            j2=ceil(ii/N_a);
                            Values1(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell1{:});
                            Values2(ii)=FnsToEvaluate{j}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell2{:});
                        end
                    else % l_d>0
                        for ii=1:N_a*N_z
                            j1=rem(ii-1,N_a)+1;
                            j2=ceil(ii/N_a);
                            Values1(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell1{:});
                            Values2(ii)=FnsToEvaluate{j}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell2{:});
                        end
                    end
                end
                
                Mean1=sum(Values1.*StationaryDistVec);
                Mean2=sum(Values2.*StationaryDistVec);
                StdDev1=sqrt(sum(StationaryDistVec.*((Values1-Mean1.*ones(N_a*N_z,1)).^2)));
                StdDev2=sqrt(sum(StationaryDistVec.*((Values2-Mean2.*ones(N_a*N_z,1)).^2)));
                
                Numerator=sum((Values1-Mean1*ones(N_a*N_z,1,'gpuArray')).*(Values2-Mean2*ones(N_a*N_z,1,'gpuArray')).*StationaryDistVec);
                CrossSectionCorr(i,j)=Numerator/(StdDev1*StdDev2);
            end
        end
end

%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    CrossSectionCorr2=CrossSectionCorr;
    clear CrossSectionCorr
    CrossSectionCorr=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff1=1:length(AggVarNames)
        for ff2=1:length(AggVarNames)
            minf1=min(ff1,ff2);
            maxf2=max(ff1,ff2);
            CrossSectionCorr.(AggVarNames{ff1}).(AggVarNames{ff2})=CrossSectionCorr2(minf1,maxf2);
        end
    end
end

end