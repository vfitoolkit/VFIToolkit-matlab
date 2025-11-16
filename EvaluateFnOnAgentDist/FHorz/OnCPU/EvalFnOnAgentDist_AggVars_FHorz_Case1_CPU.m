function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_CPU(StationaryDist,Policy, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)
% cpu version only does the basics

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=1; % hardcoded for CPU
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
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        AggVarNames=simoptions.AggVarNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end


%%

AggVars=zeros(length(FnsToEvaluate),1);

a_gridvals=CreateGridvals(n_a,a_grid,2);
z_gridvals=CreateGridvals(n_z,z_grid,2);

StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);

sizePolicyIndexes=size(Policy);
if length(Policy)>4 % If not in vectorized form
    Policy=reshape(Policy,[sizePolicyIndexes(1),N_a,N_z,N_j]);
end

for ii=1:length(FnsToEvaluate)
    Values=zeros(N_a,N_z,N_j);
    if l_d==0
        for jj=1:N_j

            [~, aprime_gridvals]=CreateGridvals_Policy(Policy(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
            if ~isempty(FnsToEvaluateParamNames(ii).Names)
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
            end
            for a_c=1:N_a
                for z_c=1:N_z
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names)
                        Values(a_c,z_c,jj)=FnsToEvaluate{ii}(aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:});
                    else
                        Values(a_c,z_c,jj)=FnsToEvaluate{ii}(aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:},FnToEvaluateParamsCell{:});
                    end
                end
            end
        end
    else
        for jj=1:N_j

            [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(Policy(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
            if ~isempty(FnsToEvaluateParamNames(ii).Names)
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
            end
            for a_c=1:N_a
                for z_c=1:N_z
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names)
                        Values(a_c,z_c,jj)=FnsToEvaluate{ii}(d_gridvals{a_c+(z_c-1)*N_a,:},aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:});
                    else
                        Values(a_c,z_c,jj)=FnsToEvaluate{ii}(d_gridvals{a_c+(z_c-1)*N_a,:},aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:},FnToEvaluateParamsCell{:});
                    end
                end
            end
        end
    end
    Values=reshape(Values,[N_a*N_z*N_j,1]);
    AggVars(ii)=sum(Values.*StationaryDistVec);
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
