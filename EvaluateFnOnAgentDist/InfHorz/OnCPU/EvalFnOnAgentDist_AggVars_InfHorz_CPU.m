function AggVars=EvalFnOnAgentDist_AggVars_InfHorz_CPU(StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions)
% CPU-only path for EvalFnOnAgentDist_AggVars_InfHorz.
% Called from the main InfHorz file when simoptions.parallel==1.
% On CPU, only the basics are allowed: no e, no semiz, z_grid stays in input form.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

if N_d==0 && isscalar(n_a) && simoptions.gridinterplayer==0
    l_daprime=1;
else
    l_daprime=size(Policy,1);
    if simoptions.gridinterplayer==1
        l_daprime=l_daprime-1;
    end
end
a_gridvals=CreateGridvals(n_a,a_grid,1);
z_gridvals=z_grid; % On cpu, only basics are allowed. No e.

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end

%%
[d_gridvals, aprime_gridvals]=CreateGridvals_Policy(Policy,n_d,n_a,n_a,n_z,d_grid,a_grid,simoptions,1, 2);
a_gridvals=num2cell(a_gridvals);
z_gridvals=num2cell(z_gridvals);

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
StationaryDistVec=gather(StationaryDistVec);

AggVars=zeros(length(FnsToEvaluate),1);

if l_d>0

    for ff=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'SSvalueParamNames={}'
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
            end
            % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistVec;
            AggVars(ff)=sum(temp(~isnan(temp)));
        else
            FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{ff}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
            end
            % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistVec;
            AggVars(ff)=sum(temp(~isnan(temp)));
        end
    end

else % l_d=0

    for ff=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names) % check for 'SSvalueParamNames={}'
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
            end
            % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistVec;
            AggVars(ff)=sum(temp(~isnan(temp)));
        else
            FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{ff}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
            end
            % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistVec;
            AggVars(ff)=sum(temp(~isnan(temp)));
        end
    end
end

%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    AggVars2=AggVars;
    clear AggVars
    AggVars=struct();
    for ff=1:length(FnsToEvalNames)
        AggVars.(FnsToEvalNames{ff}).Mean=AggVars2(ff);
    end
end

end
