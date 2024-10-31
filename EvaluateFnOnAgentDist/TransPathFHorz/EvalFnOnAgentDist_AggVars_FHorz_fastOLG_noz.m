function AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(AgentDist,Policy, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,l_d,n_a,N_j,daprime_gridvals,a_gridvals,outputasstructure)
% fastOLG: so (a,j)-by-1
% Policy is in Kron form

% daprime_gridvals is [N_d*N_aprime,l_d+l_aprime]
% a_grivdals is [N_a,l_a]

% parameters that depend on age must be [1,N_j]

% Note: FnsToEvaluate is already cell (converted from struct)

% simoptions.outputasstructure=0; % hardcoded

l_a=length(n_a);
N_a=prod(n_a);

PolicyValues=reshape(daprime_gridvals(Policy(:),:),[N_a,N_j,l_d+l_a]);

%%
if outputasstructure==1
    AggVars=struct();
else % outputasstructure==0
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
end

% AgentDist is [N_a*N_j,1]
% Policy is [N_a*N_j,1], it contains the index for (d,aprime)

for ff=1:length(FnsToEvaluate)
    Values=zeros(N_a,N_j,'gpuArray');

    if isempty(FnsToEvaluateParamNames(ff).Names)
        ParamCell=cell(0,1);
    else
        % Create a matrix containing all the return function parameters (in order).
        % Each column will be a specific parameter with the values at every age.
        FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

        nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

        ParamCell=cell(nFnToEvaluateParams,1);
        for ii=1:nFnToEvaluateParams
            ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-1)}; % (a,j,l_d+l_a), so we want j to be after a (which is N_a)
        end
    end
    
    if l_d==0 && l_a==1
        Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,1), a_gridvals, ParamCell{:});
    elseif l_d==1 && l_a==1
        Values=arrayfun(FnsToEvaluate{ff}, PolicyValues(:,:,1), PolicyValues(:,:,2), a_gridvals, ParamCell{:});
    end
    
    if outputasstructure==1
        AggVars.(AggVarNames{ff}).Mean=sum(Values(:).*AgentDist);
    else % outputasstructure==0
        AggVars(ff)=sum(Values(:).*AgentDist);
    end
end



end
