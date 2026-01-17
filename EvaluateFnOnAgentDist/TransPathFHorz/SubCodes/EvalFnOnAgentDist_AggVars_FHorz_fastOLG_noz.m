function AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(AgentDist,PolicyValues_d, PolicyValues_aprime, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,N_a,a_gridvals,outputasstructure)
% fastOLG: so (a,j)-by-1
% Policy is in Kron form

% If no d variable, set l_d=0, and then PolicyValues_d=[], d_gridvals=[].

% PolicyValues_d is [N_a,N_j,l_d]
% PolicyValues_aprime is [N_a,N_j,l_aprime]
% a_gridals is [N_a,l_a]

% parameters that depend on age must be [1,N_j]

% Note: FnsToEvaluate is already cell (converted from struct)

% if l_d==0
%     PolicyValues_aprime=reshape(aprime_gridvals(Policy_aprime(:),:),[N_a,N_j,l_aprime]);
% else
%     PolicyValues_d=reshape(d_gridvals(Policy_d(:),:),[N_a,N_j,l_d]);
%     PolicyValues_aprime=reshape(aprime_gridvals(Policy_aprime(:),:),[N_a,N_j,l_aprime]);
% end

if l_a~=l_aprime
    error('cannot yet handle l_a different from l_aprime, need more if-else statements in main body of EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz command to handle that ')
end

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
    
    if l_d==0
        if l_aprime==1
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,1), a_gridvals, ParamCell{:});
        elseif l_aprime==2
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,1), PolicyValues_aprime(:,:,2), a_gridvals(:,1), a_gridvals(:,2), ParamCell{:});
        elseif l_aprime==3
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,1), PolicyValues_aprime(:,:,2), PolicyValues_aprime(:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), ParamCell{:});
        end
    elseif l_d==1
        if l_aprime==1
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_aprime(:,:,1), a_gridvals, ParamCell{:});
        elseif l_aprime==2
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_aprime(:,:,1), PolicyValues_aprime(:,:,2), a_gridvals(:,1), a_gridvals(:,2), ParamCell{:});
        elseif l_aprime==3
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_aprime(:,:,1), PolicyValues_aprime(:,:,2), PolicyValues_aprime(:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), ParamCell{:});
        end
    elseif l_d==2 
        if l_aprime==1
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_d(:,:,2), PolicyValues_aprime(:,:,1), a_gridvals, ParamCell{:});
        elseif l_aprime==2
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_d(:,:,2), PolicyValues_aprime(:,:,1), PolicyValues_aprime(:,:,2), a_gridvals(:,1), a_gridvals(:,2), ParamCell{:});
        elseif l_aprime==3
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_d(:,:,2), PolicyValues_aprime(:,:,1), PolicyValues_aprime(:,:,2), PolicyValues_aprime(:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), ParamCell{:});
        end
    elseif l_d==3 
        if l_aprime==1
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_d(:,:,2), PolicyValues_d(:,:,3), PolicyValues_aprime(:,:,1), a_gridvals, ParamCell{:});
        elseif l_aprime==2
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_d(:,:,2), PolicyValues_d(:,:,3), PolicyValues_aprime(:,:,1), PolicyValues_aprime(:,:,2), a_gridvals(:,1), a_gridvals(:,2), ParamCell{:});
        elseif l_aprime==3
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_d(:,:,2), PolicyValues_d(:,:,3), PolicyValues_aprime(:,:,1), PolicyValues_aprime(:,:,2), PolicyValues_aprime(:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), ParamCell{:});
        end
    elseif l_d==4
        if l_aprime==1
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_d(:,:,2), PolicyValues_d(:,:,3), PolicyValues_d(:,:,4), PolicyValues_aprime(:,:,1), a_gridvals, ParamCell{:});
        elseif l_aprime==2
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_d(:,:,2), PolicyValues_d(:,:,3), PolicyValues_d(:,:,4), PolicyValues_aprime(:,:,1), PolicyValues_aprime(:,:,2), a_gridvals(:,1), a_gridvals(:,2), ParamCell{:});
        elseif l_aprime==3
            Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,1), PolicyValues_d(:,:,2), PolicyValues_d(:,:,3), PolicyValues_d(:,:,4), PolicyValues_aprime(:,:,1), PolicyValues_aprime(:,:,2), PolicyValues_aprime(:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), ParamCell{:});
        end
    end
    
    if outputasstructure==1
        AggVars.(AggVarNames{ff}).Mean=sum(Values(:).*AgentDist);
    else % outputasstructure==0
        AggVars(ff)=sum(Values(:).*AgentDist);
    end
end



end
