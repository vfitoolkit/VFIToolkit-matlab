function AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist,PolicyValues_d, PolicyValues_aprime, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_z,N_a,N_z,a_gridvals,z_gridvals_J_fastOLG,outputasstructure)
% fastOLG: so (a,j)-by-z
% Policy can be in fastOLG for or not, use fastOLGpolicy=1 or 0 to indicate this
% No z is treated elsewhere
% If you have e or semiz, just disguise them as z for this command

% If no d variable, set l_d=0, and then PolicyValues_d=[], d_gridvals=[].

% PolicyValues_d is [N_a,N_j,N_z,l_d]
% PolicyValues_aprime is [N_a,N_j,N_z,l_aprime]
% a_gridvals is [N_a,l_a]
% z_gridvals_J_fastOLG is [1,N_j,N_z,l_z] (convert internally to [1,N_j,N_z,l_z])

% parameters that depend on age must be [1,N_j]

% Note: FnsToEvaluate is already cell (converted from struct)

if l_a~=l_aprime
    error('cannot yet handle l_a different from l_aprime, need more if-else statements in main body of EvalFnOnAgentDist_AggVars_FHorz_fastOLG command to handle that ')
end


%%
if outputasstructure==1
    AggVars=struct();
else % outputasstructure==0
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
end

% AgentDist is [N_a*N_j*N_z,1] or [N_a*N_j,N_e] or [N_a*N_j*N_z,N_e]
% PolicyValues is [N_a,N_j,N_ze]

for ff=1:length(FnsToEvaluate)
    Values=zeros(N_a,N_j,N_z,'gpuArray');


    if isempty(FnsToEvaluateParamNames(ff).Names)
        ParamCell=cell(0,1);
    else
        % Create a matrix containing all the return function parameters (in order).
        % Each column will be a specific parameter with the values at every age.
        FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

        nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

        ParamCell=cell(nFnToEvaluateParams,1);
        for ii=1:nFnToEvaluateParams
            ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-1)}; % (a,j,z,l_d+l_a), so we want j to be after a (which is N_a)
        end
    end
    
    if l_z==1
        if l_d==0
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            end
        elseif l_d==1
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            end
        elseif l_d==2
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            end
        elseif l_d==3
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            end
        elseif l_d==4
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), ParamCell{:});
            end
        end
    elseif l_z==2
        if l_d==0
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            end
        elseif l_d==1
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            end
        elseif l_d==2
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            end
        elseif l_d==3
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            end
        elseif l_d==4
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), ParamCell{:});
            end
        end
    elseif l_z==3
        if l_d==0
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            end
        elseif l_d==1
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            end
        elseif l_d==2
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            end
        elseif l_d==3
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            end
        elseif l_d==4
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), ParamCell{:});
            end
        end
    elseif l_z==4
        if l_d==0
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            end
        elseif l_d==1
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            end
        elseif l_d==2
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            end
        elseif l_d==3
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            end
        elseif l_d==4
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), ParamCell{:});
            end
        end
    elseif l_z==5
        if l_d==0
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            end
        elseif l_d==1
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            end
        elseif l_d==2
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            end
        elseif l_d==3
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            end
        elseif l_d==4
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), ParamCell{:});
            end
        end
    elseif l_z==6
        if l_d==0
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            end
        elseif l_d==1
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            end
        elseif l_d==2
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            end
        elseif l_d==3
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            end
        elseif l_d==4
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), ParamCell{:});
            end
        end
    elseif l_z==7
        if l_d==0
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            end
        elseif l_d==1
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            end
        elseif l_d==2
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            end
        elseif l_d==3
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            end
        elseif l_d==4
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), ParamCell{:});
            end
        end
    elseif l_z==8
        if l_d==0
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            end
        elseif l_d==1
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            end
        elseif l_d==2
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            end
        elseif l_d==3
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            end
        elseif l_d==4
            if l_aprime==1
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), a_gridvals, z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            elseif l_aprime==2
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), a_gridvals(:,1), a_gridvals(:,2), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            elseif l_aprime==3
                Values=arrayfun(FnsToEvaluate{ff}, PolicyValues_d(:,:,:,1), PolicyValues_d(:,:,:,2), PolicyValues_d(:,:,:,3), PolicyValues_d(:,:,:,4), PolicyValues_aprime(:,:,:,1), PolicyValues_aprime(:,:,:,2), PolicyValues_aprime(:,:,:,3), a_gridvals(:,1), a_gridvals(:,2), a_gridvals(:,3), z_gridvals_J_fastOLG(1,:,:,1), z_gridvals_J_fastOLG(1,:,:,2), z_gridvals_J_fastOLG(1,:,:,3), z_gridvals_J_fastOLG(1,:,:,4), z_gridvals_J_fastOLG(1,:,:,5), z_gridvals_J_fastOLG(1,:,:,6), z_gridvals_J_fastOLG(1,:,:,7), z_gridvals_J_fastOLG(1,:,:,8), ParamCell{:});
            end
        end
    end

    if outputasstructure==1
        AggVars.(AggVarNames{ff}).Mean=sum(Values(:).*AgentDist(:));
    else % outputasstructure==0
        AggVars(ff)=sum(Values(:).*AgentDist(:));
    end
end


end
