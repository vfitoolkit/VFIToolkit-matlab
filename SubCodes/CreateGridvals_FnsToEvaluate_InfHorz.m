function [n_z,z_gridvals,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_InfHorz(n_z,z_grid,simoptions,Parameters)
% Note: removes n_e from simoptions (if it is there)
% Note: removes n_semiz from simoptions (if it is there)

% FnsToEvaluate commands do not create n_e and n_semiz
if ~isfield(simoptions,'n_e')
    simoptions.n_e=0;
end
if ~isfield(simoptions,'n_semiz')
    simoptions.n_semiz=0;
end
if ~isfield(simoptions,'alreadygridvals')
    simoptions.alreadygridvals=0;
end
if ~isfield(simoptions,'alreadygridvals_semiexo')
    simoptions.alreadygridvals_semiexo=0;
end


%% Create z_gridvals (which will combine all three of semiz, z, and e)

% First z
if simoptions.alreadygridvals==0
    [z_gridvals, ~, simoptions]=ExogShockSetup_InfHorz(n_z,z_grid,[],Parameters,simoptions,1);
elseif simoptions.alreadygridvals==1
    z_gridvals=z_grid;
end
% n_z
N_z=prod(n_z);

% Now e (which ExogShockSetup_InfHorz() created in simoptions)
if prod(simoptions.n_e)>0
    % Now put e into z as that is easiest way to handle it from now on
    if N_z==0
        z_gridvals=simoptions.e_gridvals;
        n_z=simoptions.n_e;
        N_z=prod(n_z);
    else
        z_gridvals=[repmat(z_gridvals,prod(simoptions.n_e),1),repelem(simoptions.e_gridvals,N_z,1)];
        n_z=[n_z,simoptions.n_e];
        N_z=prod(n_z);
    end
    simoptions=rmfield(simoptions,'n_e'); % From now on, e is just treated as part of z (for rest of EvalFnOnAgentDist)
end

if simoptions.alreadygridvals_semiexo==0
    % Also semiz if that is used
    if prod(simoptions.n_semiz)>0 % If using semi-exogenous shocks
        simoptions=SemiExogShockSetup_InfHorz([],simoptions.d_grid,Parameters,simoptions,2,1);
        % user may input simoptions.semiz_grid or simoptions.semiz_gridvals

        if N_z==0
            z_gridvals=simoptions.semiz_gridvals;
            n_z=simoptions.n_semiz;
            N_z=prod(n_z);
        else
            % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
            z_gridvals=[repmat(simoptions.semiz_gridvals,N_z,1),repelem(z_gridvals,prod(simoptions.n_semiz),1)];
            n_z=[simoptions.n_semiz,n_z];
            N_z=prod(n_z);
        end
    end
    if isfield(simoptions,'n_semiz')
        simoptions=rmfield(simoptions,'n_semiz'); % From now on, semiz is just treated as part of z (for rest of EvalFnOnAgentDist)
    end
else
    if N_z==0
        z_gridvals=simoptions.semiz_gridvals;
        n_z=simoptions.n_semiz;
        N_z=prod(n_z);
    else
        % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
        z_gridvals=[repmat(simoptions.semiz_gridvals,N_z,1),repelem(z_gridvals,prod(simoptions.n_semiz),1)];
        n_z=[simoptions.n_semiz,n_z];
        N_z=prod(n_z);
    end
    simoptions=rmfield(simoptions,'n_semiz'); % From now on, semiz is just treated as part of z (for rest of EvalFnOnAgentDist)
end


%% Clean up
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end

end
