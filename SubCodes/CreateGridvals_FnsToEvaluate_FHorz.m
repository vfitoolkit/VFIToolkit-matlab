function [n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters)
% Note: removes n_e from simoptions (if it is there)
% Note: removes n_semiz from simoptions (if it is there)

%% Create z_gridvals_J (which will combine all three of semiz, z, and e)

% First z
if simoptions.alreadygridvals==0
    [z_gridvals_J, ~, simoptions]=ExogShockSetup_FHorz(n_z,z_grid,[],N_j,Parameters,simoptions,1);
elseif simoptions.alreadygridvals==1
    z_gridvals_J=z_grid;
end
% n_z
N_z=prod(n_z);

% Now e (which ExogShockSetup_FHorz() created in simoptions)
if isfield(simoptions,'n_e')
    % Now put e into z as that is easiest way to handle it from now on
    if N_z==0
        z_gridvals_J=simoptions.e_gridvals_J;
        n_z=simoptions.n_e;
        N_z=prod(n_z);
    else
        z_gridvals_J=[repmat(z_gridvals_J,prod(simoptions.n_e),1),repelem(simoptions.e_gridvals_J,N_z,1)];
        n_z=[n_z,simoptions.n_e];
        N_z=prod(n_z);
    end
    simoptions=rmfield(simoptions,'n_e'); % From now on, e is just treated as part of z (for rest of EvalFnOnAgentDist)
end

% Also semiz if that is used
if isfield(simoptions,'n_semiz') % If using semi-exogenous shocks
    simoptions=SemiExogShockSetup_FHorz([],N_j,simoptions.d_grid,Parameters,simoptions,2);
    % user may input simoptions.semiz_grid or simoptions.semiz_gridvals_J

    if N_z==0
        z_gridvals_J=simoptions.semiz_gridvals_J;
        n_z=simoptions.n_semiz;
        N_z=prod(n_z);
    else
        % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
        z_gridvals_J=[repmat(simoptions.semiz_gridvals_J,N_z,1),repelem(z_gridvals_J,prod(simoptions.n_semiz),1)];
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
