function [n_semizze,semizze_gridvals_J,N_semizze,l_semizze,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters)
% Note: removes n_e from simoptions (if it is there)
% Note: removes n_semiz from simoptions (if it is there)

% FnsToEvaluate commands do not create n_e and n_semiz
N_semiz=prod(simoptions.n_semiz);
N_z=prod(n_z);
N_e=prod(simoptions.n_e);


%%
if simoptions.alreadygridvals==0
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    [z_gridvals_J, ~, simoptions]=ExogShockSetup_FHorz(n_z,z_grid,[],N_j,Parameters,simoptions,1);
    % note: output z_gridvals_J, pi_z_J, and simoptions.e_gridvals_J, simoptions.pi_e_J
    %
    % size(z_gridvals_J)=[prod(n_z),length(n_z),N_j]
    % size(pi_z_J)=[prod(n_z),prod(n_z),N_j]
    % size(e_gridvals_J)=[prod(n_e),length(n_e),N_j]
    % size(pi_e_J)=[prod(n_e),N_j]
    % If no z, then z_gridvals_J=[] and pi_z_J=[]
    % If no e, then e_gridvals_J=[] and pi_e_J=[]
else
    z_gridvals_J=z_grid;
end
if simoptions.alreadygridvals_semiexo==0
    if N_semiz>0 % If using semi-exogenous shocks
        simoptions=SemiExogShockSetup_FHorz([],N_j,simoptions.d_grid,Parameters,simoptions,1);
        % user may input simoptions.semiz_grid or simoptions.semiz_gridvals_J
    end
end

%% Create semizze_gridvals_J (which will combine all three of semiz, z, and e)

if N_semiz==0
    if N_z==0 
        if N_e==0 % no semiz, no z, no e
            n_semizze=0;
            semizze_gridvals_J=[];
        else % no semiz, no z, e
            n_semizze=simoptions.n_e;
            semizze_gridvals_J=simoptions.e_gridvals_J;
        end
    else
        if N_e==0 % no semiz, z, no e
            n_semizze=n_z;
            semizze_gridvals_J=z_gridvals_J;
        else % no semiz, z, e
            n_semizze=[n_z,simoptions.n_e];
            semizze_gridvals_J=[repmat(z_gridvals_J,N_e,1),repelem(simoptions.e_gridvals_J,N_z,1)];
        end
    end
else
    if N_z==0 
        if N_e==0 % semiz, no z, no e
            n_semizze=simoptions.n_semiz;
            semizze_gridvals_J=simoptions.semiz_gridvals_J;
        else % semiz, no z, e
            n_semizze=[simoptions.n_semiz,simoptions.n_e];
            semizze_gridvals_J=[repmat(simoptions.semiz_gridvals_J,N_e,1),repelem(simoptions.e_gridvals_J,N_semiz,1)];
        end
    else
        if N_e==0 % semiz, z, no e
            n_semizze=[simoptions.n_semiz,n_z];
            semizze_gridvals_J=[repmat(simoptions.semiz_gridvals_J,N_z,1),repelem(z_gridvals_J,N_semiz,1)];
        else % semiz, z, e
            n_semizze=[simoptions.n_semiz,n_z,simoptions.n_e];
            semizze_gridvals_J=[repmat(simoptions.semiz_gridvals_J,N_z*N_e,1),repelem(repmat(z_gridvals_J,N_e,1),N_semiz,1),repelem(simoptions.e_gridvals_J,N_semiz*N_z,1)];
        end
    end
end


%%
N_semizze=prod(n_semizze);
if prod(n_semizze)==0
    l_semizze=0;
else
    l_semizze=length(n_semizze);
end

%% Clean up
if isfield(simoptions,'n_e')
    simoptions=rmfield(simoptions,'n_e'); % From now on, e is just treated as part of z (for rest of EvalFnOnAgentDist)
end
if isfield(simoptions,'n_semiz')
    simoptions=rmfield(simoptions,'n_semiz'); % From now on, semiz is just treated as part of z (for rest of EvalFnOnAgentDist)
end

end
