function [semiz_gridvals_J, pi_semiz_J, pi_semiz_J_sim, transpathoptions, vfoptions, simoptions]=SemiExogShockSetup_FHorz_TPath(n_d,N_j,d_grid,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,vfoptions,simoptions,gridpiboth)
% Set up the semi-exogenous state semiz for the FHorz transition path.
% Internally calls SemiExogShockSetup_FHorz to build the grids/transitions, then does the
% TPath-specific work: the semizgridsinGE/semizpathtrivial checks, building the time-varying
% transpathoptions.semiz_gridvals_J_T and transpathoptions.pi_semiz_J_T when semizpathtrivial=0,
% stashing setup_semiexo in vfoptions/simoptions, and (when transpathoptions.fastOLG=1)
% re-orienting semiz_gridvals_J and pi_semiz_J (including appending the j=N_j zero transition
% row) so the fastOLG SingleStep value fn raws receive ready-made arrays.
% Handles only semiz: combining semiz with z and e (bothz/bothze/semizze) is done by the caller,
% just as for the baseline FHorz commands.
% The vfoptions input/output is whichever options struct carries the semiz setup (n_semiz, SemiExoStateFn or
% pi_semiz, semiz_grid, l_dsemiz): vfoptions for value-fn callers, simoptions for agent-dist callers
% (gridpiboth=2; pass a throwaway struct() in the simoptions slot in that case).
%
% gridpiboth=4: sometimes (trans path GE) we want both grid and transition probabilities, including pi_semiz_J_sim alternative transition probs
% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilities
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilities, including pi_semiz_J_sim alternative transition probs
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid
%
% Inputs:
%   n_d, N_j, d_grid, Parameters:      as in SemiExogShockSetup_FHorz
%   PricePathNames:                    used for the semizgridsinGE check
%   ParamPath, ParamPathNames, ParamPathSizeVec, T:
%                                      used for the semizpathtrivial check, and to evaluate
%                                      SemiExoStateFn period-by-period when semizpathtrivial=0
%                                      (ParamPath is the T-by-'number of path parameters' matrix)
%
% Outputs (positional outputs are the value-function-oriented ones, except pi_semiz_J_sim; each is []
% when not requested by gridpiboth, or when there is no semiz):
%   semiz_gridvals_J:                  (gridpiboth=1,3,4)
%     [N_semiz, l_semiz, N_j]                     if transpathoptions.fastOLG=0
%     [N_j, N_semiz, l_semiz]                     if transpathoptions.fastOLG=1
%   pi_semiz_J:                        (gridpiboth=2,3,4)
%     [N_semiz, N_semiz, N_dsemiz, N_j-1]         if transpathoptions.fastOLG=0, or gridpiboth=2
%                                                 (semiz, semiz', d2, j); slice jj is the transition from
%                                                 period jj to jj+1; on cpu for gridpiboth=2 (agent
%                                                 distribution iteration is performed on cpu)
%     [N_j, N_semiz, N_semiz, N_dsemiz]           if transpathoptions.fastOLG=1 with gridpiboth=3,4
%                                                 (j, semiz', semiz, d2); the j=N_j slice is an appended row
%                                                 of zeros: there is no continuation value in the final
%                                                 period (vfoptions.EVpre=0, which TPath always uses). [A
%                                                 Matched Expectations Path / EVpre=1 caller must NOT route
%                                                 through this setup: MEP needs a genuine (non-zero) final
%                                                 transition row, supplied by the MEP code itself along with
%                                                 the fastOLG orientation.]
%   pi_semiz_J_sim:                    (gridpiboth=2,4)
%     [N_semiz, N_semiz, N_dsemiz, N_j-1]         standard form for the agent distribution, which always
%                                                 uses it regardless of fastOLG (the dist raws do their own
%                                                 sparsity trick); on cpu for gridpiboth=2, on gpu for
%                                                 gridpiboth=4
%     When semizpathtrivial=0 the pi outputs are placeholders (evaluated at the Parameters as input); the
%     per-period arrays are in transpathoptions.pi_semiz_J_T.
%   transpathoptions:
%     .semizgridsinGE=1 if SemiExoStateFn depends on a PricePath parameter (so the semiz transitions
%       would need to be recomputed every GE iteration; currently errors), else 0
%     .semizpathtrivial=0 if SemiExoStateFn depends on a ParamPath parameter, or a 5-D vfoptions.pi_semiz
%       was supplied, so pi_semiz_J varies over the path; else 1 (only checked for gridpiboth=2,3,4)
%     .semiz_gridvals_J_T   (when semizpathtrivial=0 and gridpiboth=3,4)
%       [N_semiz, l_semiz, N_j, T]                if transpathoptions.fastOLG=0
%       [N_j, N_semiz, l_semiz, T]                if transpathoptions.fastOLG=1
%       (the semiz grid itself cannot vary over the path -- semiz_grid is a fixed input and SemiExoStateFn
%       only determines transitions -- so this is the same grid repeated over T, provided for interface
%       uniformity with transpathoptions.z_gridvals_J_T)
%     .pi_semiz_J_T         (when semizpathtrivial=0); slice tt is the pi_semiz_J applying at path period tt
%       [N_semiz, N_semiz, N_dsemiz, N_j-1, T]    if transpathoptions.fastOLG=0, or gridpiboth=2 (on cpu)
%       [N_j, N_semiz, N_semiz, N_dsemiz, T]      if transpathoptions.fastOLG=1 with gridpiboth=3,4 (j=N_j
%                                                 slice is the appended zero row, as for pi_semiz_J)
%     .pi_semiz_J_sim_T     (when semizpathtrivial=0 and gridpiboth=2,4): standard-form copy of pi_semiz_J_T
%       for the agent distribution, [N_semiz, N_semiz, N_dsemiz, N_j-1, T], unaffected by fastOLG (on cpu for
%       gridpiboth=2); slice tt is the pi_semiz_J applying at path period tt
%   vfoptions:
%     .l_dsemiz defaulted, .setup_semiexo, and from SemiExogShockSetup_FHorz (per gridpiboth):
%     .semiz_gridvals_J [N_semiz,l_semiz,N_j] and .pi_semiz_J [N_semiz,N_semiz,N_dsemiz,N_j-1] (both
%     standard form)
%   simoptions:
%     .l_dsemiz, .setup_semiexo, .d_grid, and standard-form copies .semiz_gridvals_J and .pi_semiz_J
%     of whichever were built (the agent distribution iteration always uses the standard form)

transpathoptions.semizgridsinGE=0; % will be overwritten if appropriate
transpathoptions.semizpathtrivial=1;  % will be overwritten if appropriate

if prod(vfoptions.n_semiz)==0
    % No semiz: pass-through defaults so the shooting inputs are always defined
    semiz_gridvals_J=[];
    pi_semiz_J=[];
    pi_semiz_J_sim=[];
    return
end

if ~isfield(vfoptions,'l_dsemiz')
    vfoptions.l_dsemiz=1;
end
simoptions.l_dsemiz=vfoptions.l_dsemiz;
% Split decision variables: d1 (standard) and d2 (drives the semi-exogenous transition)
if length(n_d)>vfoptions.l_dsemiz
    n_d1=n_d(1:end-vfoptions.l_dsemiz); d1_grid=d_grid(1:sum(n_d1));
else
    n_d1=0; d1_grid=[];
end
n_d2=n_d(end-vfoptions.l_dsemiz+1:end); d2_grid=d_grid(sum(n_d1)+1:end);

N_semiz=prod(vfoptions.n_semiz);
N_dsemiz=prod(n_d2);

if gridpiboth==2 || gridpiboth==3 || gridpiboth==4 % transition probabilities are wanted
    % Check if SemiExoStateFn depends on a PricePath or ParamPath parameter
    if isfield(vfoptions,'SemiExoStateFn')
        temp=getAnonymousFnInputNames(vfoptions.SemiExoStateFn);
        nargsSemiExo=2*length(vfoptions.n_semiz)+vfoptions.l_dsemiz; % inputs are (semiz,semizprime,dsemiz,...)
        if length(temp)>nargsSemiExo
            SemiExoStateFnParamNames={temp{nargsSemiExo+1:end}};
            % First, check if SemiExoStateFn depends on a PricePath parameter
            overlap=0;
            for kk=1:length(SemiExoStateFnParamNames)
                if any(strcmp(SemiExoStateFnParamNames{kk},PricePathNames))
                    overlap=1;
                end
            end
            if overlap==1
                transpathoptions.semizgridsinGE=1;
                transpathoptions.semizpathtrivial=0; % pi_semiz_J varies over the path
                error('Not yet implemented to use SemiExoStateFn which includes parameters from PricePath (email me if you want this)')
            else % overlap==0
                % Next, check if SemiExoStateFn depends on a ParamPath parameter
                overlap2=0;
                for kk=1:length(SemiExoStateFnParamNames)
                    if any(strcmp(SemiExoStateFnParamNames{kk},ParamPathNames))
                        overlap2=1;
                    end
                end
                if overlap2==1
                    transpathoptions.semizpathtrivial=0; % pi_semiz_J varies over the path
                    % Build transpathoptions.pi_semiz_J_T: evaluate SemiExoStateFn period-by-period on the path
                    % Note, we know the PricePath is irrelevant for the current purpose
                    if gridpiboth==2
                        transpathoptions.pi_semiz_J_T=zeros(N_semiz,N_semiz,N_dsemiz,N_j-1,T); % agent distribution iteration is performed on cpu
                    else
                        transpathoptions.pi_semiz_J_T=zeros(N_semiz,N_semiz,N_dsemiz,N_j-1,T,'gpuArray');
                    end
                    for tt=1:T
                        Parameters_tt=Parameters;
                        for pp=1:length(ParamPathNames)
                            Parameters_tt.(ParamPathNames{pp})=ParamPath(tt,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp));
                        end
                        if gridpiboth==2
                            vfoptions_tt=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters_tt,vfoptions,2);
                        else
                            vfoptions_tt=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters_tt,vfoptions,3);
                        end
                        transpathoptions.pi_semiz_J_T(:,:,:,:,tt)=vfoptions_tt.pi_semiz_J;
                    end
                end
            end
        end
    elseif isfield(vfoptions,'pi_semiz')
        if ndims(vfoptions.pi_semiz)>4
            transpathoptions.semizpathtrivial=0; % pi_semiz_J varies over the path
            % User-supplied time-varying pi_semiz: T always comes after N_j (the final-period slice is never
            % read on the transition path and can be omitted)
            if all(size(vfoptions.pi_semiz)==[N_semiz,N_semiz,N_dsemiz,N_j,T])
                transpathoptions.pi_semiz_J_T=gpuArray(vfoptions.pi_semiz(:,:,:,1:N_j-1,:)); % drop the (never-read) final-period slice
            elseif all(size(vfoptions.pi_semiz)==[N_semiz,N_semiz,N_dsemiz,N_j-1,T])
                transpathoptions.pi_semiz_J_T=gpuArray(vfoptions.pi_semiz);
            else
                error('vfoptions.pi_semiz is 5D but size does not match [N_semiz, N_semiz, N_dsemiz, N_j (or N_j-1), T]')
            end
            if gridpiboth==2
                transpathoptions.pi_semiz_J_T=gather(transpathoptions.pi_semiz_J_T); % agent distribution iteration is performed on cpu
            end
            vfoptions.pi_semiz=gather(transpathoptions.pi_semiz_J_T(:,:,:,:,1)); % placeholder for the SemiExogShockSetup_FHorz call below (which errors on 5D input)
        end
    end
end

% Build semiz_gridvals_J and/or pi_semiz_J (d2-dependent) per gridpiboth
if gridpiboth==4
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3); % vfoptions.semiz_gridvals_J [N_semiz,l_semiz,N_j], vfoptions.pi_semiz_J [N_semiz,N_semiz',N_dsemiz,N_j-1]
else
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,gridpiboth);
end
% Stash for the substeps (read by Step1/Step2/Step3tt/Step4tt SemiExo variants)
setup_semiexo.n_d1=n_d1;
setup_semiexo.n_d2=n_d2;
setup_semiexo.N_dsemiz=N_dsemiz;
setup_semiexo.d1_gridvals=CreateGridvals(n_d1,d1_grid,1);
setup_semiexo.d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
% store setup_semiexo in vfoptions and simoptions
vfoptions.setup_semiexo=setup_semiexo;
simoptions.setup_semiexo=setup_semiexo;
% simoptions needs some extra info (the agent distribution always uses the standard-form pi_semiz_J)
simoptions.d_grid=d_grid;
if gridpiboth~=2 % grid was built
    simoptions.semiz_gridvals_J=vfoptions.semiz_gridvals_J;
end
if gridpiboth~=1 % transition probabilities were built
    simoptions.pi_semiz_J=vfoptions.pi_semiz_J;
end

if transpathoptions.semizpathtrivial==0 && gridpiboth~=2
    % The semiz grid itself cannot vary over the path (semiz_grid is a fixed input; SemiExoStateFn only
    % determines transitions): repeat it over T for interface uniformity with z_gridvals_J_T
    transpathoptions.semiz_gridvals_J_T=repmat(vfoptions.semiz_gridvals_J,1,1,1,T); % [N_semiz,l_semiz,N_j,T]
end

% Positional outputs
semiz_gridvals_J=[];
pi_semiz_J=[];
pi_semiz_J_sim=[];
if gridpiboth~=2
    semiz_gridvals_J=vfoptions.semiz_gridvals_J; % [N_semiz,l_semiz,N_j]
end
if gridpiboth~=1
    pi_semiz_J=vfoptions.pi_semiz_J; % [N_semiz,N_semiz',N_dsemiz,N_j-1] (on cpu for gridpiboth=2)
end
if gridpiboth==2 || gridpiboth==4
    pi_semiz_J_sim=vfoptions.pi_semiz_J; % standard form for the agent distribution
    if transpathoptions.semizpathtrivial==0
        transpathoptions.pi_semiz_J_sim_T=transpathoptions.pi_semiz_J_T; % standard-form copy for the agent distribution [N_semiz,N_semiz,N_dsemiz,N_j-1,T] (taken here, before the fastOLG re-orientation of pi_semiz_J_T below)
    end
end

if transpathoptions.fastOLG==1
    % Re-orient for the fastOLG SingleStep value fn raws, which form expectations vectorized over age
    if gridpiboth~=2
        semiz_gridvals_J=permute(semiz_gridvals_J,[3,1,2]); % fastOLG form (N_j,N_semiz,l_semiz)
    end
    if gridpiboth==3 || gridpiboth==4
        pi_semiz_J=permute(pi_semiz_J,[4,2,1,3]); % (j,semiz',semiz,d2)
        pi_semiz_J=cat(1,pi_semiz_J,zeros(1,N_semiz,N_semiz,N_dsemiz,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (vfoptions.EVpre=0; a MEP/EVpre=1 caller must instead supply a genuine final transition row, and must not route through this setup)
        if transpathoptions.semizpathtrivial==0
            transpathoptions.semiz_gridvals_J_T=permute(transpathoptions.semiz_gridvals_J_T,[3,1,2,4]); % fastOLG form (N_j,N_semiz,l_semiz,T)
            transpathoptions.pi_semiz_J_T=permute(transpathoptions.pi_semiz_J_T,[4,2,1,3,5]); % (j,semiz',semiz,d2,t)
            transpathoptions.pi_semiz_J_T=cat(1,transpathoptions.pi_semiz_J_T,zeros(1,N_semiz,N_semiz,N_dsemiz,T,'gpuArray')); % append the j=N_j zero row to every period's slice
        end
    end
end


end
