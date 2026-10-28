function [semiz_gridvals, pi_semiz, transpathoptions, options]=SemiExogShockSetup_InfHorz_TPath(n_d,d_grid,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,options,gridpiboth)
% Set up the semi-exogenous state semiz for the InfHorz transition path.
% Internally calls SemiExogShockSetup_InfHorz to build the grid/transitions (on a COPY of options, as that command
% removes SemiExoStateFn), and does the TPath-specific work: checking whether the semiz transitions vary over the
% path, and if so building transpathoptions.pi_semiz_T.
% Handles only semiz: combining semiz with z and e (bothz=(semiz,z), semizze=(semiz,z,e), semiz fastest) is done by
% the caller.
% options is whichever options struct carries the semiz setup (n_semiz, SemiExoStateFn or pi_semiz, semiz_grid,
% l_dsemiz): vfoptions for value-fn callers, simoptions for agent-dist callers.
% Unlike ExogShockSetup_InfHorz_TPath this does NOT remove any field from options, so it can be called again
% on the same options struct.
%
% gridpiboth=4: (trans path GE) we want both grid and transition probabilities
% gridpiboth=3: (value fn iter) we want both grid and transition probabilities
% gridpiboth=2: (agent dist)    we want just transition probabilities (on cpu)
% gridpiboth=1: (FnsToEvaluate) we want just grid
%
% Inputs:
%   n_d, d_grid, Parameters:   as for SemiExogShockSetup_InfHorz (d_grid stacked-column, or joint-grid [N_d,l_d])
%   PricePathNames:            SemiExoStateFn depending on a PricePath parameter is an error (not implemented)
%   ParamPath, ParamPathNames, ParamPathSizeVec, T:
%                              ParamPath is the T-by-'number of path parameters' matrix; used to evaluate
%                              SemiExoStateFn period-by-period when it depends on a ParamPath parameter
%
% Outputs ([] when not requested by gridpiboth, or when there is no semiz):
%   semiz_gridvals:   [N_semiz,l_semiz] (gpu)                       (gridpiboth=1,3,4)
%   pi_semiz:         [N_semiz,N_semiz,N_dsemiz] (semiz,semiz',d2)  (gridpiboth=2,3,4); gpu, except cpu for gridpiboth=2
%                     When semizpathtrivial=0 this is the t=1 slice of transpathoptions.pi_semiz_T (a placeholder).
%   transpathoptions:
%     .semizpathtrivial=1, or =0 when SemiExoStateFn depends on a ParamPath parameter or options.pi_semiz is
%       given 4-D [N_semiz,N_semiz,N_dsemiz,T] (only checked for gridpiboth=2,3,4)
%     .pi_semiz_T  [N_semiz,N_semiz,N_dsemiz,T], when semizpathtrivial=0 (gpu; cpu for gridpiboth=2).
%       Slice t is the transition from t to t+1.
%   options:
%     .l_dsemiz defaulted (=1)
%     .setup_semiexo with fields n_d1, n_d2, l_d1, l_d2, N_d1, N_dsemiz, d1_gridvals, d2_gridvals (the last l_dsemiz
%       decision variables are d2, they drive pi_semiz; any others are d1; d1_gridvals is [] without d1)

semiz_gridvals=[];
pi_semiz=[];
transpathoptions.semizpathtrivial=1; % will be overwritten if appropriate

if ~isfield(options,'n_semiz')
    return
end
if prod(options.n_semiz)==0
    return
end

if ~isfield(options,'l_dsemiz')
    options.l_dsemiz=1; % by default, only one decision variable influences the semi-exogenous state
end

if isempty(n_d) || prod(n_d)==0
    error('Semi-exogenous states (n_semiz) require a decision variable: the last options.l_dsemiz decision variables determine the semiz transitions')
end
if options.l_dsemiz>length(n_d)
    error('options.l_dsemiz is larger than the number of decision variables')
end

N_semiz=prod(options.n_semiz);
l_semiz=length(options.n_semiz);

%% Split decision variables: d1 (standard) and d2 (the last l_dsemiz, they drive the semi-exogenous transition)
l_d=length(n_d);
l_d2=options.l_dsemiz;
n_d2=n_d(end-l_d2+1:end);
N_dsemiz=prod(n_d2);
if l_d>l_d2
    n_d1=n_d(1:end-l_d2);
    l_d1=length(n_d1);
else
    n_d1=0;
    l_d1=0;
end
N_d1=prod(n_d1);
% Joint grid on d (d1 varying first, then d2)
if all(size(d_grid)==[sum(n_d),1])
    d_gridvals=CreateGridvals(n_d,gpuArray(d_grid),1);
    dgridisstacked=1;
elseif all(size(d_grid)==[prod(n_d),l_d])
    d_gridvals=gpuArray(d_grid);
    dgridisstacked=0;
else
    error('d_grid is not the correct shape [should be stacked-column of size sum(n_d)-by-1, or a joint-grid of size prod(n_d)-by-length(n_d)]')
end
if N_d1==0
    d1_gridvals=[];
    d2_gridvals=d_gridvals;
else
    d1_gridvals=d_gridvals(1:N_d1,1:l_d1);
    d2_gridvals=d_gridvals(1:N_d1:end,l_d1+1:end);
end

% SemiExogShockSetup_InfHorz reads the d2 grid as d_grid(sum(n_d1)+1:sum(n_d)), so needs a stacked-column d_grid
if isfield(options,'SemiExoStateFn') && dgridisstacked==0
    if l_d==1
        d_grid=d_grid(:); % for a single decision variable the two shapes coincide
    else
        error('Semi-exogenous states with SemiExoStateFn on a transition path need d_grid as a stacked column of size sum(n_d)-by-1 (not a joint grid)')
    end
end

%% Check whether the semiz transitions vary over the path
optionsinner=options; % SemiExogShockSetup_InfHorz removes SemiExoStateFn from its options, so always work on a copy
if gridpiboth==2 || gridpiboth==3 || gridpiboth==4 % transition probabilities are wanted
    if isfield(options,'SemiExoStateFn')
        temp=getAnonymousFnInputNames(options.SemiExoStateFn);
        nargsSemiExo=2*l_semiz+options.l_dsemiz; % inputs are (semiz,semizprime,dsemiz,...)
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
                error('Not yet implemented to use SemiExoStateFn which includes parameters from PricePath (email me if you want this)')
            end
            % Next, check if SemiExoStateFn depends on a ParamPath parameter
            overlap2=0;
            for kk=1:length(SemiExoStateFnParamNames)
                if any(strcmp(SemiExoStateFnParamNames{kk},ParamPathNames))
                    overlap2=1;
                end
            end
            if overlap2==1
                transpathoptions.semizpathtrivial=0; % pi_semiz varies over the path
                % Evaluate SemiExoStateFn period-by-period on the path (the PricePath is irrelevant here)
                transpathoptions.pi_semiz_T=zeros(N_semiz,N_semiz,N_dsemiz,T); % built on cpu, moved to gpu below if wanted
                for tt=1:T
                    Parameters_tt=Parameters;
                    for pp=1:length(ParamPathNames)
                        Parameters_tt.(ParamPathNames{pp})=ParamPath(tt,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp));
                    end
                    optionsinner_tt=SemiExogShockSetup_InfHorz(n_d,d_grid,Parameters_tt,options,2); % a copy of options, so options keeps its SemiExoStateFn
                    transpathoptions.pi_semiz_T(:,:,:,tt)=gather(optionsinner_tt.pi_semiz);
                end
                if gridpiboth~=2
                    transpathoptions.pi_semiz_T=gpuArray(transpathoptions.pi_semiz_T);
                end
            end
        end
    elseif isfield(options,'pi_semiz')
        if ndims(options.pi_semiz)==4 || (N_dsemiz==1 && ndims(options.pi_semiz)==3 && size(options.pi_semiz,3)==T && T>1)
            % User-supplied time-varying pi_semiz [N_semiz,N_semiz,N_dsemiz,T]
            % (with N_dsemiz=1 MATLAB drops the singleton, so [N_semiz,N_semiz,T] is read as [N_semiz,N_semiz,1,T])
            if ndims(options.pi_semiz)==4
                if ~all(size(options.pi_semiz)==[N_semiz,N_semiz,N_dsemiz,T])
                    error('options.pi_semiz is 4-D but its size does not match [N_semiz, N_semiz, N_dsemiz, T]')
                end
            end
            transpathoptions.semizpathtrivial=0; % pi_semiz varies over the path
            if gridpiboth==2
                transpathoptions.pi_semiz_T=gather(reshape(options.pi_semiz,[N_semiz,N_semiz,N_dsemiz,T])); % agent distribution iteration is performed on cpu
            else
                transpathoptions.pi_semiz_T=gpuArray(reshape(options.pi_semiz,[N_semiz,N_semiz,N_dsemiz,T]));
            end
            optionsinner.pi_semiz=gather(transpathoptions.pi_semiz_T(:,:,:,1)); % placeholder for the SemiExogShockSetup_InfHorz call below (which errors on 4-D input)
        end
    end
end

%% Build semiz_gridvals and/or pi_semiz (d2-dependent) per gridpiboth
if gridpiboth==4
    optionsinner=SemiExogShockSetup_InfHorz(n_d,d_grid,Parameters,optionsinner,3);
else
    optionsinner=SemiExogShockSetup_InfHorz(n_d,d_grid,Parameters,optionsinner,gridpiboth);
end
if gridpiboth~=2
    semiz_gridvals=gpuArray(optionsinner.semiz_gridvals); % [N_semiz,l_semiz]
end
if gridpiboth~=1
    if transpathoptions.semizpathtrivial==0
        pi_semiz=transpathoptions.pi_semiz_T(:,:,:,1); % placeholder, the per-period arrays are in transpathoptions.pi_semiz_T
    else
        pi_semiz=optionsinner.pi_semiz; % [N_semiz,N_semiz,N_dsemiz]
    end
    if gridpiboth==2
        pi_semiz=gather(pi_semiz); % agent distribution iteration is performed on cpu
    else
        pi_semiz=gpuArray(pi_semiz);
    end
end

%% Stash the decision-variable split (read by the TPath substeps)
setup_semiexo.n_d1=n_d1;
setup_semiexo.n_d2=n_d2;
setup_semiexo.l_d1=l_d1;
setup_semiexo.l_d2=l_d2;
setup_semiexo.N_d1=N_d1;
setup_semiexo.N_dsemiz=N_dsemiz;
setup_semiexo.d1_gridvals=d1_gridvals;
setup_semiexo.d2_gridvals=d2_gridvals;
options.setup_semiexo=setup_semiexo;

end
