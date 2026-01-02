function SimPanelValues=SimPanelValues_TransPath_InfHorz(PolicyPath, PricePath, ParamPath, T, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, FnsToEvaluate, Parameters, simoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 

% Simulates a panel based on PolicyPath of 'numbersims' agents of length
% 'T' beginning from randomly drawn InitialDist.
% SimPanelValues is a 3-dimensional matrix with first dimension being the
% number of 'variables' to be simulated, second dimension is FHorz, and
% third dimension is the number-of-simulations
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-T, or n_a-by-n_z)

%% Check which simoptions have been declared, set all others to defaults 
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.verbose=0;
    simoptions.simperiods=T;
    simoptions.numbersims=10^3;
    simoptions.lowmemory=0; % setting to 1 slows the simulations, but reduces memory
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.experienceasset=0;
    simoptions.experienceassetu=0; % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    % For internal use only
    simoptions.keepoutputasmatrix=0;
    simoptions.simpanelindexkron=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'simperiods')
        simoptions.simperiods=T; % This can be made shorter, but not longer
    end
    if ~isfield(simoptions,'numbersims')
        simoptions.numbersims=10^3;
    end
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0; % setting to 1 slows the simulations, but reduces memory
    end
    % Model setup
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    simoptions.experienceassetu=0; % cannot actually be used in InfHorz, but need to set for some sub-commands
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions, 'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    % For internal use only
    if ~isfield(simoptions,'keepoutputasmatrix')
        simoptions.keepoutputasmatrix=0;
    end
    if ~isfield(simoptions,'simpanelindexkron')
        simoptions.simpanelindexkron=0;
    end
end

if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
    simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
end

simoptions.simperiods=gather(simoptions.simperiods);
simoptions.numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()

%% Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);


%%
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

N_d=prod(n_d);
N_a=prod(n_a);
% N_semizze set below

%% Simulate Panel Indexes
d_grid=gather(d_grid);
a_grid=gather(a_grid);

N_semiz=prod(simoptions.n_semiz);
N_z=prod(n_z);
N_e=prod(simoptions.n_e);

if N_z==0
    if N_e==0
        if N_semiz==0
            n_semizze=0;
        else
            n_semizze=simoptions.n_semiz;
        end
    else
        if N_semiz==0
            n_semizze=simoptions.n_e;
        else
            n_semizze=[simoptions.n_semiz,simoptions.n_e];
        end
    end
else
    if N_e==0
        if N_semiz==0
            n_semizze=n_z;
        else
            n_semizze=[simoptions.n_semiz,n_z];
        end
    else
        if N_semiz==0
            n_semizze=[n_z,simoptions.n_e];
        else
            n_semizze=[simoptions.n_semiz,n_z,simoptions.n_e];
        end
    end
end
N_semizze=prod(n_semizze);

if N_z>0
    if ndims(pi_z)==3
        pi_z_T=pi_z;
    else
        pi_z_T=repelem(pi_z,1,1,T);
    end
else
    pi_z_T=[];
end

if N_e>0
    if size(simoptions.pi_e,2)==T
        simoptions.pi_e_T=simoptions.pi_e;
    else
        simoptions.pi_e_T=repelem(simoptions.pi_e,1,T);
    end
end

%% Simulate the indexes
% Can just pretend this is a standard FHorz model and use that to implement the simulation
SimPanelIndexes=SimPanelIndexes_FHorz(gather(AgentDist_initial),gather(PolicyPath),n_d,n_a,n_z,T,pi_z_T, simoptions);

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate)
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames);
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tminus1AggVarsNames=[];
    tplus1pricePathkk=[];
end

use_tplus1price=0;
if ~isempty(tplus1priceNames)
    use_tplus1price=1;
end
use_tminus1price=0;
if ~isempty(tminus1priceNames)
    use_tminus1price=1;
    for tt=1:length(tminus1priceNames)
        if ~isfield(transpathoptions.initialvalues,tminus1priceNames{tt})
            fprintf('ERROR: Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{tt})
            dbstack
            break
        end
    end
end
use_tminus1AggVars=0;
if ~isempty(tminus1AggVarsNames)
    use_tminus1AggVars=1;
    for tt=1:length(tminus1AggVarsNames)
        if ~isfield(transpathoptions.initialvalues,tminus1AggVarsNames{tt})
            fprintf('ERROR: Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1AggVarsNames{tt})
            dbstack
            break
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

%% Exogenous shock grids (must come after the SimPanelIndexes as it then strips n_semiz and n_e out of simoptions)
% Pretend to be FHorz of length T
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate 
[~,semizze_gridvals_T,~,~,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,T,simoptions,Parameters);
% N_semizze=prod(n_semizze);
if N_semizze==0
    l_semizze=0;
else
    l_semizze=length(n_semizze);
end
% Note: semiz, z and e are from here on all just rolled together in n_z, z_gridvals_J, N_z and l_z


%% Implement new way of handling FnsToEvaluate
% Figure out l_daprime from Policy
l_daprime=size(PolicyPath,1);
if simoptions.gridinterplayer==1
    l_daprime=l_daprime-1;
end

% Note: l_semizze
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_semizze)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_semizze+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    end
end

nFnsToEvalute=length(FnsToEvaluate);


%% Precompute the gridvals vectors.
if N_semizze==0
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,1,T]);
else
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_semizze,T]);
end

% Note that dPolicy and aprimePolicy will depend on age
l_aprime=l_a;
if simoptions.experienceasset==1 || simoptions.experienceassetu==1
    l_aprime=l_aprime-1;
end
if N_d==0
    if N_semizze==0
        daprimePolicy_gridvals=zeros(N_a,l_aprime,T);
    else
        daprimePolicy_gridvals=zeros(N_a*N_semizze,l_aprime,T);
    end
else
    if N_semizze==0
        daprimePolicy_gridvals=zeros(N_a,l_d+l_aprime,T);
    else
        daprimePolicy_gridvals=zeros(N_a*N_semizze,l_d+l_aprime,T);
    end
end

for tt=1:T
    if N_d==0
        [~,aprimePolicy_gridvals_t]=CreateGridvals_Policy(PolicyPath(:,:,:,tt),n_d,n_a,n_a,n_semizze,[],a_grid,simoptions,1, 1); % handles gridinterplayer=1 in here
        daprimePolicy_gridvals(:,:,tt)=aprimePolicy_gridvals_t;
    else
        [dPolicy_gridvals_j,aprimePolicy_gridvals_t]=CreateGridvals_Policy(PolicyPath(:,:,:,tt),n_d,n_a,n_a,n_semizze,d_grid,a_grid,simoptions,1, 1); % handles gridinterplayer=1 in here
        daprimePolicy_gridvals(:,:,tt)=[dPolicy_gridvals_j, aprimePolicy_gridvals_t];
    end
end

a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 at end indicates output as matrices.


%% Now switch everything to gpu so can use arrayfun() to evaluates all the FnsToEvaluate
daprimePolicy_gridvals=gpuArray(daprimePolicy_gridvals);
SimPanelIndexes=gpuArray(SimPanelIndexes);

SimPanelValues=nan(length(FnsToEvaluate), T, simoptions.numbersims,'gpuArray'); % needs to be NaN to permit that some people might be 'born' later than age j=1
% Note, having the whole T at this stage makes assiging the values based on the indexes vastly faster


%% Create PanelValues from PanelIndexes
if N_semizze>0
    for tt=1:T
        SimPanelIndexes_tt=SimPanelIndexes(:,tt,:);

        relevantindices=(~isnan(SimPanelIndexes_tt(1,1,:))); % Note, is just across the ii dimension
        sumrelevantindices=sum(relevantindices);

        % Update Parameters based on t
        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end
        if use_tminus1price==1
            for pp=1:length(tminus1priceNames)
                if tt>1
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                else
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                end
            end
        end
        if use_tplus1price==1
            for pp=1:length(tplus1priceNames)
                kk=tplus1pricePathkk(pp);
                Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
            end
        end
        if use_tminus1AggVars==1
            for pp=1:length(tminus1AggVarsNames)
                if tt>1
                    % The AggVars have not yet been updated, so they still contain previous period values
                    Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
                else
                    Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
                end
            end
        end

        if sumrelevantindices>0 % Does the simulation even contain anyone in period tt?
            currentPanelIndexes_tt=SimPanelIndexes_tt(:,1,relevantindices);
            currentPanelValues_tt=zeros(sumrelevantindices,nFnsToEvalute); % transpose will be taken before storing

            az_ind=squeeze(currentPanelIndexes_tt(1,1,:)+N_a*(currentPanelIndexes_tt(2,1,:)-1));
            % a_ind=currentPanelIndexes_tt(1,1,:);
            % z_ind=currentPanelIndexes_tt(2,1,:); % this is semiz,z,e all together
            % j_ind=currentPanelIndexes_tt(3,1,:);

            a_val=a_gridvals(currentPanelIndexes_tt(1,1,:),:); % a_grid does depend on age
            z_val=semizze_gridvals_T(currentPanelIndexes_tt(2,1,:),:,tt);

            for vv=1:nFnsToEvalute
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                    ParamCell={};
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,tt);
                    ParamCell=num2cell(ValuesFnParamsVec);
                end
                daprime_val=daprimePolicy_gridvals(az_ind,:,tt);
                currentPanelValues_tt(:,vv)=EvalFnOnSimPanelIndex(FnsToEvaluate{vv},ParamCell,daprime_val,a_val,z_val,l_daprime,l_a,l_semizze);
            end
            SimPanelValues(:,tt,relevantindices)=reshape(currentPanelValues_tt',[nFnsToEvalute,1,sumrelevantindices]);

        end
    end
else % N_semizze==0
    for tt=1:T
        SimPanelIndexes_tt=SimPanelIndexes(:,tt,:);

        relevantindices=(~isnan(SimPanelIndexes_tt(1,1,:))); % Note, is just across the ii dimension
        sumrelevantindices=sum(relevantindices);

        % Update Parameters based on t
        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end
        if use_tminus1price==1
            for pp=1:length(tminus1priceNames)
                if tt>1
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                else
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                end
            end
        end
        if use_tplus1price==1
            for pp=1:length(tplus1priceNames)
                kk=tplus1pricePathkk(pp);
                Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
            end
        end
        if use_tminus1AggVars==1
            for pp=1:length(tminus1AggVarsNames)
                if tt>1
                    % The AggVars have not yet been updated, so they still contain previous period values
                    Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
                else
                    Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
                end
            end
        end

        if sumrelevantindices>0 % Does the simulation even contain anyone in period tt?
            currentPanelIndexes_tt=SimPanelIndexes_tt(:,1,relevantindices);
            currentPanelValues_tt=zeros(sumrelevantindices,nFnsToEvalute); % transpose will be taken before storing

            a_ind=squeeze(currentPanelIndexes_tt(1,1,:));
            % j_ind=currentPanelIndexes_tt(2,1,:);

            a_val=a_gridvals(a_ind,:); % a_grid does depend on age

            for vv=1:nFnsToEvalute
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                    ParamCell={};
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,tt);
                    ParamCell=num2cell(ValuesFnParamsVec);
                end
                daprime_val=daprimePolicy_gridvals(a_ind,:,tt);
                currentPanelValues_tt(:,vv)=EvalFnOnSimPanelIndex(FnsToEvaluate{vv},ParamCell,daprime_val,a_val,[],l_daprime,l_a,0);
            end
            SimPanelValues(:,tt,relevantindices)=reshape(currentPanelValues_tt',[nFnsToEvalute,1,sumrelevantindices]);

        end
    end
end



%% Implement new way of handling FnsToEvaluate: convert results
if FnsToEvaluateStruct==1
    % Change the output into a structure
    SimPanelValues2=SimPanelValues;
    clear SimPanelValues
    SimPanelValues=struct();
    for ff=1:length(FnsToEvalNames)
        SimPanelValues.(FnsToEvalNames{ff})=shiftdim(SimPanelValues2(ff,:,:),1);
    end
end




end



