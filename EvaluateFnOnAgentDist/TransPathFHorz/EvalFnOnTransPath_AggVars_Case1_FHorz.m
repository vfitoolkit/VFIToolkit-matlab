function AggVarsPath=EvalFnOnTransPath_AggVars_Case1_FHorz(FnsToEvaluate, AgentDistPath, PolicyPath, PricePath, ParamPath, Parameters, T, n_d, n_a, n_z, N_j, d_grid, a_grid,z_grid, transpathoptions, simoptions)
% AggVarsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values.
%
%
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes)

% PricePath is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePath


%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
end

%% Check which simoptions have been used, set all others to defaults
if exist('simoptions','var')==0
    simoptions.parallel=2;
    simoptions.verbose=0;
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
    simoptions.policy_forceintegertype=0;
    simoptions.fastOLG=1; % parallel over j, faster but uses more memory
    simoptions.gridinterplayer=0;
else
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=2;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'iterate')
        simoptions.iterate=1;
    end
    if ~isfield(simoptions,'policy_forceintegertype')
        simoptions.policy_forceintegertype=1;
    end
    if ~isfield(simoptions,'fastOLG')
        simoptions.fastOLG=1; % parallel over j, faster but uses more memory
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    elseif simoptions.gridinterplayer==1
        if ~isfield(simoptions,'ngridinterp')
            error('You have simoptions.gridinterplayer, so must also set simoptions.ngridinterp')
        end
    end
end

%%
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end
l_ze=l_z;

N_e=0;
if isfield(simoptions,'n_e')
    l_e=length(simoptions.n_e);
    l_ze=l_z+l_e;
    N_e=prod(simoptions.n_e);
end


%% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
% Actually, some of those prices are 1-by-N_j, so is more subtle than this.
PricePathNames=fieldnames(PricePath);
PricePathStruct=PricePath;
PricePathSizeVec=zeros(1,length(PricePathNames)); % Allows for a given price param to depend on age (or permanent type)
for ii=1:length(PricePathNames)
    temp=PricePathStruct.(PricePathNames{ii});
    tempsize=size(temp);
    PricePathSizeVec(ii)=tempsize(tempsize~=T); % Get the dimension which is not T
end
PricePathSizeVec=cumsum(PricePathSizeVec);
if length(PricePathNames)>1
    PricePathSizeVec=[[1,PricePathSizeVec(1:end-1)+1];PricePathSizeVec];
else
    PricePathSizeVec=[1;PricePathSizeVec];
end
PricePath=zeros(T,PricePathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for ii=1:length(PricePathNames)
    if size(PricePathStruct.(PricePathNames{ii}),1)==T
        PricePath(:,PricePathSizeVec(1,ii):PricePathSizeVec(2,ii))=PricePathStruct.(PricePathNames{ii});
    else % Need to transpose
        PricePath(:,PricePathSizeVec(1,ii):PricePathSizeVec(2,ii))=PricePathStruct.(PricePathNames{ii})';
    end
end

ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath;
ParamPathSizeVec=zeros(1,length(ParamPathNames)); % Allows for a given price param to depend on age (or permanent type)
for ii=1:length(ParamPathNames)
    temp=ParamPathStruct.(ParamPathNames{ii});
    tempsize=size(temp);
    ParamPathSizeVec(ii)=tempsize(tempsize~=T); % Get the dimension which is not T
end
ParamPathSizeVec=cumsum(ParamPathSizeVec);
if length(ParamPathNames)>1
    ParamPathSizeVec=[[1,ParamPathSizeVec(1:end-1)+1];ParamPathSizeVec];
else
    ParamPathSizeVec=[1;ParamPathSizeVec];
end
ParamPath=zeros(T,ParamPathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for ii=1:length(ParamPathNames)
    if size(ParamPathStruct.(ParamPathNames{ii}),1)==T
        ParamPath(:,ParamPathSizeVec(1,ii):ParamPathSizeVec(2,ii))=ParamPathStruct.(ParamPathNames{ii});
    else % Need to transpose
        ParamPath(:,ParamPathSizeVec(1,ii):ParamPathSizeVec(2,ii))=ParamPathStruct.(ParamPathNames{ii})';
    end
end

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate)
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames);
    if transpathoptions.verbose>1
        tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk
    end
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tminus1AggVarsNames=[];
    tplus1pricePathkk=[];
end
 
use_tplus1price=0;
if length(tplus1priceNames)>0
    use_tplus1price=1;
end
use_tminus1price=0;
if length(tminus1priceNames)>0
    use_tminus1price=1;
    for ii=1:length(tminus1priceNames)
        if ~isfield(transpathoptions.initialvalues,tminus1priceNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{ii})
        end
    end
end
use_tminus1AggVars=0;
if length(tminus1AggVarsNames)>0
    use_tminus1AggVars=1;
    for ii=1:length(tminus1AggVarsNames)
        if ~isfield(transpathoptions.initialvalues,tminus1AggVarsNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1AggVarsNames{ii})
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.


%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
% Figure out l_daprime from Policy
l_daprime=size(PolicyPath,1)-simoptions.gridinterplayer; % Note: simoptions.gridinterplayer=1 means that PolicyIndexes has an extra 'second layer index'

AggVarNames=fieldnames(FnsToEvaluate);
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    if length(temp)>(l_daprime+l_a+l_ze)
        FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_ze+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
end
FnsToEvaluate=FnsToEvaluate2;
% Change FnsToEvaluate out of structure form, but want to still create AggVars as a structure
simoptions.outputasstructure=1;
simoptions.AggVarNames=AggVarNames;


%% Set up exogenous shock processes
[z_gridvals_J, ~, e_gridvals_J, ~, transpathoptions, simoptions]=ExogShockSetup_TPath_FHorz(n_z,z_grid,[],N_j,Parameters,PricePathNames,ParamPathNames,transpathoptions,simoptions,1);
% Convert z and e to age-dependent joint-grids and transtion matrix
% output: z_gridvals_J, pi_z_J, e_gridvals_J, pi_e_J, transpathoptions,vfoptions,simoptions

% Sets up
% transpathoptions.zpathtrivial=1; % z_gridvals_J and pi_z_J are not varying over the path
%                              =0; % they vary over path, so z_gridvals_J_T and pi_z_J_T
% transpathoptions.epathtrivial=1; % e_gridvals_J and pi_e_J are not varying over the path
%                              =0; % they vary over path, so e_gridvals_J_T and pi_e_J_T
% and
% transpathoptions.gridsinGE=1; % grids depend on a GE parameter and so need to be recomputed every iteration
%                           =0; % grids are exogenous

%%
% n_dalt=n_d(1:end-simoptions.gridinterplayer);

AggVarsPath=struct();

if N_e==0
    if N_z==0
        AgentDistPath=reshape(AgentDistPath,[N_a,N_j,T]);
        PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_j,T]);

        for tt=1:T

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

            Policy_tt=reshape(PolicyPath(:,:,:,tt),[size(PolicyPath,1),n_a,N_j]);

            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDistPath(:,:,tt), Policy_tt, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, [], simoptions);

            for ff=1:length(AggVarNames)
                AggVarsPath.(AggVarNames{ff}).Mean(tt)=AggVars.(AggVarNames{ff}).Mean;
            end
        end
    else % N_z>0
        AgentDistPath=reshape(AgentDistPath,[N_a,N_z,N_j,T]);
        PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,N_j,T]);

        for tt=1:T

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

            if transpathoptions.zpathtrivial==0
                z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,tt);
            end

            Policy_tt=reshape(PolicyPath(:,:,:,:,tt),[size(PolicyPath,1),n_a,n_z,N_j]);

            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDistPath(:,:,:,tt), Policy_tt, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, simoptions);

            for ff=1:length(AggVarNames)
                AggVarsPath.(AggVarNames{ff}).Mean(tt)=AggVars.(AggVarNames{ff}).Mean;
            end
        end
    end

else
    if N_z==0
        AgentDistPath=reshape(AgentDistPath,[N_a,N_e,N_j,T]);
        PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_e,N_j,T]);

        for tt=1:T

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

            if transpathoptions.epathtrivial==0
                e_gridvals_J=transpathoptions.e_grid_J_T(:,:,tt);
            end

            Policy_tt=reshape(PolicyPath(:,:,:,:,tt),[size(PolicyPath,1),n_a,n_e,N_j]);

            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDistPath(:,:,:,:,tt), Policy_tt, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_e, N_j, d_grid, a_grid, e_gridvals_J, simoptions);
            
            for ff=1:length(AggVarNames)
                AggVarsPath.(AggVarNames{ff}).Mean(tt)=AggVars.(AggVarNames{ff}).Mean;
            end
        end
    else
        simoptions.n_e=n_e;
        simoptions.e_grid=e_gridvals_J;
        simoptions.pi_e_J=pi_e_J;
        AgentDistPath=reshape(AgentDistPath,[N_a,N_z,N_e,N_j,T]);
        PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,N_e,N_j,T]);

        for tt=1:T

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

            if transpathoptions.zpathtrivial==0
                z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,tt);
            end
            if transpathoptions.epathtrivial==0
                simoptions.e_grid_J=transpathoptions.e_grid_J_T(:,:,tt);
            end

            Policy_tt=reshape(PolicyPath(:,:,:,:,:,tt),[size(PolicyPath,1),n_a,n_z,n_e,N_j]);

            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDistPath(:,:,:,:,tt), Policy_tt, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, simoptions);
            
            for ff=1:length(AggVarNames)
                AggVarsPath.(AggVarNames{ff}).Mean(tt)=AggVars.(AggVarNames{ff}).Mean;
            end
        end
    end
end



end