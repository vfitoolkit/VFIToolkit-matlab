function AgeConditionalStatsPath=LifeCycleProfiles_TransPath_FHorz_Case1(FnsToEvaluate, AgentDistPath,PolicyPath, PricePath, ParamPath, Parameters,T,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)
% Changing z or e over the trasition is not supported.
%
% Works from AgentDistPath to calculate life-cycle profiles over the transition.
% Where applicable it is faster and more accurate.
% options.agegroupings can be used to do conditional on 'age bins' rather than age
% e.g., options.agegroupings=1:10:N_j will divide into 10 year age bins and calculate stats for each of them
% options.npoints can be used to determine how many points are used for the lorenz curve
% options.nquantiles can be used to change from reporting (age conditional) ventiles, to quartiles/deciles/percentiles/etc.
%
% Note that the quantile are what are typically reported as life-cycle profiles (or more precisely, the quantile cutoffs).
%
% Output takes following form
% For 'initialage', for Mean it is a matrix in which first dimension indexes age in period 1 of transtion, second dimension is current age
%                   for LorenzCurve, QuantileCuttoffs, QuantileMeans, these are the second & third dimensions
% For 'bornduringtranstion', for Mean it is a matrix in which first dimension indexes period of transition in which born (age period 1), second dimension is current age

if exist('simoptions','var')
    if isfield(simoptions,'nquantiles')==0
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if isfield(simoptions,'agegroupings')==0
        simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    end
    if isfield(simoptions,'npoints')==0
        simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    end
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
else
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.parallel=2;
    simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
end

%% Not the fastest approach (as unnecessary overhead), but just loop over t (the transtion time periods) and for each one run life-cycle profile FHorz Case1 command.
% Then just reorganise the results.
ngroups=length(simoptions.agegroupings);
if ngroups~=N_j
    error('LifeCycleProfiles_TransPath_FHorz_Case1 only permits using agegroupings of an individual model period (was the only obvious use as otherwise the grouping composition changes, if you envisage a use that is not this, let me know)')
end
AggVarNames=fieldnames(FnsToEvaluate);

OutputTypes={'Mean','Median','Variance','LorenzCurve','Gini','QuantileCutoffs','QuantileMeans'};

AgeConditionalStatsPath=struct();
for ff=1:length(AggVarNames)
    for oo=1:length(OutputTypes)
        if any(oo==[1,2,3,5]) % Mean, Median, Variance, Gini
            AgeConditionalStatsPath.(AggVarNames{ff}).initialages.(OutputTypes{oo})=nan(ngroups,ngroups); % Note: will be triangular (rest remains nan): first dimension is initial age (during period 1 of transtion, so age at reform), second dimension is their age
            AgeConditionalStatsPath.(AggVarNames{ff}).bornduringtranstion.(OutputTypes{oo})=nan(T,ngroups); % First dimensions is the transition period during which they were born, second dimension is their age
        elseif oo==4 % Lorenz curve
            AgeConditionalStatsPath.(AggVarNames{ff}).initialages.(OutputTypes{oo})=nan(simoptions.npoints,ngroups,ngroups); % Note: will be triangular (rest remains nan): first dimension is initial age (during period 1 of transtion, so age at reform), second dimension is their age
            AgeConditionalStatsPath.(AggVarNames{ff}).bornduringtranstion.(OutputTypes{oo})=nan(simoptions.npoints,T,ngroups); % First dimensions is the transition period during which they were born, second dimension is their age
        elseif oo==6 % Quantile cutoffs
            AgeConditionalStatsPath.(AggVarNames{ff}).initialages.(OutputTypes{oo})=nan(simoptions.nquantiles+1,ngroups,ngroups); % Note: will be triangular (rest remains nan): first dimension is initial age (during period 1 of transtion, so age at reform), second dimension is their age
            AgeConditionalStatsPath.(AggVarNames{ff}).bornduringtranstion.(OutputTypes{oo})=nan(simoptions.nquantiles+1,T,ngroups); % First dimensions is the transition period during which they were born, second dimension is their age
        elseif oo==7 % Quantile Means
            AgeConditionalStatsPath.(AggVarNames{ff}).initialages.(OutputTypes{oo})=nan(simoptions.nquantiles,ngroups,ngroups); % Note: will be triangular (rest remains nan): first dimension is initial age (during period 1 of transtion, so age at reform), second dimension is their age
            AgeConditionalStatsPath.(AggVarNames{ff}).bornduringtranstion.(OutputTypes{oo})=nan(simoptions.nquantiles,T,ngroups); % First dimensions is the transition period during which they were born, second dimension is their age
        end
    end
end

%% Set up a few things
N_a=prod(n_a);

if n_z(1)>0
    N_z=prod(n_z);
    if isfield(simoptions,'n_e')
        N_e=prod(simoptions.n_e);
        AgentDistPath=reshape(AgentDistPath, [N_a,N_z,N_e,N_j,T]);
    else
        AgentDistPath=reshape(AgentDistPath, [N_a,N_z,N_j,T]);
    end
else
    AgentDistPath=reshape(AgentDistPath, [N_a,N_j,T]);
end

if n_z(1)==0
    PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1_noz(PolicyPath, n_d, n_a, N_j, T);
else
    if isfield(simoptions,'n_e')
        PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1(PolicyPath, n_d, n_a, n_z, N_j, T, n_e);
    else
        PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1(PolicyPath, n_d, n_a, n_z, N_j, T);
    end
end

%% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
% Actually, some of those prices are 1-by-N_j, so is more subtle than this.
PricePathNames=fieldnames(PricePath);
PricePathStruct=PricePath; 
PricePathSizeVec=zeros(1,length(PricePathNames)); % Allows for a given price param to depend on age (or permanent type)
for tt=1:length(PricePathNames)
    temp=PricePathStruct.(PricePathNames{tt});
    tempsize=size(temp);
    PricePathSizeVec(tt)=tempsize(tempsize~=T); % Get the dimension which is not T
end
PricePathSizeVec=cumsum(PricePathSizeVec);
if length(PricePathNames)>1
    PricePathSizeVec=[[1,PricePathSizeVec(1:end-1)+1];PricePathSizeVec];
else
    PricePathSizeVec=[1;PricePathSizeVec];
end
PricePath=zeros(T,PricePathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for tt=1:length(PricePathNames)
    if size(PricePathStruct.(PricePathNames{tt}),1)==T
        PricePath(:,PricePathSizeVec(1,tt):PricePathSizeVec(2,tt))=PricePathStruct.(PricePathNames{tt});
    else % Need to transpose
        PricePath(:,PricePathSizeVec(1,tt):PricePathSizeVec(2,tt))=PricePathStruct.(PricePathNames{tt})';
    end
end

ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath;
ParamPathSizeVec=zeros(1,length(ParamPathNames)); % Allows for a given price param to depend on age (or permanent type)
for tt=1:length(ParamPathNames)
    temp=ParamPathStruct.(ParamPathNames{tt});
    tempsize=size(temp);
    ParamPathSizeVec(tt)=tempsize(tempsize~=T); % Get the dimension which is not T
end
ParamPathSizeVec=cumsum(ParamPathSizeVec);
if length(ParamPathNames)>1
    ParamPathSizeVec=[[1,ParamPathSizeVec(1:end-1)+1];ParamPathSizeVec];
else
    ParamPathSizeVec=[1;ParamPathSizeVec];
end
ParamPath=zeros(T,ParamPathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for tt=1:length(ParamPathNames)
    if size(ParamPathStruct.(ParamPathNames{tt}),1)==T
        ParamPath(:,ParamPathSizeVec(1,tt):ParamPathSizeVec(2,tt))=ParamPathStruct.(ParamPathNames{tt});
    else % Need to transpose
        ParamPath(:,ParamPathSizeVec(1,tt):ParamPathSizeVec(2,tt))=ParamPathStruct.(ParamPathNames{tt})';
    end
end


%% Check if z_grid and/or pi_z depend on prices. If not then create pi_z_J and z_grid_J for the entire transition before we start
% If 'exogenous shock fn' is used, then precompute it to save evaluating it numerous times
% Check if using 'exogenous shock fn' (exogenous state has a grid and transition matrix that depends on age)

transpathoptions.zpathprecomputed=0;
if isfield(simoptions,'pi_z_J')
    transpathoptions.zpathprecomputed=1;
    transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
elseif isfield(simoptions,'ExogShockFn')
    % Note: If ExogShockFn depends on the path, it must be done via a parameter
    % that depends on the path (i.e., via ParamPath or PricePath)
    overlap=0;
    for ii=1:length(simoptions.ExogShockFnParamNames)
        if strcmp(simoptions.ExogShockFnParamNames{ii},PricePathNames)
            overlap=1;
        end
    end
    if overlap==0
        transpathoptions.zpathprecomputed=1;
        % If ExogShockFn does not depend on any of the prices (in PricePath), then
        % we can simply create it now rather than within each 'subfn' or 'p_grid'
        
        % Check if it depends on the ParamPath
        transpathoptions.zpathtrivial=1;
        for ii=1:length(simoptions.ExogShockFnParamNames)
            if strcmp(simoptions.ExogShockFnParamNames{ii},ParamPathNames)
                transpathoptions.zpathtrivial=0;
            end
        end
        if transpathoptions.zpathtrivial==1
            pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
            z_grid_J=zeros(N_z,N_j,'gpuArray');
            for jj=1:N_j
                if isfield(simoptions,'ExogShockFnParamNames')
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                else
                    [z_grid,pi_z]=simoptions.ExogShockFn(jj);
                end
                pi_z_J(:,:,jj)=gpuArray(pi_z);
                z_grid_J(:,jj)=gpuArray(z_grid);
            end
            % Now store them in simoptions
            simoptions.pi_z_J=pi_z_J;
            simoptions.z_grid_J=z_grid_J;
        elseif transpathoptions.zpathtrivial==0
            % z_grid_J and/or pi_z_J varies along the transition path (but only depending on ParamPath, not PricePath
            transpathoptions.pi_z_J_T=zeros(N_z,N_z,N_j,T,'gpuArray');
            transpathoptions.z_grid_J_T=zeros(sum(n_z),N_j,T,'gpuArray');
            pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
            z_grid_J=zeros(sum(n_z),N_j,'gpuArray');
            for tt=1:T
                for ii=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                end
                % Note, we know the PricePath is irrelevant for the current purpose
                for jj=1:N_j
                    if isfield(simoptions,'ExogShockFnParamNames')
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                        end
                        [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    else
                        [z_grid,pi_z]=simoptions.ExogShockFn(jj);
                    end
                    pi_z_J(:,:,jj)=gpuArray(pi_z);
                    z_grid_J(:,jj)=gpuArray(z_grid);
                end
                transpathoptions.pi_z_J_T(:,:,:,tt)=pi_z_J;
                transpathoptions.z_grid_J_T(:,:,tt)=z_grid_J;
            end
        end
    end
end

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate) %&& isstruct(GeneralEqmEqns)
    GeneralEqmEqns=struct();
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames);
%     tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk
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

%% The loop itself
for tt=1:T
        
    % Get current AgentDist
    if n_z(1)>0
        if isfield(simoptions,'n_e')
            AgentDist=AgentDistPath(:,:,:,:,tt);
        else
            AgentDist=AgentDistPath(:,:,:,tt);
        end
    else
        AgentDist=AgentDistPath(:,:,tt);
    end
    if simoptions.parallel==2
        AgentDist=gpuArray(AgentDist);
    end
    % Get current Policy
    if n_d(1)>0
        if n_z(1)>0
            if isfield(simoptions,'n_e')
                Policy=PolicyPath(:,:,:,:,:,tt); % (d,a'),a,z,e,j
            else
                Policy=PolicyPath(:,:,:,:,tt); % (d,a'),a,z,j
            end
        else
            Policy=PolicyPath(:,:,:,tt); % (d,a'),a,j
        end
    else
        if n_z(1)>0
            if isfield(simoptions,'n_e')
                Policy=PolicyPath(:,:,:,:,tt); % a,z,e,j
            else
                Policy=PolicyPath(:,:,:,tt); % a,z,j
            end
        else
            Policy=PolicyPath(:,:,tt); % a,j
        end        
    end
    % Get current ParamPath and PricePath
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
    
    % Get current shocks (if applicable)
    if transpathoptions.zpathprecomputed==1
        if transpathoptions.zpathtrivial==1
            simoptions.pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
            simoptions.z_grid_J=transpathoptions.z_grid_J_T(:,:,tt);
        end
        % transpathoptions.zpathtrivial==0 % Does not depend on T, so is just in simoptions already
    end
    % transpathoptions.zpathprecomputed==0 % Depends on the price path  parameters, so just have to use simoptions.ExogShockFn within  ValueFnIter command
    
    if n_z(1)==0
        PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz_noz(Policy, n_d, n_a, N_j,simoptions);
    else
        PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz(Policy, n_d, n_a, n_z, N_j,simoptions);
    end
    tempAgeConditionalStats=LifeCycleProfiles_FHorz_Case1(AgentDist,PolicyUnKron,FnsToEvaluate,[],Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
    
    
    for ff=1:length(AggVarNames)
        for oo=1:length(OutputTypes)
            temp=tempAgeConditionalStats.(AggVarNames{ff}).(OutputTypes{oo});
            % First, sort out the initial ages
            if tt<=ngroups
                temp2=AgeConditionalStatsPath.(AggVarNames{ff}).initialages.(OutputTypes{oo}); % Note: will be triangular (rest remains nan)
                for jj=1:ngroups-tt % Age at time of reform (age in period when the ParamPath is revealed)
                    for ii=1:N_j
                        if any(oo==[1,2,3,5])
                            % Age jj in first period, are now aged jj+tt-1
                            temp2(jj,jj+tt-1)=temp(jj+tt-1); % Mean, Median, Variance, Gini
                        else
                            temp2(:,jj,jj+tt-1)=temp(:,jj+tt-1); % LorenzCurve, QuantileCutoffs, QuantileMean
                        end
                    end
                end
                AgeConditionalStatsPath.(AggVarNames{ff}).initialages.(OutputTypes{oo})=temp2;
            end
            % Now, those born in the transition
            temp3=AgeConditionalStatsPath.(AggVarNames{ff}).bornduringtranstion.(OutputTypes{oo}); 
            for ii=max(1,tt-N_j+1):tt
                if any(oo==[1,2,3,5])
                    % Born in ii, are now aged tt-ii+1
                    temp3(ii,tt-ii+1)=temp(tt-ii+1); % Mean, Median, Variance, Gini
                else
                    temp3(:,ii,tt-ii+1)=temp(:,tt-ii+1); % LorenzCurve, QuantileCutoffs, QuantileMean
                end
            end
        end
    end
    
    
end



end


