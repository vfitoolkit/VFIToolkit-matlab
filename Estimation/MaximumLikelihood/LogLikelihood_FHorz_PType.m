function [LogL,LogLi,PTypePosterior,LogLi_bytype]=LogLikelihood_FHorz_PType(Data,MeasurementError,jequaloneDist,PTypeDistParamNames,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_grid,pi_z,simoptions,estimoptions)
% Log-likelihood of panel data for a finite-horizon (life-cycle) model with permanent types: a finite mixture over the types.
%
% For each permanent type k, LogLikelihood_FHorz gives each individual's log-likelihood conditional on being of type k, log L_i^k.
% The individual's log-likelihood is log( sum_k w_k * L_i^k ), with w_k the masses of the permanent types (PTypeDistParamNames).
% Computed as a log-sum-exp, so that it does not underflow.
%
% Inputs: as for LogLikelihood_FHorz, plus PTypeDistParamNames and Names_i, with everything that can differ by permanent type given
% as for the other PType commands (e.g., ValueFnIter_Case1_FHorz_PType, StationaryDist_Case1_FHorz_PType): Policy is the structure
% over the types from ValueFnIter_Case1_FHorz_PType; parameters as structures over the types or with a dimension of length N_i;
% grids, n_d, n_a, n_z, N_j and options as structures over the types; jequaloneDist in any of the forms StationaryDist_Case1_FHorz_PType
% takes. A FnsToEvaluate can differ by type, as FnsToEvaluate.(name).(typename). Data and MeasurementError are common to the types.
%
% Data.ptype (optional): N-by-1, the permanent type of each individual when it is observed (1 to N_i, as the order of Names_i), NaN when
% it is not. For an individual whose type is observed the likelihood is w_k*L_i^k for that type (the probability of being that type
% and of the data), not the mixture.
%
% estimoptions.initialcondition='conditional': the likelihood conditional on the observations at the first age,
% log( sum_k w_k L_i^k ) - log( sum_k w_k L_i^k(first age) ). (The mixture weights also enter the first age, which is why this is not
% simply the mixture of the type-by-type conditional likelihoods.)
%
% Outputs
%   LogL: the log-likelihood
%   LogLi: N-by-1, the log-likelihood of each individual
%   PTypePosterior: N-by-N_i, the posterior probability that each individual is of each permanent type, given their data
%   LogLi_bytype: N-by-N_i, log L_i^k, each individual's log-likelihood conditional on being of each type (with the default
%                 initial condition, whatever estimoptions.initialcondition is)

if ~exist('estimoptions','var')
    estimoptions=struct();
end
if ~isfield(estimoptions,'initialcondition')
    estimoptions.initialcondition='model';
end
if ~any(strcmp(estimoptions.initialcondition,{'model','conditional'}))
    error('LogLikelihood_FHorz_PType: estimoptions.initialcondition must be ''model'' or ''conditional''')
end

%% Names_i and N_i
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i={'ptype001'};
    for ii=2:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end
if ~iscell(PTypeDistParamNames)
    error('PTypeDistParamNames should be a cell, it is not')
end

%% jequaloneDist (in any of the PType forms) and the type masses
[jequaloneDist,Parameters]=jequaloneDist_PType(jequaloneDist,Parameters,simoptions,n_a,n_z,N_i,Names_i,PTypeDistParamNames);
ptweights=gather(reshape(Parameters.(PTypeDistParamNames{1}),[],1));
if length(ptweights)~=N_i
    error('LogLikelihood_FHorz_PType: the parameter in PTypeDistParamNames does not have the same number of permanent types as N_i/Names_i')
end
if abs(sum(ptweights)-1)>10^(-12)
    warning('The permanent type mass weights must sum to one (PTypeDistParamNames points to weights that do not sum to one)')
end
if any(ptweights<0)
    error('LogLikelihood_FHorz_PType: the permanent type masses (PTypeDistParamNames) must be non-negative')
end

%% Observed types
if isfield(Data,'ptype')
    ptypeobs=gather(reshape(Data.ptype,[],1));
    Data=rmfield(Data,'ptype');
    temp=ptypeobs(~isnan(ptypeobs));
    if any(temp~=round(temp)) || any(temp<1) || any(temp>N_i)
        error('LogLikelihood_FHorz_PType: Data.ptype must be integers from 1 to N_i (or NaN where the type is not observed)')
    end
else
    ptypeobs=[];
end

%% The likelihood of each individual conditional on each type
ObsNames=fieldnames(Data);
ObsNames=ObsNames(~strcmp(ObsNames,'age') & ~strcmp(ObsNames,'agej0'));
N=size(Data.(ObsNames{1}),1);
estimoptions_ii=estimoptions;
estimoptions_ii.initialcondition='model'; % the conditional version is done below, on the mixture
LogLi_bytype=[];
LogLfirst_bytype=[];
for ii=1:N_i
    iistr=Names_i{ii};
    simoptions_temp=PType_Options(simoptions,iistr);
    if isstruct(n_d)
        n_d_temp=n_d.(iistr);
    else
        n_d_temp=n_d;
    end
    if isstruct(n_a)
        n_a_temp=n_a.(iistr);
    else
        n_a_temp=n_a;
    end
    if isstruct(N_j)
        N_j_temp=N_j.(iistr);
    else
        N_j_temp=N_j;
    end
    if isstruct(d_grid)
        d_grid_temp=d_grid.(iistr);
    else
        d_grid_temp=d_grid;
    end
    if isstruct(a_grid)
        a_grid_temp=a_grid.(iistr);
    else
        a_grid_temp=a_grid;
    end
    [n_z_temp,z_grid_temp,pi_z_temp,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,pi_z,simoptions_temp,3);
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);
    if isstruct(jequaloneDist)
        if ~isfield(jequaloneDist,iistr)
            error(['You must input a jequaloneDist for permanent type ', iistr])
        end
        jequaloneDist_temp=jequaloneDist.(iistr);
    else
        jequaloneDist_temp=jequaloneDist;
    end
    FnNames=fieldnames(FnsToEvaluate);
    FnsToEvaluate_temp=struct();
    for ff=1:length(FnNames)
        if isstruct(FnsToEvaluate.(FnNames{ff}))
            if isfield(FnsToEvaluate.(FnNames{ff}),iistr)
                FnsToEvaluate_temp.(FnNames{ff})=FnsToEvaluate.(FnNames{ff}).(iistr);
            end
        else
            FnsToEvaluate_temp.(FnNames{ff})=FnsToEvaluate.(FnNames{ff});
        end
    end

    if ptweights(ii)>0
        [~,LogLi_ii,LogLij_ii]=LogLikelihood_FHorz(Data,MeasurementError,jequaloneDist_temp,Policy.(iistr),FnsToEvaluate_temp,Parameters_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,z_grid_temp,pi_z_temp,simoptions_temp,estimoptions_ii);
        LogLi_ii=gather(LogLi_ii);
        % the log normaliser of the first age of each individual (the first non-NaN entry of each row of LogLij)
        [~,firstage]=max(~isnan(LogLij_ii),[],2);
        LogLfirst_ii=LogLij_ii(sub2ind(size(LogLij_ii),(1:1:size(LogLij_ii,1))',firstage));
    else
        % a type with zero mass contributes nothing (and its model need not even be well defined)
        LogLi_ii=-Inf(N,1);
        LogLfirst_ii=LogLi_ii;
    end
    LogLi_bytype=[LogLi_bytype, LogLi_ii];
    LogLfirst_bytype=[LogLfirst_bytype, gather(LogLfirst_ii)];
end

%% The mixture
logw=log(ptweights'); % 1-by-N_i (log(0)=-Inf for a type with zero mass)
logjoint=LogLi_bytype+logw; % N-by-N_i: log(w_k*L_i^k)
logfirstjoint=LogLfirst_bytype+logw;
if ~isempty(ptypeobs)
    for ii=1:N_i
        notii=(~isnan(ptypeobs) & ptypeobs~=ii);
        logjoint(notii,ii)=-Inf;
        logfirstjoint(notii,ii)=-Inf;
    end
end
% log-sum-exp over the types
m=max(logjoint,[],2);
mfinite=m;
mfinite(~isfinite(m))=0;
LogLi=mfinite+log(sum(exp(logjoint-mfinite),2));
LogLi(m==-Inf)=-Inf;
PTypePosterior=exp(logjoint-LogLi);
PTypePosterior(m==-Inf,:)=NaN; % no type can generate this individual's data
if strcmp(estimoptions.initialcondition,'conditional')
    mf=max(logfirstjoint,[],2);
    mffinite=mf;
    mffinite(~isfinite(mf))=0;
    LogLfirst=mffinite+log(sum(exp(logfirstjoint-mffinite),2));
    LogLfirst(mf==-Inf)=-Inf;
    LogLi=LogLi-LogLfirst;
end
LogL=sum(LogLi);

end

