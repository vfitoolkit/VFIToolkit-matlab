function [LogL,LogLi,LogLij,PopDist]=LogLikelihood_FHorz(Data,MeasurementError,jequaloneDist,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,pi_z,simoptions,estimoptions)
% Log-likelihood of panel data for a finite-horizon (life-cycle) model, evaluated exactly on the model grid.
%
% The model is treated as a hidden Markov model. The latent state is the agent's state on the grid (a,z,e). The transitions are
% those of the solved model (Policy, including the grid interpolation layer lotteries, then pi_z, then pi_e), which is the same
% operation as the agent distribution iteration (with the Tan improvement), here applied to one distribution per individual.
% The observations are FnsToEvaluate (functions of decisions and states) plus measurement error. The likelihood is evaluated with
% the forward algorithm (Hamilton filter): at each age, reweight each individual's distribution over the latent state by the
% probability of what was observed at each grid state (the emission), normalise, and accumulate the log of the normaliser; then
% move the distribution forward one period using the model.
%
% Inputs
%   Data: structure. One field per observed variable, named as in FnsToEvaluate, each an N-by-T matrix (N individuals, T periods),
%         with NaN for missing observations. Plus the age of each observation, either
%           Data.age: N-by-T matrix of ages (model period j, integers 1 to N_j); NaN where the individual is not in the panel
%           Data.agej0: N-by-1 vector, the age in the first column; the T columns are then consecutive ages (columns that
%                       would be after age N_j must be NaN)
%         T=1 (a cross-section) is allowed.
%   MeasurementError: structure, one field per observed variable (the same names as in Data). Each has
%         .distribution: 'normal' (default), 't', 'exact', 'gridlottery', or 'misclassified'
%         .type: 'additive' (default) or 'log' (multiplicative error; only for 'normal' and 't')
%         .sigma: name of a parameter in Parameters, the standard deviation (scale for 't') of the measurement error ('normal', 't')
%         .dof: name of a parameter in Parameters, the degrees of freedom ('t' only)
%         .misclassification: 'uniform' ('misclassified' only; the only form currently implemented)
%         .misclassprob: name of a parameter in Parameters, the probability p of misreporting ('misclassified' only)
%         .values: (optional, 'misclassified' only) the K possible values of the observed variable; default is every value the
%                  FnToEvaluate takes on the grid at any age
%         Parameters named by sigma, dof and misclassprob can be scalars or depend on age (1-by-N_j).
%       The distributions:
%         'normal', 't': observed = model value + error ('additive'), or log(observed) = log(model value) + error ('log'; the
%                        density is then that of the observed value, including the 1/observed Jacobian, and is zero at grid
%                        states where the model value is not positive)
%         'exact': the observed value must equal the model value (for discrete-valued observables); emission is one or zero
%         'gridlottery': for a variable observed without error whose model values form a grid (e.g., an endogenous state): the
%                        observation is put on the two neighbouring values (among the distinct values the FnToEvaluate takes on
%                        the grid at that age) with linear-interpolation weights, and the emission at a grid state is the weight
%                        of its value. Observations outside the range of model values are put on the nearest end point (with a
%                        warning). This is an approximation.
%         'misclassified': for discrete-valued observables. With probability 1-p the observed value is the model value; otherwise
%                          it is one of the K-1 other values, each with probability p/(K-1) ('uniform').
%   jequaloneDist: the distribution of agents at age j=1 (array, or a function as for StationaryDist_FHorz_Case1)
%   Policy: from ValueFnIter_Case1_FHorz
%   FnsToEvaluate: structure of functions; must contain each observed variable (others are ignored)
%   Parameters, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, pi_z, simoptions: as for StationaryDist and EvalFnOnAgentDist
%   estimoptions (optional):
%         .initialcondition: 'model' (default) the latent state at the first age an individual is observed is distributed as the
%                            model agent distribution at that age (from jequaloneDist); 'conditional' uses the same distribution,
%                            but the likelihood is conditional on the first-age observations (the log normaliser of the first
%                            age is left out of LogLi)
%         .nindividualsbatch: number of individuals filtered at once (default: all); smaller uses less GPU memory
%
% Outputs
%   LogL: the log-likelihood, sum(LogLi)
%   LogLi: N-by-1, the log-likelihood of each individual
%   LogLij: N-by-N_j, the log normaliser of each individual at each age, log p(y_ij | y_i,earlier ages), at the ages from the
%           individual's first to last age (zero at an age within this range with no observations), NaN at other ages.
%           LogLij always includes the first age, also when estimoptions.initialcondition='conditional'.
%   PopDist: the model agent distribution at each age, NOT weighted by the age weights (each age has mass one); same shape
%            as StationaryDist. This is the distribution an individual has at the first age they are observed.
%
% Currently implemented for standard endogenous states (one to four), with or without d, z and e, and the grid interpolation
% layer. Not yet for semi-exogenous states, experience assets, risky assets, or residual assets.

%% Options
if ~exist('simoptions','var')
    simoptions=struct();
end
if ~isfield(simoptions,'gridinterplayer')
    simoptions.gridinterplayer=0;
elseif simoptions.gridinterplayer==1
    if ~isfield(simoptions,'ngridinterp')
        error('When using simoptions.gridinterplayer=1 you must set simoptions.ngridinterp (number of points to interpolate for aprime between each consecutive pair of points in a_grid)')
    end
end
if ~isfield(simoptions,'n_e')
    simoptions.n_e=0;
end
if ~isfield(simoptions,'n_semiz')
    simoptions.n_semiz=0;
end
if ~isfield(simoptions,'jequaloneDist_usergrids')
    simoptions.jequaloneDist_usergrids=1;
end
if ~isfield(simoptions,'alreadygridvals')
    simoptions.alreadygridvals=0; % =1 when called from inside an estimation command: z_grid and pi_z are then z_gridvals_J and pi_z_J, and simoptions holds e_gridvals_J and pi_e_J
end
if isfield(simoptions,'jequaloneDistAge')
    if simoptions.jequaloneDistAge~=1
        error('LogLikelihood_FHorz: simoptions.jequaloneDistAge other than 1 is not yet implemented')
    end
end
% Things not yet implemented
if prod(simoptions.n_semiz)>0
    error('LogLikelihood_FHorz: semi-exogenous states are not yet implemented')
end
temp={'experienceasset','experienceassetu','experienceassete','experienceassetz','experienceassetze','experienceassetsemiz','riskyasset','residualasset'};
for tt=1:length(temp)
    if isfield(simoptions,temp{tt})
        if simoptions.(temp{tt})>=1
            error('LogLikelihood_FHorz: simoptions.%s is not yet implemented',temp{tt})
        end
    end
end
if ~exist('estimoptions','var')
    estimoptions=struct();
end
if ~isfield(estimoptions,'initialcondition')
    estimoptions.initialcondition='model';
end
if ~any(strcmp(estimoptions.initialcondition,{'model','conditional'}))
    error('LogLikelihood_FHorz: estimoptions.initialcondition must be ''model'' or ''conditional''')
end
if ~isfield(estimoptions,'nindividualsbatch')
    estimoptions.nindividualsbatch=Inf;
end
exacttolerance=10^(-9); % 'exact' and 'misclassified': observed and model values are treated as equal when they differ by no more than this (relative to max(1,|observed|))

simoptions_vog=simoptions; % the simoptions as the user gave them, for EvalFnOnAgentDist_ValuesOnGrid (before the shock setup below adds to them)

%% Sizes
if prod(n_d)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(simoptions.n_e);
N_zint=max(N_z,1); % internally, no z is treated as a single z with transition probability one
N_eint=max(N_e,1); % internally, no e is treated as a single e with probability one
N_s=N_a*N_zint*N_eint; % number of latent grid states (a,z,e), with a fastest and e slowest

%% The data
ObsNames=fieldnames(Data);
ObsNames=ObsNames(~strcmp(ObsNames,'age') & ~strcmp(ObsNames,'agej0'));
nObs=length(ObsNames);
if nObs==0
    error('LogLikelihood_FHorz: Data contains no observed variables (only age)')
end
[N,T]=size(Data.(ObsNames{1}));
for vv=2:nObs
    if size(Data.(ObsNames{vv}),1)~=N || size(Data.(ObsNames{vv}),2)~=T
        error('LogLikelihood_FHorz: Data.%s is not the same size as Data.%s (every observed variable must be N-by-T)',ObsNames{vv},ObsNames{1})
    end
end
if isfield(Data,'age')
    if isfield(Data,'agej0')
        error('LogLikelihood_FHorz: Data should contain either age or agej0, not both')
    end
    Age=Data.age;
    if size(Age,1)~=N || size(Age,2)~=T
        error('LogLikelihood_FHorz: Data.age must be N-by-T, the same size as the observed variables')
    end
elseif isfield(Data,'agej0')
    if numel(Data.agej0)~=N
        error('LogLikelihood_FHorz: Data.agej0 must be N-by-1')
    end
    Age=gather(reshape(Data.agej0,[N,1]))+(0:1:T-1);
    Age(Age>N_j)=NaN; % periods after the final age (there must be no observations in them, checked below)
else
    error('LogLikelihood_FHorz: Data must contain the ages, either Data.age (N-by-T) or Data.agej0 (N-by-1)')
end
Age=gather(Age);
if any(Age(~isnan(Age))~=round(Age(~isnan(Age)))) || any(Age(~isnan(Age))<1) || any(Age(~isnan(Age))>N_j)
    error('LogLikelihood_FHorz: the ages in Data must be integers from 1 to N_j (or NaN)')
end
if any(all(isnan(Age),2))
    error('LogLikelihood_FHorz: every individual must have at least one age in Data (some rows of the ages are all NaN)')
end
tempAge=sort(Age,2); % NaN sort to the end
if any(any(diff(tempAge,1,2)==0))
    error('LogLikelihood_FHorz: an individual has two observations at the same age in Data')
end
j0=min(Age,[],2); % first age of each individual (min ignores NaN)
jlast=max(Age,[],2); % last age of each individual
% Observations by age: N-by-N_j for each observed variable
[iiind,ttind]=find(~isnan(Age));
ageind=Age(sub2ind([N,T],iiind,ttind));
for vv=1:nObs
    tempdata=gather(Data.(ObsNames{vv}));
    if any(~isnan(tempdata(isnan(Age))))
        error('LogLikelihood_FHorz: Data.%s has observations in periods where the age is NaN',ObsNames{vv})
    end
    ObsByAge.(ObsNames{vv})=NaN(N,N_j);
    ObsByAge.(ObsNames{vv})(sub2ind([N,N_j],iiind,ageind))=tempdata(sub2ind([N,T],iiind,ttind));
end

%% The measurement error
MEnames=fieldnames(MeasurementError);
for vv=1:length(MEnames)
    if ~any(strcmp(MEnames{vv},ObsNames))
        error('LogLikelihood_FHorz: MeasurementError.%s has no corresponding observed variable in Data',MEnames{vv})
    end
end
for vv=1:nObs
    if ~isfield(MeasurementError,ObsNames{vv})
        error('LogLikelihood_FHorz: Data.%s needs a measurement error specification, MeasurementError.%s',ObsNames{vv},ObsNames{vv})
    end
    ME=MeasurementError.(ObsNames{vv});
    if ~isfield(ME,'distribution')
        ME.distribution='normal';
    end
    if ~isfield(ME,'type')
        ME.type='additive';
    end
    if ~any(strcmp(ME.distribution,{'normal','t','exact','gridlottery','misclassified'}))
        error('LogLikelihood_FHorz: MeasurementError.%s.distribution must be one of normal, t, exact, gridlottery, misclassified',ObsNames{vv})
    end
    if ~any(strcmp(ME.type,{'additive','log'}))
        error('LogLikelihood_FHorz: MeasurementError.%s.type must be additive or log',ObsNames{vv})
    end
    if strcmp(ME.type,'log') && ~any(strcmp(ME.distribution,{'normal','t'}))
        error('LogLikelihood_FHorz: MeasurementError.%s.type=''log'' is only possible with the normal and t distributions',ObsNames{vv})
    end
    % The parameters, as 1-by-N_j vectors
    if any(strcmp(ME.distribution,{'normal','t'}))
        tempparamnames={'sigma'};
        if strcmp(ME.distribution,'t')
            tempparamnames={'sigma','dof'};
        end
    elseif strcmp(ME.distribution,'misclassified')
        tempparamnames={'misclassprob'};
        if ~isfield(ME,'misclassification')
            ME.misclassification='uniform';
        end
        if ~strcmp(ME.misclassification,'uniform')
            error('LogLikelihood_FHorz: MeasurementError.%s.misclassification=''%s'' is not implemented (only ''uniform'')',ObsNames{vv},ME.misclassification)
        end
    else
        tempparamnames={};
    end
    for pp=1:length(tempparamnames)
        if ~isfield(ME,tempparamnames{pp})
            error('LogLikelihood_FHorz: MeasurementError.%s.%s is required (the name of a parameter in Parameters)',ObsNames{vv},tempparamnames{pp})
        end
        if ~isfield(Parameters,ME.(tempparamnames{pp}))
            error('LogLikelihood_FHorz: MeasurementError.%s.%s is ''%s'', which is not in Parameters',ObsNames{vv},tempparamnames{pp},ME.(tempparamnames{pp}))
        end
        tempval=gather(Parameters.(ME.(tempparamnames{pp})));
        if isscalar(tempval)
            tempval=tempval*ones(1,N_j);
        elseif numel(tempval)==N_j
            tempval=reshape(tempval,[1,N_j]);
        else
            error('LogLikelihood_FHorz: the parameter %s (MeasurementError.%s.%s) must be a scalar or 1-by-N_j',ME.(tempparamnames{pp}),ObsNames{vv},tempparamnames{pp})
        end
        if strcmp(tempparamnames{pp},'misclassprob')
            if any(tempval<0) || any(tempval>1)
                error('LogLikelihood_FHorz: the parameter %s (MeasurementError.%s.misclassprob) must be between 0 and 1',ME.misclassprob,ObsNames{vv})
            end
        elseif any(tempval<=0)
            error('LogLikelihood_FHorz: the parameter %s (MeasurementError.%s.%s) must be positive',ME.(tempparamnames{pp}),ObsNames{vv},tempparamnames{pp})
        end
        MEvals.(ObsNames{vv}).(tempparamnames{pp})=tempval;
    end
    if strcmp(ME.type,'log')
        if any(ObsByAge.(ObsNames{vv})(:)<=0)
            error('LogLikelihood_FHorz: Data.%s has non-positive observations, which are impossible with MeasurementError.%s.type=''log''',ObsNames{vv},ObsNames{vv})
        end
    end
    MeasurementError.(ObsNames{vv})=ME;
end

%% Model values of the observed variables at every grid state and age
for vv=1:nObs
    if ~isfield(FnsToEvaluate,ObsNames{vv})
        error('LogLikelihood_FHorz: Data.%s has no corresponding FnsToEvaluate.%s',ObsNames{vv},ObsNames{vv})
    end
    FnsToEvaluate_obs.(ObsNames{vv})=FnsToEvaluate.(ObsNames{vv});
end
ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(Policy,FnsToEvaluate_obs,Parameters,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions_vog);
for vv=1:nObs
    ModelValues.(ObsNames{vv})=reshape(ValuesOnGrid.(ObsNames{vv}),[N_s,N_j]);
end
clear ValuesOnGrid
% The K values of each misclassified variable
for vv=1:nObs
    if strcmp(MeasurementError.(ObsNames{vv}).distribution,'misclassified')
        if isfield(MeasurementError.(ObsNames{vv}),'values')
            tempvalues=gpuArray(reshape(MeasurementError.(ObsNames{vv}).values,[],1));
            if numel(unique(tempvalues))<numel(tempvalues)
                error('LogLikelihood_FHorz: MeasurementError.%s.values contains repeated values',ObsNames{vv})
            end
            if any(min(abs(ModelValues.(ObsNames{vv})(:)-tempvalues'),[],2)>exacttolerance*max(1,abs(ModelValues.(ObsNames{vv})(:))))
                error('LogLikelihood_FHorz: FnsToEvaluate.%s takes values on the grid that are not in MeasurementError.%s.values',ObsNames{vv},ObsNames{vv})
            end
        else
            tempvalues=unique(ModelValues.(ObsNames{vv})(:));
        end
        if numel(tempvalues)<2
            error('LogLikelihood_FHorz: MeasurementError.%s is misclassified, which needs at least two possible values (set MeasurementError.%s.values)',ObsNames{vv},ObsNames{vv})
        end
        tempdata=gpuArray(ObsByAge.(ObsNames{vv})(~isnan(ObsByAge.(ObsNames{vv}))));
        if any(min(abs(tempdata-tempvalues'),[],2)>exacttolerance*max(1,abs(tempdata)))
            error('LogLikelihood_FHorz: Data.%s contains values that are not among the possible values of the misclassified variable',ObsNames{vv})
        end
        MEvals.(ObsNames{vv}).K=numel(tempvalues);
    end
end

%% Exogenous shocks
if isa(jequaloneDist,'function_handle')
    % jequaloneDist as a function needs the grids (as in StationaryDist_FHorz_Case1)
    if ~isfield(simoptions,'a_grid')
        error('When using jequaloneDist as a function you must put a_grid into simoptions.a_grid')
    elseif ~isfield(simoptions,'z_grid')
        error('When using jequaloneDist as a function you must put z_grid into simoptions.z_grid')
    end
    [z_gridvals_J, pi_z_J, simoptions]=ExogShockSetup_FHorz(n_z,simoptions.z_grid,pi_z,N_j,Parameters,simoptions,3,simoptions.jequaloneDist_usergrids);
    jequaloneDistFn=jequaloneDist;
    clear jequaloneDist
    temp=getAnonymousFnInputNames(jequaloneDistFn);
    if length(temp)>4 % first 4 are a_grid,z_grid,n_a,n_z
        jequaloneDistFnParamNames={temp{5:end}};
    else
        jequaloneDistFnParamNames={};
    end
    jequaloneParamsCell={};
    for pp=1:length(jequaloneDistFnParamNames)
        jequaloneParamsCell{pp}=Parameters.(jequaloneDistFnParamNames{pp});
    end
    if simoptions.jequaloneDist_usergrids==1
        jequaloneDist=jequaloneDistFn(simoptions.a_grid,simoptions.user_z_grid(:,:,1),n_a,n_z,jequaloneParamsCell{:});
    else
        jequaloneDist=jequaloneDistFn(simoptions.a_grid,z_gridvals_J,n_a,n_z,jequaloneParamsCell{:});
    end
elseif simoptions.alreadygridvals==1
    % Called from inside an estimation command, which has already set up the shocks: pi_z is pi_z_J, and simoptions.pi_e_J is set
    pi_z_J=pi_z;
    if N_e>0 && ~isfield(simoptions,'pi_e_J')
        error('LogLikelihood_FHorz: simoptions.alreadygridvals=1 needs simoptions.pi_e_J')
    end
else
    [~, pi_z_J, simoptions]=ExogShockSetup_FHorz(n_z,[],pi_z,N_j,Parameters,simoptions,2,0);
end
% size(pi_z_J)=[N_z,N_z,N_j-1], size(simoptions.pi_e_J)=[N_e,N_j]; pi_e_J(:,jj) is the distribution of the e realized in period jj
if N_z==0
    pi_z_J=ones(1,1,N_j-1);
end
pi_z_J=gpuArray(pi_z_J);
if N_e==0
    pi_e_J=ones(1,N_j,'gpuArray');
else
    pi_e_J=gpuArray(simoptions.pi_e_J);
end

jequaloneDist=gpuArray(jequaloneDist);
if abs(sum(jequaloneDist(:))-1)>10^(-9)
    error('The jequaloneDist must be of mass one')
end
jequaloneDist=reshape(jequaloneDist,[N_s,1]);

%% Transitions: next period aprime (with the grid interpolation layer lottery), as in StationaryDist_FHorz_Case1
Policy=reshape(Policy,[size(Policy,1),N_a,N_zint*N_eint,N_j]);
if l_a==1 % one endo state
    Policy_aprime=Policy(l_d+1,:,:,:);
elseif l_a==2 % two endo states
    Policy_aprime=Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1);
elseif l_a==3 % three endo states
    Policy_aprime=Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1);
elseif l_a==4 % four endo states
    Policy_aprime=Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1);
else
    error('Not yet implemented standard endogenous states with length(n_a)>4')
end
Policy_aprime=shiftdim(Policy_aprime,1); % [N_a,N_ze,N_j]
if simoptions.gridinterplayer==0
    N_probs=1;
    Policy_aprime=reshape(Policy_aprime,[N_a,N_zint*N_eint,1,N_j]);
    PolicyProbs=ones([N_a,N_zint*N_eint,1,N_j],'gpuArray');
elseif simoptions.gridinterplayer==1
    N_probs=2;
    Policy_aprime=reshape(Policy_aprime,[N_a,N_zint*N_eint,1,N_j]);
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=ones([N_a,N_zint*N_eint,2,N_j],'gpuArray');
    % Policy_aprime(:,:,1,:) lower grid point for a1 is unchanged
    Policy_aprime(:,:,2,:)=Policy_aprime(:,:,2,:)+1; % add one to a1, to get upper grid point
    % L2flag override (1=force all weight to lower, 2=usual, 3=force all weight to upper)
    L2index=Policy(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
    L2flag=Policy(end,:,:,:);
    L2index(L2flag==1)=1;                        % force all weight to lower grid point
    L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
    aprimeProbs_upper=reshape(shiftdim((L2index-1)/(simoptions.ngridinterp+1),1),[N_a,N_zint*N_eint,1,N_j]); % probability of upper grid point
    PolicyProbs(:,:,1,:)=PolicyProbs(:,:,1,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,2,:)=PolicyProbs(:,:,2,:).*aprimeProbs_upper; % upper a1
end
% Index into (aprime,z) [Tan improvement: z stays where it is, then pi_z, then e is put back in]
Policy_aprimez=Policy_aprime+repmat(N_a*(0:1:N_zint-1),1,N_eint);
Policy_aprimez=gather(reshape(Policy_aprimez,[N_s,N_probs,N_j]));
PolicyProbs=gather(reshape(PolicyProbs,[N_s,N_probs,N_j]));
II2=repmat((1:1:N_s)',1,N_probs);
Gammatranspose_J=cell(N_j-1,1);
for jj=1:(N_j-1)
    Gammatranspose_J{jj}=gpuArray(sparse(Policy_aprimez(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_zint,N_s)); % Note: sparse() will accumulate at repeated indices
end
clear Policy_aprime Policy_aprimez PolicyProbs

%% Forward algorithm
LogLij=NaN(N,N_j);
PopDist=zeros(N_s,N_j,'gpuArray');
nclamped=0; % gridlottery observations outside the range of model values
if isinf(estimoptions.nindividualsbatch)
    nbatch=1;
    batchsize=N;
else
    batchsize=estimoptions.nindividualsbatch;
    nbatch=ceil(N/batchsize);
end
for bb=1:nbatch
    batchinds=((bb-1)*batchsize+1):min(bb*batchsize,N);
    j0_b=j0(batchinds);
    jlast_b=jlast(batchinds);
    % D: column 1 is the population (no observations), columns 2:end are the individuals currently in the panel
    D=jequaloneDist;
    colid=zeros(0,1); % position within batchinds of the individual in each column 2:end
    for jj=1:N_j
        if bb==1
            PopDist(:,jj)=D(:,1);
        end
        % Individuals whose first age is jj enter with the population distribution
        newi=find(j0_b==jj);
        if ~isempty(newi)
            D=[D,repmat(D(:,1),1,numel(newi))];
            colid=[colid;newi];
        end
        ncol=numel(colid);
        if ncol>0
            % Log emission: log p(observations at age jj | latent grid state), summed over the observed variables
            logE=zeros(N_s,ncol,'gpuArray');
            for vv=1:nObs
                y=gpuArray(reshape(ObsByAge.(ObsNames{vv})(batchinds(colid),jj),[1,ncol]));
                obsd=~isnan(y);
                if ~any(obsd)
                    continue
                end
                m=ModelValues.(ObsNames{vv})(:,jj); % [N_s,1]
                ME=MeasurementError.(ObsNames{vv});
                if strcmp(ME.distribution,'normal') || strcmp(ME.distribution,'t')
                    sigma=MEvals.(ObsNames{vv}).sigma(jj);
                    if strcmp(ME.type,'additive')
                        r=(y-m)/sigma;
                        logJacobian=0;
                    else % 'log'
                        mpos=m;
                        mpos(m<=0)=NaN; % multiplicative error cannot generate a positive observation from a non-positive model value
                        r=(log(y)-log(mpos))/sigma;
                        logJacobian=log(y);
                    end
                    if strcmp(ME.distribution,'normal')
                        logEv=-0.5*r.^2-log(sqrt(2*pi)*sigma)-logJacobian;
                    else % 't'
                        nu=MEvals.(ObsNames{vv}).dof(jj);
                        logEv=gammaln((nu+1)/2)-gammaln(nu/2)-0.5*log(nu*pi)-log(sigma)-((nu+1)/2)*log(1+r.^2/nu)-logJacobian;
                    end
                    if strcmp(ME.type,'log')
                        logEv(isnan(logEv) & ~isnan(y) & isnan(mpos))=-Inf;
                    end
                elseif strcmp(ME.distribution,'exact')
                    logEv=zeros(N_s,ncol,'gpuArray');
                    logEv(abs(y-m)>exacttolerance*max(1,abs(y)))=-Inf;
                elseif strcmp(ME.distribution,'misclassified')
                    p=MEvals.(ObsNames{vv}).misclassprob(jj);
                    K=MEvals.(ObsNames{vv}).K;
                    match=(abs(y-m)<=exacttolerance*max(1,abs(y))); % [N_s,ncol]
                    logEv=zeros(N_s,ncol,'gpuArray');
                    logEv(match)=log(1-p); % assign rather than multiply, as log(0)*0 would be NaN when p=0 or p=1
                    logEv(~match)=log(p/(K-1));
                elseif strcmp(ME.distribution,'gridlottery')
                    vals=unique(m); % sorted distinct model values at this age
                    K=numel(vals);
                    if K==1
                        logEv=zeros(N_s,ncol,'gpuArray');
                        nclamped=nclamped+sum(y(obsd)~=vals);
                    else
                        idx=interp1(vals,gpuArray((1:1:K)'),y,'linear'); % fractional position of y among the values (NaN outside)
                        nclamped=nclamped+sum(obsd & (y<vals(1) | y>vals(K)));
                        idx(obsd & y<vals(1))=1;
                        idx(obsd & y>vals(K))=K;
                        idx(~obsd)=1;
                        klow=min(floor(idx),K-1);
                        wup=idx-klow;
                        Ev=(m==reshape(vals(klow),[1,ncol])).*(1-wup)+(m==reshape(vals(klow+1),[1,ncol])).*wup;
                        logEv=log(Ev);
                    end
                end
                logEv(:,~obsd)=0;
                logE=logE+logEv;
            end
            % Normalise (in logs, relative to the largest emission on the support of each individual's distribution, to avoid underflow)
            Dind=D(:,2:end);
            logEsupport=logE;
            logEsupport(Dind<=0)=-Inf;
            c=max(logEsupport,[],1); % [1,ncol]
            cfinite=c;
            cfinite(~isfinite(c))=0;
            Joint=Dind.*exp(logE-cfinite);
            Joint(Dind<=0)=0; % so that exp(-Inf+Inf) never enters
            Lnorm=sum(Joint,1);
            templogL=log(Lnorm)+cfinite;
            templogL(c==-Inf)=-Inf; % no grid state with positive probability can generate the observations
            LogLij(batchinds(colid),jj)=gather(templogL');
            zz=(Lnorm==0 | ~isfinite(templogL));
            Joint(:,zz)=Dind(:,zz); % these individuals have zero likelihood; keep their prior so that nothing becomes NaN
            Lnorm(zz)=1;
            D(:,2:end)=Joint./Lnorm;
        end
        % Individuals whose last age is jj leave
        if ncol>0
            leave=(jlast_b(colid)==jj);
            if any(leave)
                D(:,[false;leave])=[];
                colid(leave)=[];
            end
        end
        if jj==N_j
            break
        end
        if isempty(colid) && ~any(j0_b>jj) && bb>1
            break % nobody left in this batch (the first batch keeps going, to fill in PopDist)
        end
        % Move forward to age jj+1
        ncols=size(D,2);
        D=Gammatranspose_J{jj}*D; % [N_a*N_z,ncols]: aprime (and z stays where it is)
        D=reshape(permute(reshape(D,[N_a,N_zint,ncols]),[1,3,2]),[N_a*ncols,N_zint])*pi_z_J(:,:,jj); % z to zprime
        D=reshape(permute(reshape(D,[N_a,ncols,N_zint]),[1,3,2]),[N_a*N_zint,1,ncols]);
        D=reshape(D.*reshape(pi_e_J(:,jj+1),[1,N_eint]),[N_s,ncols]); % put e back in: pi_e_J(:,jj+1) is the distribution of the e realized in period jj+1
    end
end

if nclamped>0
    warning('LogLikelihood_FHorz: %i gridlottery observations were outside the range of the model values and were put on the nearest end point',nclamped)
end

%% Individual and total log-likelihood
LogLij_sum=LogLij;
if strcmp(estimoptions.initialcondition,'conditional')
    LogLij_sum(sub2ind([N,N_j],(1:1:N)',j0))=NaN; % leave out the first age
end
LogLij_sum(isnan(LogLij_sum))=0;
LogLi=sum(LogLij_sum,2);
LogL=sum(LogLi);

%% PopDist in the same shape as StationaryDist
if N_z==0 && N_e==0
    PopDist=reshape(PopDist,[n_a,N_j]);
elseif N_e==0
    PopDist=reshape(PopDist,[n_a,n_z,N_j]);
elseif N_z==0
    PopDist=reshape(PopDist,[n_a,simoptions.n_e,N_j]);
else
    PopDist=reshape(PopDist,[n_a,n_z,simoptions.n_e,N_j]);
end

end
