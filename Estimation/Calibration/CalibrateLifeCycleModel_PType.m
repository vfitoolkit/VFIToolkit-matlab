function [CalibParams,calibsummary]=CalibrateLifeCycleModel_PType(CalibParamNames,TargetMoments,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, PTypeDistParamNames, ParametrizePTypeFn, FnsToEvaluate, caliboptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats and
% LifeCycleProfiles. Lastly there is caliboptions.


%% Setup caliboptions
if ~isfield(caliboptions,'verbose')
    caliboptions.verbose=1; % sum of squares is the default
end
if ~isfield(caliboptions,'constrainpositive')
    caliboptions.constrainpositive={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    % Convert constrained positive p into x=log(p) which is unconstrained.
    % Then use p=exp(x) in the model.
end
if ~isfield(caliboptions,'constrain0to1')
    caliboptions.constrain0to1={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    % Handle 0 to 1 constraints by using log-odds function to switch parameter p into unconstrained x, so x=log(p/(1-p))
    % Then use the logistic-sigmoid p=1/(1+exp(-x)) when evaluating model.
end
if ~isfield(caliboptions,'constrainAtoB')
    caliboptions.constrainAtoB={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    % Handle A to B constraints by converting y=(p-A)/(B-A) which is 0 to 1, and then treating as constrained 0 to 1 y (so convert to unconstrained x using log-odds function)
    % Once we have the 0 to 1 y (by converting unconstrained x with the logistic sigmoid function), we convert to p=A+(B-a)*y
else
    if ~isfield(caliboptions,'constrainAtoBlimits')
        error('You have used caliboptions.constrainAtoB, but are missing caliboptions.constrainAtoBlimits')
    end
end
if ~isfield(caliboptions,'logmoments')
    caliboptions.logmoments=0;
    % =1 means log() the model moments [target moments and CoVarMatrixDataMoments should already be based on log(moments) if you are using this+
    % =1 means applies log() to all moments, unless you specify them seperately as on next line
    % You can name moments in the same way you would for the targets, e.g.
    % caliboptions.logmoments.AgeConditionalStats.earnings.Mean=1
    % Will log that moment, but not any other moments.
    % Note: the input target moment should log(moment). Same for the covariance matrix
    % of the data moments, CoVarMatrixDataMoments, should be of the log moments.
end
if ~isfield(caliboptions,'metric')
    caliboptions.metric='sum_squared'; % sum of squares is the default
    % Other options are: sum_logratiosquared: sum of squares of the log-ratio (target/model)
end
if ~isfield(caliboptions,'weights')
    caliboptions.weights=1; % all moments have equal weights is default (this is a vector of one, just don't know the length yet :)
end
if ~isfield(caliboptions,'toleranceparams')
    caliboptions.toleranceparams=10^(-4); % tolerance accuracy of the calibrated parameters
end
if ~isfield(caliboptions,'toleranceobjective')
    caliboptions.toleranceobjective=10^(-6); % tolerance accuracy of the objective function
end
if ~isfield(caliboptions,'fminalgo')
    caliboptions.fminalgo=4; % CMA-ES by default, I tried fminsearch() by default but it regularly fails to converge to a decent solution
end
caliboptions.simulatemoments=0; % Not needed here (the objectivefn is shared with other estimation commands)
caliboptions.vectoroutput=0; % Not needed here (the objectivefn is shared with other estimation commands)



%% Set up Names_i and N_i
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

%% Setup for which parameters are being calibrated
% First figure out how many parameters there are (tricky as they can be dependent on ptype)
nCalibParams=0;
nCalibParamsFinder=[]; % rows are the nCalibParams, first column is pp, second column is ii
for pp=1:length(CalibParamNames)
    if isstruct(Parameters.(CalibParamNames{pp}))
        for ii=1:N_i
            if isfield(Parameters.(CalibParamNames{pp}),Names_i{ii})
                nCalibParams=nCalibParams+1;
                nCalibParamsFinder(nCalibParams,1)=pp;
                nCalibParamsFinder(nCalibParams,2)=ii;
            end
        end
    else
        nCalibParams=nCalibParams+1;
        nCalibParamsFinder(nCalibParams,1)=pp;
        nCalibParamsFinder(nCalibParams,2)=0;
    end
end


% Backup the parameter constraint names, so I can replace them with vectors
caliboptions.constrainpositivenames=caliboptions.constrainpositive;
caliboptions.constrainpositive=zeros(nCalibParams,1); % if equal 1, then that parameter is constrained to be positive
caliboptions.constrain0to1names=caliboptions.constrain0to1;
caliboptions.constrain0to1=zeros(nCalibParams,1); % if equal 1, then that parameter is constrained to be 0 to 1
caliboptions.constrainAtoBnames=caliboptions.constrainAtoB;
caliboptions.constrainAtoB=zeros(nCalibParams,1); % if equal 1, then that parameter is constrained to be 0 to 1
if ~isempty(caliboptions.constrainAtoBnames)
    caliboptions.constrainAtoBlimitsnames=caliboptions.constrainAtoBlimits;
    caliboptions.constrainAtoBlimits=zeros(nCalibParams,2); % rows are parameters, column is lower (A) and upper (B) bounds [row will be [0,0] is unconstrained]
end


% Sometimes we want to omit parameters
if isfield(caliboptions,'omitcalibparam')
    OmitCalibParamsNames=fieldnames(caliboptions.omitcalibparam);
else
    OmitCalibParamsNames={''};
end
calibparamsvec0=[]; % column vector
calibparamsvecindex=zeros(nCalibParams+1,1); % Note, first element remains zero
calibparamssizes=zeros(nCalibParams,1); % with PType, some parameters may be matrices (depend on both j and i)
calibomitparams_counter=zeros(nCalibParams,1); % column vector: calibomitparamsvec allows omiting the parameter for certain ages
calibomitparamsmatrix=zeros(N_j,1); % Each row is of size N_j-by-1 and holds the omited values of a parameter
for pp=1:nCalibParams
    if nCalibParamsFinder(pp,2)==0 % Doesn't depend on ptype
        currentparameter=Parameters.(CalibParamNames{nCalibParamsFinder(pp,1)});
    else % depends on ptype
        currentparameter=Parameters.(CalibParamNames{nCalibParamsFinder(pp,1)}).(Names_i{nCalibParamsFinder(pp,2)});
    end
    
    calibparamssizes(pp,1:2)=size(currentparameter);
    % Get all the parameters
    if any(strcmp(OmitCalibParamsNames,CalibParamNames{nCalibParamsFinder(pp,1)})) % Omitting part of parameters cannot differ across permanent types
        % This parameter is under an omit-mask, so need to only use part of it
        tempparam=currentparameter;
        tempomitparam=caliboptions.omitcalibparam.(CalibParamNames{nCalibParamsFinder(pp,1)});
        % Make them both column vectors
        if size(tempparam,1)==1
            tempparam=tempparam';
        end
        if size(tempparam,1)==1
            tempomitparam=tempomitparam';
        end
        % If the omit and initial guess do not fit together, throw an error
        if ~all(tempomitparam(~isnan(tempomitparam))==tempparam(~isnan(tempomitparam)))
            fprintf('Following are the name, omit value, and initial value that related to following error (they should be the same in the non-NaN entries to be calibrated) \n')
            CalibParamNames{pp}
            caliboptions.omitcalibparam.(CalibParamNames{nCalibParamsFinder(pp,1)})
            currentparameter
            error('You have set an omitted calibrated parameter, but the set values do not match the initial guess')
        end
        tempparam=tempparam(isnan(tempomitparam)); % only keep those which are NaN, not those with value for omitted
        % Keep the parts which should be calibrated
        calibparamsvec0=[calibparamsvec0; tempparam]; % Note: it is already a column
        calibparamsvecindex(pp+1)=calibparamsvecindex(pp)+length(tempparam);
        % Store the whole thing
        calibomitparams_counter(pp)=1;
        calibomitparamsmatrix(:,sum(calibomitparams_counter))=tempomitparam;
    else
        % Get all the parameters
        if size(currentparameter,2)==1
            calibparamsvec0=[calibparamsvec0; currentparameter];
        else
            calibparamsvec0=[calibparamsvec0; currentparameter']; % transpose
        end
        calibparamsvecindex(pp+1)=calibparamsvecindex(pp)+length(currentparameter);
    end
    
    % If the parameter is constrained in some way then we need to transform it

    % Contraints cannot differ across ptypes
    
    % First, check the name, and convert it if relevant
    if any(strcmp(caliboptions.constrainpositivenames,CalibParamNames{nCalibParamsFinder(pp,1)}))
        caliboptions.constrainpositive(pp)=1;
    end
    if any(strcmp(caliboptions.constrain0to1names,CalibParamNames{nCalibParamsFinder(pp,1)}))
        caliboptions.constrain0to1(pp)=1;
    end
    if any(strcmp(caliboptions.constrainAtoBnames,CalibParamNames{nCalibParamsFinder(pp,1)}))
        % For parameters A to B, I convert via 0 to 1
        caliboptions.constrain0to1(pp)=1;
        caliboptions.constrainAtoB(pp)=1;
        caliboptions.constrainAtoBlimits(pp,:)=caliboptions.constrainAtoBlimitsnames.(CalibParamNames{nCalibParamsFinder(pp,1)});
    end
    if caliboptions.constrainpositive(pp)==1
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        calibparamsvec0(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=max(log(calibparamsvec0(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))),-49.99);
        % Note, the max() is because otherwise p=0 returns -Inf. [Matlab evaluates exp(-50) as about 10^-22, I overrule and use exp(-50) as zero, so I set -49.99 here so solver can realise the boundary is there; not sure if this setting -49.99 instead of my -50 cutoff actually helps, but seems like it might so I have done it here].
    end
    if caliboptions.constrainAtoB(pp)==1
        % Constraint parameter to be A to B (by first converting to 0 to 1, and then treating it as contraint 0 to 1)
        calibparamsvec0(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=(calibparamsvec0(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))-caliboptions.constrainAtoBlimits(pp,1))/(caliboptions.constrainAtoBlimits(pp,2)-caliboptions.constrainAtoBlimits(pp,1));
        % x=(y-A)/(B-A), converts A-to-B y, into 0-to-1 x
        % And then the next if-statement converts this 0-to-1 into unconstrained
    end
    if caliboptions.constrain0to1(pp)==1
        % Constrain parameter to be 0 to 1 (be working with log(p/(1-p)), where p is parameter) then always take exp()/(1+exp()) before inputting to model
        calibparamsvec0(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=min(49.99,max(-49.99,  log(calibparamsvec0(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))/(1-calibparamsvec0(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)))) ));
        % Note: the max() and min() are because otherwise p=0 or 1 returns -Inf or Inf [Matlab evaluates 1/(1+exp(-50)) as one, and 1/(1+exp(50)) as about 10^-22, so I overrule them as 1 and 0, so I set -49.99 here so solver can realise the boundary is there; not sure if this setting -49.99 instead of my -50 cutoff actually helps, but seems like it might so I have done it here].
    end
    if caliboptions.constrainpositive(pp)==1 && caliboptions.constrain0to1(pp)==1 % Double check of inputs
        fprinf(['Relating to following error message: Parameter ',num2str(pp),' of ',num2str(length(CalibParamNames))])
        error('You cannot constrain parameter twice (you are constraining one of the parameters using both caliboptions.constrainpositive and in one of caliboptions.constrain0to1 and caliboptions.constrainAtoB')
    end
end




%% Setup for which moments are being targeted
% Only calculate each of AllStats and LifeCycleProfiles when being used (so as faster when not using both)
if isfield(TargetMoments,'AllStats')
    usingallstats=1;
else
    usingallstats=0;
end
if isfield(TargetMoments,'AgeConditionalStats')
    usinglcp=1;
else
    usinglcp=0;
end

% NEED TO ADD A CHECK THAT THE INPUT TARGETS ARE THE CORRECT SIZES!!!


% PType means we need the third level a3vec

% Get all of the moments out of TargetMoments and make them into a vector
% Also, store all the names
targetmomentvec=[]; % Can't preallocate as have no idea how big this will be
% Ends up a colmumn vector (create row vector, then transpose)

% First, do those in AllStats
if usingallstats==1
    allstatmomentnames=cell(1,3);
    allstatmomentcounter=0;
    allstatmomentsizes=0;
    if isfield(TargetMoments,'AllStats')
        a1vec=fieldnames(TargetMoments.AllStats); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}));% These will be Mean, etc
                    for a3=1:length(a3vec)
                        allstatmomentcounter=allstatmomentcounter+1;
                        if size(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}),2)==1 % already column vector
                            targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})]; % append to end
                        else
                            targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})']; % transpose, then append to end
                        end
                        allstatmomentnames(allstatmomentcounter,:)={a1vec{a1},a2vec{a2},a3vec{a3}};
                        allstatmomentsizes(allstatmomentcounter)=length(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));
                    end
                else
                    allstatmomentcounter=allstatmomentcounter+1;
                    if size(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}),2)==1 % already column vector
                        targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2})]; % append to end
                    else
                        targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2})']; % transpose, then append to end
                    end
                    allstatmomentnames(allstatmomentcounter,1:2)={a1vec{a1},a2vec{a2}};
                    allstatmomentsizes(allstatmomentcounter)=length(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}));
                end
            end
        end
    end
    allstatcummomentsizes=cumsum(allstatmomentsizes); % Note: this is zero is AllStats is unused
    % To do AllStats faster, we use simoptions.whichstats so that we only compute the stats we want.
    AllStats_whichstats=zeros(7,1);
    if any(strcmp(allstatmomentnames(:,2),'Mean'))
        AllStats_whichstats(1)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'Median'))
        AllStats_whichstats(2)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'RatioMeanToMedian'))
        AllStats_whichstats(1)=1;
        AllStats_whichstats(2)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'Variance')) || any(strcmp(allstatmomentnames(:,2),'StdDeviation'))
        AllStats_whichstats(3)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'LorenzCurve')) || any(strcmp(allstatmomentnames(:,2),'Gini'))
        AllStats_whichstats(4)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'Maximum')) || any(strcmp(allstatmomentnames(:,2),'Minimum'))
        AllStats_whichstats(5)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'QuantileCutoffs')) || any(strcmp(allstatmomentnames(:,2),'QuantileMeans'))
        AllStats_whichstats(5)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'MoreInequality'))
        AllStats_whichstats(7)=1;
    end
    if any(strcmp(allstatmomentnames(:,3),'Mean'))
        AllStats_whichstats(1)=1;
    end
    if any(strcmp(allstatmomentnames(:,3),'Median'))
        AllStats_whichstats(2)=1;
    end
    if any(strcmp(allstatmomentnames(:,3),'Variance')) || any(strcmp(allstatmomentnames(:,3),'StdDeviation'))
        AllStats_whichstats(3)=1;
    end
    if any(strcmp(allstatmomentnames(:,3),'LorenzCurve')) || any(strcmp(allstatmomentnames(:,3),'Gini'))
        AllStats_whichstats(4)=1;
    end
    if any(strcmp(allstatmomentnames(:,3),'Maximum')) || any(strcmp(allstatmomentnames(:,3),'Minimum'))
        AllStats_whichstats(5)=1;
    end
    if any(strcmp(allstatmomentnames(:,3),'QuantileCutoffs')) || any(strcmp(allstatmomentnames(:,3),'QuantileMeans'))
        AllStats_whichstats(5)=1;
    end
    if any(strcmp(allstatmomentnames(:,3),'MoreInequality'))
        AllStats_whichstats(7)=1;
    end
else
    % Placeholders
    allstatmomentnames=cell(1,3);
    allstatcummomentsizes=0;
    AllStats_whichstats=zeros(7,1);
end





% Second, do those in AgeConditionalStats
if usinglcp==1
    acsmomentnames=cell(1,3);
    acsmomentcounter=0;
    acsmomentsizes=0;
    if isfield(TargetMoments,'AgeConditionalStats')
        a1vec=fieldnames(TargetMoments.AgeConditionalStats); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}));% These will be Mean, etc
                    for a3=1:length(a3vec)
                        acsmomentcounter=acsmomentcounter+1;
                        if size(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}),2)==1 % already column vector
                            targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})]; % append to end
                        else
                            targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})']; % transpose, then append to end
                        end
                        acsmomentnames(acsmomentcounter,:)={a1vec{a1},a2vec{a2},a3vec{a3}};
                        acsmomentsizes(acsmomentcounter)=length(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));
                    end
                else
                    acsmomentcounter=acsmomentcounter+1;
                    if size(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}),2)==1 % already column vector
                        targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2})]; % append to end
                    else
                        targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2})']; % transpose, then append to end
                    end
                    acsmomentnames(acsmomentcounter,1:2)={a1vec{a1},a2vec{a2}};
                    acsmomentsizes(acsmomentcounter)=length(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}));
                end
            end
        end
    end
    acscummomentsizes=cumsum(acsmomentsizes); % Note: this is zero is AgeConditionalStats is unused
    % To do AgeConditionalStats faster, we use simoptions.whichstats so that we only compute the stats we want.
    ACStats_whichstats=zeros(7,1);
    if any(strcmp(acsmomentnames(:,2),'Mean'))
        ACStats_whichstats(1)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'Median'))
        ACStats_whichstats(2)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'RatioMeanToMedian'))
        ACStats_whichstats(1)=1;
        ACStats_whichstats(2)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'Variance')) || any(strcmp(acsmomentnames(:,2),'StdDeviation'))
        ACStats_whichstats(3)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'LorenzCurve')) || any(strcmp(acsmomentnames(:,2),'Gini'))
        ACStats_whichstats(4)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'Maximum')) || any(strcmp(acsmomentnames(:,2),'Minimum'))
        ACStats_whichstats(5)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'QuantileCutoffs')) || any(strcmp(acsmomentnames(:,2),'QuantileMeans'))
        ACStats_whichstats(5)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'MoreInequality'))
        ACStats_whichstats(7)=1;
    end
    if any(strcmp(acsmomentnames(:,3),'Mean'))
        ACStats_whichstats(1)=1;
    end
    if any(strcmp(acsmomentnames(:,3),'Median'))
        ACStats_whichstats(2)=1;
    end
    if any(strcmp(acsmomentnames(:,3),'Variance')) || any(strcmp(acsmomentnames(:,3),'StdDeviation'))
        ACStats_whichstats(3)=1;
    end
    if any(strcmp(acsmomentnames(:,3),'LorenzCurve')) || any(strcmp(acsmomentnames(:,3),'Gini'))
        ACStats_whichstats(4)=1;
    end
    if any(strcmp(acsmomentnames(:,3),'Maximum')) || any(strcmp(acsmomentnames(:,3),'Minimum'))
        ACStats_whichstats(5)=1;
    end
    if any(strcmp(acsmomentnames(:,3),'QuantileCutoffs')) || any(strcmp(acsmomentnames(:,3),'QuantileMeans'))
        ACStats_whichstats(5)=1;
    end
    if any(strcmp(acsmomentnames(:,3),'MoreInequality'))
        ACStats_whichstats(7)=1;
    end
else
    % Placeholders
    acsmomentnames=cell(1,3);
    acscummomentsizes=0;
    ACStats_whichstats=zeros(7,1);
end
% age-conditional stats should be of length N_j
for ii=1:length(acsmomentsizes)
    if acsmomentsizes(ii)~=N_j
        errorstr=['Target Age-Conditional Stats must be of length() N_j (if you want to ignore some ages, use NaN for those ages); problem is with ', acsmomentnames{ii,1}, ' ', acsmomentnames{ii,2}, ' ',acsmomentnames{ii,3},' \n'];
        error(errorstr)
    end
end


%% Set-up/check caliboptions.weights
actualtarget=(~isnan(targetmomentvec)); % I use NaN to omit targets
if isscalar(caliboptions.weights)
    caliboptions.weights=caliboptions.weights.*ones(size(targetmomentvec(actualtarget)));
end
if length(caliboptions.weights)~=length(targetmomentvec(actualtarget))
    error('caliboptions.weights is not the length same as number of target moments (ignoring any NaN)')
end

%% Now, a bunch of things to avoid redoing them every parameter vector we want to try
% Note: I avoid doing this for ReturnFnParamNames because they are so
% dependent on the setup. Same for FnsToEvaluateParamNames
ReturnFnParamNames=[];
FnsToEvaluateParamNames=[];


% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
% Gradually rolling these out so that all the commands build off of these
z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
pi_z_J=zeros(prod(n_z),prod(n_z),'gpuArray');
if isfield(vfoptions,'ExogShockFn')
    if isfield(vfoptions,'ExogShockFnParamNames')
        for jj=1:N_j
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            pi_z_J(:,:,jj)=gpuArray(pi_z);
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
        end
    else
        for jj=1:N_j
            [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
            pi_z_J(:,:,jj)=gpuArray(pi_z);
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
        end
    end
elseif prod(n_z)==0 % no z
    z_gridvals_J=[];
elseif ndims(z_grid)==3 % already an age-dependent joint-grid
    if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
        z_gridvals_J=z_grid;
    end
    pi_z_J=pi_z;
elseif all(size(z_grid)==[sum(n_z),N_j]) % age-dependent grid
    for jj=1:N_j
        z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid(:,jj),1);
    end
    pi_z_J=pi_z;
elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
    z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
    pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
    z_gridvals_J=CreateGridvals(n_z,z_grid,1).*ones(1,1,N_j,'gpuArray');
    pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
end

% If using e variable, do same for this
if isfield(vfoptions,'n_e')
    if prod(vfoptions.n_e)==0
        vfoptions=rmfield(vfoptions,'n_e');
    else
        if isfield(vfoptions,'e_grid_J')
            error('No longer use vfoptions.e_grid_J, instead just put the age-dependent grid in vfoptions.e_grid (functionality of VFI Toolkit has changed to make it easier to use)')
        end
        if ~isfield(vfoptions,'e_grid') % && ~isfield(vfoptions,'e_grid_J')
            error('You are using an e (iid) variable, and so need to declare vfoptions.e_grid')
        elseif ~isfield(vfoptions,'pi_e')
            error('You are using an e (iid) variable, and so need to declare vfoptions.pi_e')
        end

        vfoptions.e_gridvals_J=zeros(prod(vfoptions.n_e),length(vfoptions.n_e),'gpuArray');
        vfoptions.pi_e_J=zeros(prod(vfoptions.n_e),prod(vfoptions.n_e),'gpuArray');

        if isfield(vfoptions,'EiidShockFn')
            if isfield(vfoptions,'EiidShockFnParamNames')
                for jj=1:N_j
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [vfoptions.e_grid,vfoptions.pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
                    vfoptions.pi_e_J(:,jj)=gpuArray(vfoptions.pi_e);
                    if all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1])
                        vfoptions.e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1));
                    else % already joint-grid
                        vfoptions.e_gridvals_J(:,:,jj)=gpuArray(vfoptions.e_grid,1);
                    end
                end
            else
                for jj=1:N_j
                    [vfoptions.e_grid,vfoptions.pi_e]=vfoptions.EiidShockFn(N_j);
                    vfoptions.pi_e_J(:,jj)=gpuArray(vfoptions.pi_e);
                    if all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1])
                        vfoptions.e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1));
                    else % already joint-grid
                        vfoptions.e_gridvals_J(:,:,jj)=gpuArray(vfoptions.e_grid,1);
                    end
                end
            end
        elseif ndims(vfoptions.e_grid)==3 % already an age-dependent joint-grid
            if all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e),N_j])
                vfoptions.e_gridvals_J=vfoptions.e_grid;
            end
            vfoptions.pi_e_J=vfoptions.pi_e;
        elseif all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),N_j]) % age-dependent stacked-grid
            for jj=1:N_j
                vfoptions.e_gridvals_J(:,:,jj)=CreateGridvals(vfoptions.n_e,vfoptions.e_grid(:,jj),1);
            end
            vfoptions.pi_e_J=vfoptions.pi_e;
        elseif all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e)]) % joint grid
            vfoptions.e_gridvals_J=vfoptions.e_grid.*ones(1,1,N_j,'gpuArray');
            vfoptions.pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
        elseif all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1]) % basic grid
            vfoptions.e_gridvals_J=CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1).*ones(1,1,N_j,'gpuArray');
            vfoptions.pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
        end
    end
    simoptions.e_gridvals_J=vfoptions.e_gridvals_J;
    simoptions.pi_e_J=vfoptions.pi_e_J;
end



%% 
% caliboptions.logmoments can be specified by names
if isstruct(caliboptions.logmoments)
    logmomentnames=caliboptions.logmoments;
    % replace caliboptions.logmoments with a vector as this is what gets used internally
    caliboptions.logmoments=zeros(length(targetmomentvec),1);
    if any(fieldnames(logmomentnames),'AllStats')
        caliboptions.logmoments(1:allstatcummomentsizes(1))=caliboptions.logmoments.AllStats.(allstatmomentnames{1,1}).(allstatmomentnames{1,2})*ones(allstatcummomentsizes(1),1);
        for ii=2:size(allstatmomentnames,1)
            caliboptions.logmoments(allstatcummomentsizes(ii-1)+1:allstatcummomentsizes(ii))=caliboptions.logmoments.AllStats.(allstatmomentnames{ii,1}).(allstatmomentnames{ii,2})*ones(allstatcummomentsizes(ii)-allstatcummomentsizes(ii-1),1);
        end
    end
    if any(fieldnames(logmomentnames),'AgeConditionalStats')
        caliboptions.logmoments(1:acscummomentsizes(1))=caliboptions.logmoments.AllStats.(acsmomentnames{1,1}).(acsmomentnames{1,2})*ones(acscummomentsizes(1),1);
        for ii=2:size(acsmomentnames,1)
            caliboptions.logmoments(acscummomentsizes(ii-1)+1:acscummomentsizes(ii))=caliboptions.logmoments.AllStats.(acsmomentnames{ii,1}).(acsmomentnames{ii,2})*ones(acscummomentsizes(ii)-acscummomentsizes(ii-1),1);
        end
    end

% If caliboptions.logmoments is not a structure, then...
% caliboptions.logmoments will either be scalar, or a vector of zeros and ones
%    [scalar of zero is interpreted as vector of zeros, scalar of one is interpreted as vector of ones]
elseif any(caliboptions.logmoments>0) % =1 means log of moments (can be set up as vector, zeros(length(CalibParamNames),1)
   % If set this up, and then set up 
   if isscalar(caliboptions.logmoments)
       caliboptions.logmoments=ones(length(targetmomentvec),1); % log all of them
   else
        if length(caliboptions.logmoments)==(length(acsmomentnames)+length(allstatmomentnames))
            % Covert caliboptions.logmoments from being about CalibParamNames
            temp=caliboptions.logmoments;
            caliboptions.logmoments=zeros(length(targetmomentvec),1);
            cumsofar=1;
            for mm=1:length(temp)
                if mm<=allstatmomentsizes
                    caliboptions.logmoments(cumsofar:cumsofar+allstatmomentsizes(mm))=temp(mm);
                    cumsofar=cumsofar+allstatmomentsizes(mm);
                else
                    caliboptions.logmoments(cumsofar:cumsofar+acsmomentsizes(mm))=temp(mm);
                    cumsofar=cumsofar+acsmomentsizes(mm);
                end
            end
        elseif length(caliboptions.logmoments)==length(targetmomentvec)
            % This is fine (already in the appropriate form)
        else
            fprintf('Relevant to following error: length(caliboptions.logmoments)=%i \n', length(caliboptions.logmoments))
            fprintf('Relevant to following error: length(acsmomentnames)=%i, length(allstatmomentnames)=%i \n', length(acsmomentnames), length(allstatmomentnames))
            error('You are using caliboptions.logmoments, but length(caliboptions.logmoments) does not match number of moments to calibrate [they should be equal]')
        end
   end
   % log of targetmoments [no need to do this as inputs should already be log()]
   % targetmomentvec=(1-caliboptions.logmoments).*targetmomentvec + caliboptions.logmoments.*log(targetmomentvec.*caliboptions.logmoments+(1-caliboptions.logmoments)); % Note: take log, and for those we don't log I end up taking log(1) (which becomes zero and so disappears)
end



%% Set up the objective function and the initial calibration parameter vector
CalibrationObjectiveFn=@(calibparamsvec) CalibrateLifeCycleModel_PType_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames,PTypeDistParamNames, ParametrizePTypeFn, FnsToEvaluate, usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, nCalibParams, nCalibParamsFinder, calibparamsvecindex, calibparamssizes, calibomitparams_counter, calibomitparamsmatrix, caliboptions, vfoptions, simoptions);

% calibparamsvec0 is our initial guess for calibparamsvec


%% Choosing algorithm for the optimization problem
% https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
minoptions = optimset('TolX',caliboptions.toleranceparams,'TolFun',caliboptions.toleranceobjective);
if caliboptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
    caliboptions.multiGEcriterion=0;
    [calibparamsvec,calibobjvalue]=fzero(CalibrationObjectiveFn,calibparamsvec0,minoptions);    
elseif caliboptions.fminalgo==1
    [calibparamsvec,calibobjvalue]=fminsearch(CalibrationObjectiveFn,calibparamsvec0,minoptions);
elseif caliboptions.fminalgo==2
    % Use the optimization toolbox so as to take advantage of automatic differentiation
    z=optimvar('z',length(calibparamsvec0));
    optimfun=fcn2optimexpr(CalibrationObjectiveFn, z);
    prob = optimproblem("Objective",optimfun);
    z0.z=calibparamsvec0;
    [sol,calibobjvalue]=solve(prob,z0);
    calibparamsvec=sol.z;
    % Note, doesn't really work as automatic differentiation is only for
    % supported functions, and the objective here is not a supported function
elseif caliboptions.fminalgo==3
    goal=zeros(length(calibparamsvec0),1);
    weight=ones(length(calibparamsvec0),1); % I already implement weights via caliboptions
    [calibparamsvec,calibsummaryVec] = fgoalattain(CalibrationObjectiveFn,calibparamsvec0,goal,weight);
    calibobjvalue=sum(abs(calibsummaryVec));
elseif caliboptions.fminalgo==4 % CMA-ES algorithm (Covariance-Matrix adaptation - Evolutionary Stategy)
    % https://en.wikipedia.org/wiki/CMA-ES
    % https://cma-es.github.io/
    % Code is cmaes.m from: https://cma-es.github.io/cmaes_sourcecode_page.html#matlab
    if ~isfield(caliboptions,'insigma')
        % insigma: initial coordinate wise standard deviation(s)
        caliboptions.insigma=0.3*abs(calibparamsvec0)+0.1*(calibparamsvec0==0); % Set standard deviation to 30% of the initial parameter value itself (cannot input zero, so add 0.1 to any zeros)
    end
    if ~isfield(caliboptions,'inopts')
        % inopts: options struct, see defopts below
        caliboptions.inopts=[];
    end
    % varargin (unused): arguments passed to objective function 
    if caliboptions.verbose==1
        disp('VFI Toolkit is using the CMA-ES algorithm, consider giving a cite to: Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution Strategy on Multimodal Test Functions' )
    end
	% This is a minor edit of cmaes, because I want to use 'CalibrationObjectiveFn' as a function_handle, but the original cmaes code only allows for 'CalibrationObjectiveFn' as a string
    [calibparamsvec,calibobjvalue,counteval,stopflag,out,bestever] = cmaes_vfitoolkit(CalibrationObjectiveFn,calibparamsvec0,caliboptions.insigma,caliboptions.inopts); % ,varargin);
elseif caliboptions.fminalgo==5
    % Update based on rules in caliboptions.fminalgo5.howtoupdate
    error('fminalgo=5 is not possible with model calibration/estimation')
elseif caliboptions.fminalgo==6
    if ~isfield(caliboptions,'lb') || ~isfield(caliboptions,'ub')
        error('When using constrained optimization (caliboptions.fminalgo=6) you must set the lower and upper bounds of the GE price parameters using caliboptions.lb and caliboptions.ub') 
    end
    [calibparamsvec,calibobjvalue]=fmincon(CalibrationObjectiveFn,calibparamsvec0,[],[],[],[],caliboptions.lb,caliboptions.ub,[],minoptions);    
end


%% Clean up output
for pp=1:nCalibParams
    % If parameter is constrained, switch it back to the unconstrained value
    if caliboptions.constrainpositive(pp)==1 % Forcing this parameter to be positive
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=exp(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)));
    elseif caliboptions.constrain0to1(pp)==1
        % Constrain parameter to be 0 to 1 (be working with x=log(p/(1-p)), where p is parameter) then always take 1/(1+exp(-x)) before inputting to model
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=1/(1+exp(-calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))));
    end
    % Note: sometimes, need to do both of constrainAtoB and constrain0to1, so cannot use elseif
    if caliboptions.constrainAtoB(pp)==1
        % Constrain parameter to be A to B
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=caliboptions.constrainAtoBlimits(pp,1)+(caliboptions.constrainAtoBlimits(pp,2)-caliboptions.constrainAtoBlimits(pp,1))*calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        % Note, this parameter will have first been converted to 0 to 1 already, so just need to further make it A to B
        % y=A+(B-A)*x, converts 0-to-1 x, into A-to-B y
    end

    % Now store the unconstrained values
    if calibomitparams_counter(pp)>0
        currparamraw=calibomitparamsmatrix(:,sum(calibomitparams_counter(1:pp)));
        currparamraw(isnan(currparamraw))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        if nCalibParamsFinder(pp,2)==0 % does not depend on ptype
            CalibParams.(CalibParamNames{nCalibParamsFinder(pp,1)})=currparamraw;
        else % depends on ptype
            CalibParams.(CalibParamNames{nCalibParamsFinder(pp,1)}).(Names_i{nCalibParamsFinder(pp,2)})=currparamraw;
        end
    else
        if nCalibParamsFinder(pp,2)==0 % does not depend on ptype
            CalibParams.(CalibParamNames{nCalibParamsFinder(pp,1)})=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        else
            CalibParams.(CalibParamNames{nCalibParamsFinder(pp,1)}).(Names_i{nCalibParamsFinder(pp,2)})=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        end
    end
end
clear calibparamsvec % I modified it, so want to make sure I don't accidently use it again later

calibsummary.objvalue=calibobjvalue; % Output the objective value





end