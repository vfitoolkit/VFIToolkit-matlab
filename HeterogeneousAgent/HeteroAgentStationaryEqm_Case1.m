function varargout=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions, EntryExitParamNames)
% Outputs: [p_eqm, GeneralEqmConditions]
% Unless you use n_p and p_grid, in which case [p_eq, p_eqm_index, GeneralEqmConditions]

% Input heteroagentoptions and those that appear after it are all optional.


%% Check 'double fminalgo'
if exist('heteroagentoptions','var')
    if isfield(heteroagentoptions,'fminalgo')
        if length(heteroagentoptions.fminalgo)>1
            if isfield(heteroagentoptions,'toleranceGEcondns')
                heteroagentoptions.toleranceGEcondns=heteroagentoptions.toleranceGEcondns.*ones(1,length(heteroagentoptions.fminalgo));
            else
                heteroagentoptions.toleranceGEcondns=10^(-4).*ones(1,length(heteroagentoptions.fminalgo)); % Accuracy of general eqm prices
            end
            if isfield(heteroagentoptions,'toleranceGEprices')
                heteroagentoptions.toleranceGEprices=heteroagentoptions.toleranceGEprices.*ones(1,length(heteroagentoptions.fminalgo));
            else
                heteroagentoptions.toleranceGEprices=10^(-4).*ones(1,length(heteroagentoptions.fminalgo)); % Accuracy of general eqm prices
            end
            if ~exist('EntryExitParamNames','var')
                EntryExitParamNames={};
            end
            temp=heteroagentoptions.fminalgo;
            temp2=heteroagentoptions.toleranceGEcondns;
            temp3=heteroagentoptions.toleranceGEprices;
            heteroagentoptions.fminalgo=heteroagentoptions.fminalgo(1:end-1);
            heteroagentoptions.toleranceGEcondns=heteroagentoptions.toleranceGEcondns(1:end-1);
            heteroagentoptions.toleranceGEprices=heteroagentoptions.toleranceGEprices(1:end-1);
            p_eqm_previous=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions, EntryExitParamNames);
            for pp=1:length(GEPriceParamNames)
                Parameters.(GEPriceParamNames{pp})=p_eqm_previous.(GEPriceParamNames{pp});
            end
            heteroagentoptions.fminalgo=temp(end);
            heteroagentoptions.toleranceGEcondns=temp2(end);
            heteroagentoptions.toleranceGEprices=temp3(end);
        end
    end
end


%% Check which options have been used, set all others to defaults 
N_p=prod(n_p);
if isempty(n_p)
    N_p=0;
end

if ~exist('vfoptions','var')
    %If vfoptions is not given, just use all the defaults
    vfoptions.parallel=1+(gpuDeviceCount>0);
    %Note that the defaults will be set when we call 'ValueFnIter...'
    %commands and the like, so no need to set them here except for a few.
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
end

if exist('simoptions','var')==0
    %If vfoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0);    
    %Note that the defaults will be set when we call 'StationaryDist...'
    %commands and the like, so no need to set them here except for a few.
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(vfoptions,'gridinterplayer')
        if ~isfield(simoptions,'gridinterplayer')
            error('When setting vfoptions.gridinterplayer you must also set simoptions.gridinterplayer')
        end
    end
end

if exist('heteroagentoptions','var')==0
    heteroagentoptions.multiGEcriterion=1;
    heteroagentoptions.multiGEweights=ones(1,length(fieldnames(GeneralEqmEqns)));
    heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm eqns
    heteroagentoptions.fminalgo=1; % use fminsearch
    heteroagentoptions.verbose=0;
    heteroagentoptions.maxiter=1000; % maximum iterations of optimization routine
    heteroagentoptions.pricehistory=0; % =1, saves the history of the GEPrices during the convergence
    heteroagentoptions.oldGE=1;
    % Constrain parameters
    heteroagentoptions.constrainpositive={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    heteroagentoptions.constrain0to1={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    heteroagentoptions.constrainAtoB={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    % heteroagentoptions.outputGEform=0; % For internal use only
    heteroagentoptions.outputGEstruct=1; % output GE conditions as a structure (=2 will output as a vector)
    heteroagentoptions.outputgather=1; % output GE conditions on CPU [some optimization routines only work on CPU, some can handle GPU]
else
    if ~isfield(heteroagentoptions,'multiGEcriterion')
        heteroagentoptions.multiGEcriterion=1;
    end
    if ~isfield(heteroagentoptions,'multiGEweights')
        heteroagentoptions.multiGEweights=ones(1,length(fieldnames(GeneralEqmEqns)));
    end
    if N_p~=0
        if ~isfield(heteroagentoptions,'pgrid')
            error('You have set n_p to a non-zero value, but not declared heteroagentoptions.pgrid')
        end
    end
    if ~isfield(heteroagentoptions,'toleranceGEprices')
        heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    end
    if ~isfield(heteroagentoptions,'toleranceGEcondns')
        heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm eqns
    end
    if ~isfield(heteroagentoptions,'fminalgo')
        heteroagentoptions.fminalgo=1; % use fminsearch
    end
    if ~isfield(heteroagentoptions,'verbose')
        heteroagentoptions.verbose=0;
    end
    if ~isfield(heteroagentoptions,'maxiter')
        heteroagentoptions.maxiter=1000; % maximum iterations of optimization routine
    end
    if ~isfield(heteroagentoptions,'pricehistory')
        heteroagentoptions.pricehistory=0; % =1, saves the history of the GEPrices during the convergence
    end
    if ~isfield(heteroagentoptions,'oldGE')
        heteroagentoptions.oldGE=1;
    end
    % Constrain parameters
    if ~isfield(heteroagentoptions,'constrainpositive')
        heteroagentoptions.constrainpositive={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
        % Convert constrained positive p into x=log(p) which is unconstrained.
        % Then use p=exp(x) in the model.
    end
    if ~isfield(heteroagentoptions,'constrain0to1')
        heteroagentoptions.constrain0to1={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
        % Handle 0 to 1 constraints by using log-odds function to switch parameter p into unconstrained x, so x=log(p/(1-p))
        % Then use the logistic-sigmoid p=1/(1+exp(-x)) when evaluating model.
    end
    if ~isfield(heteroagentoptions,'constrainAtoB')
        heteroagentoptions.constrainAtoB={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
        % Handle A to B constraints by converting y=(p-A)/(B-A) which is 0 to 1, and then treating as constrained 0 to 1 y (so convert to unconstrained x using log-odds function)
        % Once we have the 0 to 1 y (by converting unconstrained x with the logistic sigmoid function), we convert to p=A+(B-a)*y
    else
        if ~isfield(heteroagentoptions,'constrainAtoBlimits')
            error('You have used heteroagentoptions.constrainAtoB, but are missing heteroagentoptions.constrainAtoBlimits')
        end
    end
    % heteroagentoptions.outputGEform=0; % For internal use only
    if ~isfield(heteroagentoptions,'outputGEstruct')
        heteroagentoptions.outputGEstruct=1; % output GE conditions as a structure (=2 will output as a vector)
    end
    if ~isfield(heteroagentoptions,'outputgather')
        heteroagentoptions.outputgather=1; % output GE conditions on CPU [some optimization routines only work on CPU, some can handle GPU]
    end
end

heteroagentoptions.useCustomModelStats=0;
if isfield(heteroagentoptions,'CustomModelStats')
    heteroagentoptions.useCustomModelStats=1;
end

if heteroagentoptions.fminalgo==0
    heteroagentoptions.outputGEform=1;
elseif heteroagentoptions.fminalgo==5
    heteroagentoptions.outputGEform=1; % Need to output GE condns as a vector when using fminalgo=5
    heteroagentoptions.outputgather=0; % leave GE condns vector on GPU
elseif heteroagentoptions.fminalgo==7
    heteroagentoptions.outputGEform=1; % Need to output GE condns as a vector when using fminalgo=7
else
    heteroagentoptions.outputGEform=0;
end

temp=size(heteroagentoptions.multiGEweights);
if temp(2)==1 % probably column vector
    heteroagentoptions.multiGEweights=heteroagentoptions.multiGEweights'; % make row vector
end
if length(heteroagentoptions.multiGEweights)~=length(fieldnames(GeneralEqmEqns))
    error('length(heteroagentoptions.multiGEweights)~=length(fieldnames(GeneralEqmEqns)) (the length of the GE weights is not equal to the number of general eqm equations')
end

nGEprices=length(GEPriceParamNames);
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);
AggVarNames=fieldnames(FnsToEvaluate);

simoptions.outputasstructure=0;

if isfield(vfoptions,'divideandconquer') && isfield(vfoptions,'gridinterplayer')
    if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
        vfoptions.divideandconquer=0;
        warning('Cannot yet use divide-and-conquer with grid interpolation layer for InfHorz, so just ignoring the divide-and-conquer (and doing the gird interpolation layer)')
    end
end

%%
N_a=prod(n_a);
N_z=prod(n_z);

% Check if gthere is an initial guess for V0
if isfield(vfoptions,'V0')
    V0=reshape(vfoptions.V0,[N_a,N_z]);
else
    if vfoptions.parallel==2
        V0=zeros([N_a,N_z], 'gpuArray');
    else
        V0=zeros([N_a,N_z]);
    end
end


%% Set up exogenous shock grids now (so they can then just be reused every time)
% Check if using ExogShockFn or EiidShockFn, and if so, do these use a
% parameter that is being determined in general eqm
heteroagentoptions.gridsinGE=0;
if isfield(vfoptions,'ExogShockFn')
    tempExogShockFnParamNames=getAnonymousFnInputNames(vfoptions.ExogShockFn);
    % can just leave action space in here as we only use it to see if GEPriceParamNames is part of it
    if ~isempty(intersect(tempExogShockFnParamNames,GEPriceParamNames))
        heteroagentoptions.gridsinGE=1;
    end        
end
if isfield(vfoptions,'EiidShockFn')
    tempEiidShockFnParamNames=getAnonymousFnInputNames(vfoptions.EiidShockFn);
    % can just leave action space in here as we only use it to see if GEPriceParamNames is part of it
    if ~isempty(intersect(tempEiidShockFnParamNames,GEPriceParamNames))
        heteroagentoptions.gridsinGE=1;
    end        
end
% If z (and e) are not determined in GE, then compute z_gridvals and pi_z now (and e_gridvals and pi_e)
if heteroagentoptions.gridsinGE==0
    if vfoptions.parallel<2 % If on cpu, only allowed to use the very basics
        % z_grid and pi_z are just simple, so leave as is.
        % e is not supported for cpu
    else
        % Some of the shock grids depend on parameters that are determined in general eqm
        [z_grid, pi_z, vfoptions]=ExogShockSetup(n_z,z_grid,pi_z,Parameters,vfoptions,3);
        % Note: these are actually z_gridvals and pi_z
        simoptions.e_gridvals=vfoptions.e_gridvals; % Note, will be [] if no e
        simoptions.pi_e=vfoptions.pi_e; % Note, will be [] if no e
    end
end
% Regardless of whether they are done here of in _subfn, they will be
% precomputed by the time we get to the value fn, staty dist, etc. So
vfoptions.alreadygridvals=1;
simoptions.alreadygridvals=1;


%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
if isstruct(FnsToEvaluate)
    l_d=length(n_d);
    if n_d(1)==0
        l_d=0;
    end
    l_a=length(n_a);
    l_aprime=l_a;
    l_z=length(n_z);
    if n_z(1)==0
        l_z=0;
    end
    if isfield(simoptions,'SemiExoStateFn')
        l_z=l_z+length(simoptions.n_semiz);
    end
    l_e=0;
    if isfield(simoptions,'n_e')
        l_e=length(simoptions.n_e);
        if simoptions.n_e(1)==0
            l_e=0;
        end
    end
    if isfield(simoptions,'experienceasset')
        % One of the endogenous states should only be counted once
        l_aprime=l_aprime-simoptions.experienceasset;
    end
    if isfield(simoptions,'experienceassetu')
        % One of the endogenous states should only be counted once
        l_aprime=l_aprime-simoptions.experienceassetu;
    end
    if isfield(simoptions,'riskyasset')
        % One of the endogenous states should only be counted once
        l_aprime=l_aprime-simoptions.riskyasset;
    end
    if isfield(simoptions,'residualassset')
        % One of the endogenous states should only be counted once
        l_aprime=l_aprime-simoptions.residualassset;
    end

    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_aprime+l_a+l_z+l_e)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_aprime+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
    end
    % Now have FnsToEvaluate as structure, FnsToEvaluateCell as cell
else
    % Do nothing
end


%% If using intermediateEqns, switch from structure to cell setup
heteroagentoptions.useintermediateEqns=0;
if isfield(heteroagentoptions,'intermediateEqns')
    heteroagentoptions.useintermediateEqns=1;
    intEqnNames=fieldnames(heteroagentoptions.intermediateEqns);
    nIntEqns=length(intEqnNames);

    heteroagentoptions.intermediateEqnsCell=cell(1,nIntEqns);
    for gg=1:nIntEqns
        temp=getAnonymousFnInputNames(heteroagentoptions.intermediateEqns.(intEqnNames{gg}));
        heteroagentoptions.intermediateEqnParamNames(gg).Names=temp;
        heteroagentoptions.intermediateEqnsCell{gg}=heteroagentoptions.intermediateEqns.(intEqnNames{gg});        
    end
    % Now:
    %  heteroagentoptions.intermediateEqns is still the structure
    %  heteroagentoptions.intermediateEqnsCell is cell
    %  heteroagentoptions.intermediateEqnParamNames(gg).Names contains the names
end

%% GE eqns, switch from structure to cell setup
GeneralEqmEqnsCell=cell(1,nGeneralEqmEqns);
for gg=1:nGeneralEqmEqns
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{gg}));
    GeneralEqmEqnParamNames(gg).Names=temp;
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns.(GEeqnNames{gg});
end
% Now: 
%  GeneralEqmEqns is still the structure
%  GeneralEqmEqnsCell is cell
%  GeneralEqmEqnParamNames(gg).Names contains the names
% Note: 


%% Set up GEparamsvec0 and parameter constraints
nGEParams=length(GEPriceParamNames);
GEparamsvec0=zeros(nGEParams,1); % column vector
if heteroagentoptions.maxiter>0
    for pp=1:nGEParams
        GEparamsvec0(pp)=Parameters.(GEPriceParamNames{pp});
    end
end

% If the parameter is constrained in some way then we need to transform it
[GEparamsvec0,heteroagentoptions]=ParameterConstraints_TransformParamsToUnconstrained(GEparamsvec0,0:1:nGEParams,GEPriceParamNames,heteroagentoptions,1);
% Also converts the constraints info in estimoptions to be a vector rather than by name.

% If you are saving the price history, preallocate for this
if heteroagentoptions.pricehistory==1
    GEpricepath=zeros(length(GEparamsvec0),heteroagentoptions.maxiter);
    GEcondnpath=zeros(length(fieldnames(GeneralEqmEqns)),heteroagentoptions.maxiter);
    itercount=0;
    save pricehistory.mat GEpricepath GEcondnpath itercount
end


%% Enough setup, Time to do the actual finding the HeteroAgentStationaryEqm:

%% If using fminalgo=5, then need some further setup

if heteroagentoptions.fminalgo==5
    heteroagentoptions.weightscheme=0; % Don't do any weightscheme, is already taken care of by GEnewprice=3
    
    if isstruct(GeneralEqmEqns) 
        % Need to make sure that order of rows in transpathoptions.GEnewprice3.howtoupdate
        % Is same as order of fields in GeneralEqmEqns
        % I do this by just reordering rows of transpathoptions.GEnewprice3.howtoupdate
        temp=heteroagentoptions.fminalgo5.howtoupdate;
        GEeqnNames=fieldnames(GeneralEqmEqns);
        for tt=1:length(GEeqnNames)
            for jj=1:size(temp,1)
                if strcmp(temp{jj,1},GEeqnNames{tt}) % Names match
                    heteroagentoptions.fminalgo5.howtoupdate{tt,1}=temp{jj,1};
                    heteroagentoptions.fminalgo5.howtoupdate{tt,2}=temp{jj,2};
                    heteroagentoptions.fminalgo5.howtoupdate{tt,3}=temp{jj,3};
                    heteroagentoptions.fminalgo5.howtoupdate{tt,4}=temp{jj,4};
                end
            end
        end
        nGeneralEqmEqns=length(GEeqnNames);
    else
        nGeneralEqmEqns=length(GeneralEqmEqns);
    end
    heteroagentoptions.fminalgo5.add=[heteroagentoptions.fminalgo5.howtoupdate{:,3}];
    heteroagentoptions.fminalgo5.factor=[heteroagentoptions.fminalgo5.howtoupdate{:,4}];
    heteroagentoptions.fminalgo5.keepold=ones(size(heteroagentoptions.fminalgo5.factor));
    
    if size(heteroagentoptions.fminalgo5.howtoupdate,1)==nGeneralEqmEqns && nGeneralEqmEqns==length(GEPriceParamNames)
        % do nothing, this is how things should be
    else
        fprintf('ERROR: heteroagentoptions.fminalgo5..howtoupdate does not fit with GeneralEqmEqns (different number of conditions/prices) \n')
    end
    heteroagentoptions.fminalgo5.permute=zeros(size(heteroagentoptions.fminalgo5.howtoupdate,1),1);
    for tt=1:size(heteroagentoptions.fminalgo5.howtoupdate,1) % number of rows is the number of prices (and number of GE conditions)
        for jj=1:length(GEPriceParamNames)
            if strcmp(heteroagentoptions.fminalgo5.howtoupdate{tt,2},GEPriceParamNames{jj})
                heteroagentoptions.fminalgo5.permute(tt)=jj;
            end
        end
    end
    if isfield(heteroagentoptions,'updateaccuracycutoff')==0
        heteroagentoptions.updateaccuracycutoff=0; % No cut-off (only changes in the price larger in magnitude that this will be made (can be set to, e.g., 10^(-6) to help avoid changes at overly high precision))
    end
end



%% If there is entry and exit, then send to relevant command
if isfield(simoptions,'agententryandexit')==1
    if simoptions.agententryandexit>=1
        [p_eqm,p_eqm_index, GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_EntryExit(n_d, n_a, n_z, n_p, d_grid, a_grid, z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions);
        % The EntryExit codes already set p_eqm as a structure.
        varargout={p_eqm,p_eqm_index,GeneralEqmConditions};
        return
    end
end


%%
if heteroagentoptions.maxiter>0 % Can use heteroagentoptions.maxiter=0 to just evaluate the current general eqm eqns

    %% Otherwise, use fminsearch to find the general equilibrium

    if heteroagentoptions.fminalgo~=3 && heteroagentoptions.fminalgo~=8
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_subfn(p, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, GeneralEqmEqnsCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions, simoptions, vfoptions);
    elseif heteroagentoptions.fminalgo==3
        heteroagentoptions.outputGEform=1; % vector
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_subfn(p, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, GeneralEqmEqnsCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions, simoptions, vfoptions);
    elseif heteroagentoptions.fminalgo==8
        heteroagentoptions.outputGEform=1; % vector
        weightsbackup=heteroagentoptions.multiGEweights;
        heteroagentoptions.multiGEweights=sqrt(heteroagentoptions.multiGEweights); % To use a weighting matrix in lsqnonlin(), we work with the square-roots of the weights
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_subfn(p, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, GeneralEqmEqnsCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions, simoptions, vfoptions);
        heteroagentoptions.multiGEweights=weightsbackup; % change it back now that we have set up CalibrateLifeCycleModel_objectivefn()
    end

    
    % Choosing algorithm for the optimization problem
    % https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
    minoptions = optimset('TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns,'MaxFunEvals',heteroagentoptions.maxiter);
    p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless
    if N_p~=0 % Solving on p_grid
        GeneralEqmConditions=zeros(size(heteroagentoptions.p_grid)); %
        for pp_c=1:size(heteroagentoptions.p_grid,1)
            pvec=heteroagentoptions.p_grid(pp_c,:);
            GeneralEqmConditions(pp_c,:)=GeneralEqmConditionsFnOpt(pvec);
        end
        [~,p_eqm_index]=max(sum(GeneralEqmConditions.^2,2));
        p_eqm=heteroagentoptions.p_grid(p_eqm_index,:);
    elseif heteroagentoptions.fminalgo==0 % fzero, is based on root-finding so it needs just the vector of GEcondns, not the sum-of-squares (it is not a minimization routine)
        [p_eqm_vec,GeneralEqmConditions]=fzero(GeneralEqmConditionsFnOpt,GEparamsvec0,minoptions);
    elseif heteroagentoptions.fminalgo==1
        [p_eqm_vec,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFnOpt,GEparamsvec0,minoptions);
    elseif heteroagentoptions.fminalgo==2
        % Use the optimization toolbox so as to take advantage of automatic differentiation
        z=optimvar('z',length(GEparamsvec0));
        optimfun=fcn2optimexpr(GeneralEqmConditionsFnOpt, z);
        prob = optimproblem("Objective",optimfun);
        z0.z=GEparamsvec0;
        [sol,GeneralEqmConditions]=solve(prob,z0);
        p_eqm_vec=sol.z;
        % Note, doesn't really work as automatic differentiation is only for supported functions, and the objective here is not a supported function
    elseif heteroagentoptions.fminalgo==3
        goal=zeros(length(GEparamsvec0),1);
        weight=ones(length(GEparamsvec0),1); % I already implement weights via heteroagentoptions
        [p_eqm_vec,GeneralEqmConditionsVec] = fgoalattain(GeneralEqmConditionsFnOpt,GEparamsvec0,goal,weight);
        GeneralEqmConditions=sum(abs(GeneralEqmConditionsVec));
    elseif heteroagentoptions.fminalgo==4 % CMA-ES algorithm (Covariance-Matrix adaptation - Evolutionary Stategy)
        % https://en.wikipedia.org/wiki/CMA-ES
        % https://cma-es.github.io/
        % Code is cmaes.m from: https://cma-es.github.io/cmaes_sourcecode_page.html#matlab
        if ~isfield(heteroagentoptions,'insigma')
            % insigma: initial coordinate wise standard deviation(s)
            heteroagentoptions.insigma=0.3*abs(GEparamsvec0)+0.1*(GEparamsvec0==0); % Set standard deviation to 30% of the initial parameter value itself (cannot input zero, so add 0.1 to any zeros)
        end
        if ~isfield(heteroagentoptions,'inopts')
            % inopts: options struct, see defopts below
            heteroagentoptions.inopts=[];
        end
        if isfield(heteroagentoptions,'toleranceGEcondns')
            heteroagentoptions.inopts.StopFitness=heteroagentoptions.toleranceGEcondns;
        end
        % varargin (unused): arguments passed to objective function
        if heteroagentoptions.verbose==1
            disp('VFI Toolkit is using the CMA-ES algorithm, consider giving a cite to: Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution Strategy on Multimodal Test Functions' )
        end
    	% This is a minor edit of cmaes, because I want to use 'GeneralEqmConditionsFnOpt' as a function_handle, but the original cmaes code only allows for 'GeneralEqmConditionsFnOpt' as a string
        [p_eqm_vec,GeneralEqmConditions,counteval,stopflag,out,bestever] = cmaes_vfitoolkit(GeneralEqmConditionsFnOpt,GEparamsvec0,heteroagentoptions.insigma,heteroagentoptions.inopts); % ,varargin);
    elseif heteroagentoptions.fminalgo==5
        % Update based on rules in heteroagentoptions.fminalgo5.howtoupdate
        % Get initial prices, p
        p=nan(1,length(GEPriceParamNames));
        for ii=1:length(GEPriceParamNames)
            p(ii)=Parameters.(GEPriceParamNames{ii});
        end
        % Given current prices solve the model to get the general equilibrium conditions as a structure
        itercounter=0;
        p_change=Inf;
        GeneralEqmConditions=Inf;
        while (any(p_change>heteroagentoptions.toleranceGEprices) || GeneralEqmConditions>heteroagentoptions.toleranceGEcondns) && itercounter<heteroagentoptions.maxiter

            p_i=GeneralEqmConditionsFnOpt(p); % using heteroagentoptions.outputGEform=1, so this is a vector (note the transpose)

            GeneralEqmConditionsVec=p_i; % Need later to look at convergence

            % Update prices based on GEstruct following the howtoupdate rules
            p_i=p_i(heteroagentoptions.fminalgo5.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
            I_makescutoff=(abs(p_i)>heteroagentoptions.updateaccuracycutoff);
            p_i=I_makescutoff.*p_i;

            p_new=(p.*heteroagentoptions.fminalgo5.keepold)+heteroagentoptions.fminalgo5.add.*heteroagentoptions.fminalgo5.factor.*p_i-(1-heteroagentoptions.fminalgo5.add).*heteroagentoptions.fminalgo5.factor.*p_i;

            % Calculate GeneralEqmConditions which measures convergence
            if heteroagentoptions.multiGEcriterion==0
                GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
            elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market
                GeneralEqmConditions=sqrt(sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2)));
            end

            % Put new prices into Parameters
            for ii=1:length(GEPriceParamNames)
                Parameters.(GEPriceParamNames{ii})=p_new(ii);
            end

            p_change=abs(p_new-p); % note, this is a vector
            % p_percentchange=max(abs(p_new-p)./abs(p));
            % p_percentchange(p==0)=abs(p_new(p==0)); %-p(p==0)); but this is just zero anyway
            % Update p for next iteration
            p=p_new;
            itercounter=itercounter+1; % increment iteration counter
        end
        p_eqm_vec=p_new; % Need to put it in p_eqm_vec so that it can be used to create the final output
    elseif heteroagentoptions.fminalgo==7 % Matlab fsolve()
        heteroagentoptions.multiGEcriterion=0;
        [p_eqm_vec,GeneralEqmConditions]=fsolve(GeneralEqmConditionsFnOpt,GEparamsvec0,minoptions);
    elseif heteroagentoptions.fminalgo==8 % Matlab lsqnonlin()
        minoptions = optimoptions('lsqnonlin','FiniteDifferenceStepSize',1e-2,'TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns,'MaxFunEvals',heteroagentoptions.maxiter,'MaxIter',heteroagentoptions.maxiter);
        [p_eqm_vec,GeneralEqmConditions]=lsqnonlin(GeneralEqmConditionsFnOpt,GEparamsvec0,[],[],[],[],[],[],[],minoptions);
    end

    % p_eqm_vec contains the (transformed) unconstrained parameters, not the original (constrained) parameter values.
    [p_eqm_vec,~]=ParameterConstraints_TransformParamsToOriginal(p_eqm_vec,0:1:nGEParams,GEPriceParamNames,heteroagentoptions);
    % p_eqm_vec is now the original (constrained) parameter values.

    for ii=1:length(GEPriceParamNames)
        p_eqm.(GEPriceParamNames{ii})=p_eqm_vec(ii);
    end

%%
elseif heteroagentoptions.maxiter==0 % Can use heteroagentoptions.maxiter=0 to just evaluate the current general eqm eqns
    % Just use the prices that are currently in Params
    p_eqm_vec=zeros(length(GEparamsvec0),1);
    p_eqm=nan; % So user cannot misuse
    p_eqm_index=nan; % In case user asks for it
    for ii=1:length(GEPriceParamNames)
        p_eqm_vec(ii)=Parameters.(GEPriceParamNames{ii});
    end
end


%% Add an evaluation of the general eqm eqns so that these can be output as a vectors/structure rather than just the sum of squares
if heteroagentoptions.outputGEstruct==1
    heteroagentoptions.outputGEform=2; % output as struct
elseif heteroagentoptions.outputGEstruct==2
    heteroagentoptions.outputGEform=1; % output as vector
end

if heteroagentoptions.outputGEstruct==1 || heteroagentoptions.outputGEstruct==2
    % Run once more to get the general eqm eqns in a nice form for output
    % Using the original (constrained) parameters, so just ignore the  constraints (otherwise it would assume they are the transformed parameters)
    if isfield(heteroagentoptions,'constrainpositive')
        heteroagentoptions.constrainpositive=zeros(length(p_eqm_vec),1);
    end
    if isfield(heteroagentoptions,'constrain0to1')
        heteroagentoptions.constrain0to1=zeros(length(p_eqm_vec),1);
    end
    if isfield(heteroagentoptions,'constrainAtoB')
        heteroagentoptions.constrainAtoB=zeros(length(p_eqm_vec),1);
    end
    GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_subfn(p, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, GeneralEqmEqnsCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions, simoptions, vfoptions);
    GeneralEqmConditions=GeneralEqmConditionsFnOpt(p_eqm_vec);
end
if heteroagentoptions.outputGEstruct==1
    % structure is much easier to read if it is on cpu, just because matlab doesn't print the value of scalar for gpu but instead tells you it is gpu
    if heteroagentoptions.outputGEstruct==1
        GENames=fieldnames(GeneralEqmEqns);
        for gg=1:length(GENames)
            GeneralEqmConditions.(GENames{gg})=gather(GeneralEqmConditions.(GENames{gg}));
        end
    end

end


%% If using pricehistory, create a clean version for output to user
if heteroagentoptions.pricehistory==1
    load pricehistory.mat GEpricepath GEcondnpath itercount
    delete('pricehistory.mat')
    GEpricepath=GEpricepath(:,1:itercount); % drop the NaN at end
    GEcondnpath=GEcondnpath(:,1:itercount); % drop the NaN at end
    PriceHistory.itercount=itercount;
    for pp=1:length(GEPriceParamNames)
        PriceHistory.(GEPriceParamNames{pp})=GEpricepath(pp,:);
    end
    GENames=fieldnames(GeneralEqmEqns);
    for gg=1:length(GENames)
        PriceHistory.(GENames{gg})=GEcondnpath(gg,:);
    end
end



%%
if heteroagentoptions.pricehistory==0
    if nargout==1
        varargout={p_eqm};
    elseif nargout==2
        varargout={p_eqm,GeneralEqmConditions};
    elseif nargout==3
        varargout={p_eqm,p_eqm_index,GeneralEqmConditions};
    end
elseif heteroagentoptions.pricehistory==1
    if nargout==1
        varargout={p_eqm};
    elseif nargout==2
        varargout={p_eqm,GeneralEqmConditions};
    elseif nargout==3
        varargout={p_eqm,GeneralEqmConditions,PriceHistory};
    elseif nargout==4
        varargout={p_eqm,p_eqm_index,GeneralEqmConditions,PriceHistory};
    end
end


end