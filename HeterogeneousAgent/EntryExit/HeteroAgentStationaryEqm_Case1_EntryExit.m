function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_EntryExit(n_d, n_a, n_z, n_p, d_grid, a_grid, z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions)
% If n_p=0 then will use fminsearch to find the general equilibrium (find
% price vector that corresponds to GeneralEqmCondition=0). By setting n_p to
% nonzero it is assumed you want to use a grid on prices, which must then
% be passed in using heteroagentoptions.p_grid

N_a=prod(n_a);
N_s=prod(n_z);
N_p=prod(n_p);

p_eqm=struct(); p_eqm_index=nan; GeneralEqmConditions=nan;

%% Check which options have been used, set all others to defaults 
if exist('vfoptions','var')==0
    %If vfoptions is not given, just use all the defaults
    vfoptions.parallel=2;
    %Note that the defaults will be set when we call 'ValueFnIter...'
    %commands and the like, so no need to set them here except for a few.
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=2;
    end
end

if exist('simoptions','var')==0
    simoptions.fakeoption=0; % create a 'placeholder' simoptions that can be passed to subcodes
    %Note that the defaults will be set when we call 'StationaryDist...'
    %commands and the like, so no need to set them here except for a few.
    simoptions.agententryandexit=1;
    simoptions.endogenousexit=0;
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=2;
    end
    if isfield(simoptions,'agententryandexit')==0
        simoptions.agententryandexit=1;
    end
    if isfield(simoptions,'endogenousexit')==0
        simoptions.endogenousexit=0;
    end
end

if exist('heteroagentoptions','var')==0
    heteroagentoptions.multiGEcriterion=1;
    heteroagentoptions.fminalgo=1;
    heteroagentoptions.verbose=0;
    heteroagentoptions.maxiter=1000;
    heteroagentoptions.showfigures=1;
else
    if isfield(heteroagentoptions,'multiGEcriterion')==0
        heteroagentoptions.multiGEcriterion=1;
    end
    if N_p~=0
        if isfield(heteroagentoptions,'pgrid')==0
            disp('VFI Toolkit ERROR: you have set n_p to a non-zero value, but not declared heteroagentoptions.pgrid')
        end
    end
    if isfield(heteroagentoptions,'verbose')==0
        heteroagentoptions.verbose=0;
    end
    if isfield(heteroagentoptions,'fminalgo')==0
        heteroagentoptions.fminalgo=1; % use fminsearch
    end
    if isfield(heteroagentoptions,'maxiter')==0
        heteroagentoptions.maxiter=1000; % use fminsearch
    end
    if isfield(heteroagentoptions,'showfigures')==0
        heteroagentoptions.showfigures=1;
    end
end

%%
% Check if gthere is an initial guess for V0
if isfield(vfoptions,'V0')
    vfoptions.V0=reshape(vfoptions.V0,[N_a,N_s]);
else
    if vfoptions.parallel==2
        vfoptions.V0=zeros([N_a,N_s], 'gpuArray');
    else
        vfoptions.V0=zeros([N_a,N_s]);
    end
end

%% Sometimes, verbose needs a figure that it updates each step
if heteroagentoptions.showfigures==1
    heteroagentoptions.verbosefighandle=figure;
end

%% If solving on p_grid
if N_p~=0
    error('Using p_grid with Entry/Exit is not currently implemented. Please email robertdkirkby@gmail.com if you have a specific want/need for it and I can easily implement it. \n')
end

%% Otherwise, use fminsearch to find the general equilibrium

GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_EntryExit_subfn(p, n_d, n_a, n_z, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions);

p0=nan(length(GEPriceParamNames),1);
for ii=1:length(GEPriceParamNames)
    p0(ii)=Parameters.(GEPriceParamNames{ii});
end

% Choosing algorithm for the optimization problem
% https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
minoptions = optimset('TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns);
if heteroagentoptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
    heteroagentoptions.multimarketcriterion=0;
    [p_eqm_vec,GeneralEqmConditions]=fzero(GeneralEqmConditionsFnOpt,p0,minoptions);    
elseif heteroagentoptions.fminalgo==1
    [p_eqm_vec,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFnOpt,p0,minoptions);
elseif heteroagentoptions.fminalgo==2
    % Use the optimization toolbox so as to take advantage of automatic differentiation
    z=optimvar('z',length(p0));
    optimfun=fcn2optimexpr(GeneralEqmConditionsFnOpt, z);
    prob = optimproblem("Objective",optimfun);
    z0.z=p0;
    [sol,GeneralEqmConditions]=solve(prob,z0);
    p_eqm_vec=sol.z;
    % Note, doesn't really work as automattic differentiation is only for
    % supported functions, and the objective here is not a supported function
elseif heteroagentoptions.fminalgo==3
    goal=zeros(length(p0),1);
    weight=ones(length(p0),1); % I already implement weights via heteroagentoptions
    [p_eqm_vec,GeneralEqmConditionsVec] = fgoalattain(GeneralEqmConditionsFnOpt,p0,goal,weight);
    GeneralEqmConditions=sum(abs(GeneralEqmConditionsVec));
elseif heteroagentoptions.fminalgo==4 % CMA-ES algorithm (Covariance-Matrix adaptation - Evolutionary Stategy)
    % https://en.wikipedia.org/wiki/CMA-ES
    % https://cma-es.github.io/
    % Code is cmaes.m from: https://cma-es.github.io/cmaes_sourcecode_page.html#matlab
    if ~isfield(heteroagentoptions,'insigma')
        % insigma: initial coordinate wise standard deviation(s)
        heteroagentoptions.insigma=0.3*abs(p0)+0.1*(p0==0); % Set standard deviation to 30% of the initial parameter value itself (cannot input zero, so add 0.1 to any zeros)
    end
    if ~isfield(heteroagentoptions,'inopts')
        % inopts: options struct, see defopts below
        heteroagentoptions.inopts=[];
    end
    % varargin (unused): arguments passed to objective function 
    if heteroagentoptions.verbose==1
        disp('VFI Toolkit is using the CMA-ES algorithm, consider giving a cite to: Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution Strategy on Multimodal Test Functions' )
    end
	% This is a minor edit of cmaes, because I want to use 'GeneralEqmConditionsFnOpt' as a function_handle, but the original cmaes code only allows for 'GeneralEqmConditionsFnOpt' as a string
    [p_eqm_vec,GeneralEqmConditions,counteval,stopflag,out,bestever] = cmaes_vfitoolkit(GeneralEqmConditionsFnOpt,p0,heteroagentoptions.insigma,heteroagentoptions.inopts); % ,varargin);
end

for ii=1:length(GEPriceParamNames)
    p_eqm.(GEPriceParamNames{ii})=p_eqm_vec(ii);
end

% Check for use of conditional entry condition.
specialgeneqmcondnsused=0;
condlentrycondnexists=0;
if isfield(heteroagentoptions,'specialgeneqmcondn')
    for ii=1:length(GeneralEqmEqns)
        if isnumeric(heteroagentoptions.specialgeneqmcondn{ii}) % numeric means equal to zero and is a standard GEqm
            % nothing
        elseif strcmp(heteroagentoptions.specialgeneqmcondn{ii},'entry')
            specialgeneqmcondnsused=1;
        elseif strcmp(heteroagentoptions.specialgeneqmcondn{ii},'condlentry')
            specialgeneqmcondnsused=1;
            condlentrycondnexists=1;
        end
    end
end
if specialgeneqmcondnsused==1
    if condlentrycondnexists==1
        % Need to compute this (is calculated inside subfn, but then lost (not kept)
        % To keep things clean and tidy I am using a function call to do
        % this, it is essentially a copy-paste of the _subfn command.
        p_eqm.(EntryExitParamNames.CondlEntryDecisions{1})=HeteroAgentStationaryEqm_Case1_EntryExit_subfn_condlentry(p_eqm_vec, n_d, n_a, n_z, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions);
    end
end
% p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless.
% Is already initalised as p_eqm_index=nan; so just leave it as is.




end