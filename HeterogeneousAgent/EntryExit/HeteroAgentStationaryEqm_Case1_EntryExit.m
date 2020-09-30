function [p_eqm,p_eqm_index,GeneralEqmCondition]=HeteroAgentStationaryEqm_Case1_EntryExit(n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions)
% If n_p=0 then will use fminsearch to find the general equilibrium (find
% price vector that corresponds to GeneralEqmCondition=0). By setting n_p to
% nonzero it is assumed you want to use a grid on prices, which must then
% be passed in using heteroagentoptions.p_grid


N_d=prod(n_d);
N_a=prod(n_a);
N_s=prod(n_s);
N_p=prod(n_p);

l_p=length(n_p);

p_eqm=struct(); p_eqm_index=nan; GeneralEqmCondition=nan;

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
if heteroagentoptions.verbose==1
    heteroagentoptions.verbosefighandle=figure;
end

%% If solving on p_grid
if N_p~=0
    fprintf('ERROR: Using p_grid with Entry/Exit is not currently implemented. Please email robertdkirkby@gmail.com if you have a specific want/need for it and I can easily implement it. \n')

%     fprintf('WARNING: Using p_grid with Entry/Exit is not likely to converge to correct solution (as it does not enforce general eqm in the entry conditions). \n')
%     fprintf('         It remains possible to use p_grid with Entry/Exit solely because it can be useful for exploratory purposes. It should not be used for solving models. \n')
%     [p_eqm,p_eqm_index,GeneralEqmCondition]=HeteroAgentStationaryEqm_Case1_EntryExit_pgrid(n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions);
%     fprintf('WARNING: Using p_grid with Entry/Exit is not likely to converge to correct solution (as it does not enforce general eqm in the entry conditions). \n')
%     fprintf('         It remains possible to use p_grid with Entry/Exit solely because it can be useful for exploratory purposes. It should not be used for solving models. \n')
    return
end

%% Otherwise, use fminsearch to find the general equilibrium

% I SHOULD IMPLEMENT A BETTER V0Kron HERE
GeneralEqmConditionsFn=@(p) HeteroAgentStationaryEqm_Case1_EntryExit_subfn(p, n_d, n_a, n_s, pi_s, d_grid, a_grid, s_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions);

p0=nan(length(GEPriceParamNames),1);
for ii=1:length(GEPriceParamNames)
    p0(ii)=Parameters.(GEPriceParamNames{ii});
end

if heteroagentoptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
    heteroagentoptions.multiGEcriterion=0;
    [p_eqm_vec,GeneralEqmCondition]=fzero(GeneralEqmConditionsFn,p0);    
elseif heteroagentoptions.fminalgo==1
    [p_eqm_vec,GeneralEqmCondition]=fminsearch(GeneralEqmConditionsFn,p0);
else
    [p_eqm_vec,GeneralEqmCondition]=fminsearch(GeneralEqmConditionsFn,p0);
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
        p_eqm.(EntryExitParamNames.CondlEntryDecisions{1})=HeteroAgentStationaryEqm_Case1_EntryExit_subfn_condlentry(p_eqm_vec, n_d, n_a, n_s, pi_s, d_grid, a_grid, s_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions);
    end
end
% p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless.
% Is already initalised as p_eqm_index=nan; so just leave it as is.




end