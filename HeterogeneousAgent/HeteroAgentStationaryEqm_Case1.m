function [p_eqm,p_eqm_index,GeneralEqmCondition]=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions, EntryExitParamNames)
% If n_p=0 then will use fminsearch to find the general equilibrium (find
% price vector that corresponds to GeneralEqmCondition=0). By setting n_p to
% nonzero it is assumed you want to use a grid on prices, which must then
% be passed in using heteroagentoptions.p_grid
%
% Input heteroagentoptions and those that appear after it are all optional.
%

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_p=prod(n_p);
if isempty(n_p)
    N_p=0;
end

l_p=length(GEPriceParamNames);

p_eqm_vec=nan; p_eqm=struct(); p_eqm_index=nan; GeneralEqmCondition=nan;

%% Check which options have been used, set all others to defaults 
if exist('vfoptions','var')==0
    %If vfoptions is not given, just use all the defaults
    vfoptions.parallel=1+(gpuDeviceCount>0);
    %Note that the defaults will be set when we call 'ValueFnIter...'
    %commands and the like, so no need to set them here except for a few.
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
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
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
end

if exist('heteroagentoptions','var')==0
    heteroagentoptions.multiGEcriterion=1;
    heteroagentoptions.multiGEweights=ones(1,length(GeneralEqmEqns));
    heteroagentoptions.fminalgo=1;
    heteroagentoptions.verbose=0;
    heteroagentoptions.maxiter=1000;
    heteroagentoptions.oldGE=1;
else
    if isfield(heteroagentoptions,'multiGEcriterion')==0
        heteroagentoptions.multiGEcriterion=1;
    end
    if isfield(heteroagentoptions,'multiGEweights')==0
        heteroagentoptions.multiGEweights=ones(1,length(GeneralEqmEqns));
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
    if isfield(heteroagentoptions,'oldGE')==0
        heteroagentoptions.oldGE=1; % use fminsearch
    end
end

%%
% Check if gthere is an initial guess for V0
if isfield(vfoptions,'V0')
    vfoptions.V0=reshape(vfoptions.V0,[N_a,N_z]);
else
    if vfoptions.parallel==2
        vfoptions.V0=zeros([N_a,N_z], 'gpuArray');
    else
        vfoptions.V0=zeros([N_a,N_z]);
    end
end

% if heteroagentoptions.oldGE==1
% %     GeneralEqmEqnInputNames=GeneralEqmEqnParamNames;
% elseif heteroagentoptions.oldGE==0
%     clear GeneralEqmEqnParamNames
%     for ii=1:length(GeneralEqmEqns)
%         GeneralEqmEqnParamNames(ii).Names=getAnonymousFnInputNames(GeneralEqmEqns{ii});
%     end
% end

%% If there is entry and exit, then send to relevant command
if isfield(simoptions,'agententryandexit')==1
    if simoptions.agententryandexit>=1
        [p_eqm,p_eqm_index, GeneralEqmCondition]=HeteroAgentStationaryEqm_Case1_EntryExit(n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions);
        % The EntryExit codes already set p_eqm as a structure.
        return
    end
end

%% If solving on p_grid
if N_p~=0
    [p_eqm_vec,p_eqm_index,GeneralEqmCondition]=HeteroAgentStationaryEqm_Case1_pgrid(n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions);
    for ii=1:length(GEPriceParamNames)
        p_eqm.(GEPriceParamNames{ii})=p_eqm_vec(ii);
    end
    return
end

%% Otherwise, use fminsearch to find the general equilibrium

if heteroagentoptions.fminalgo<3
    GeneralEqmConditionsFn=@(p) HeteroAgentStationaryEqm_Case1_subfn(p, n_d, n_a, n_z, l_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions)
elseif heteroagentoptions.fminalgo==3
    % Multi-objective verion
    GeneralEqmConditionsFn=@(p) HeteroAgentStationaryEqm_Case1_subfnMO(p, n_d, n_a, n_z, l_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions)
end
    

p0=zeros(length(GEPriceParamNames),1);
for ii=1:length(GEPriceParamNames)
    p0(ii)=Parameters.(GEPriceParamNames{ii});
end

if heteroagentoptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
    heteroagentoptions.multiGEcriterion=0;
    [p_eqm_vec,GeneralEqmCondition]=fzero(GeneralEqmConditionsFn,p0);    
elseif heteroagentoptions.fminalgo==1
    [p_eqm_vec,GeneralEqmCondition]=fminsearch(GeneralEqmConditionsFn,p0);
elseif heteroagentoptions.fminalgo==2
    % Use the optimization toolbox so as to take advantage of automatic differentiation
    z=optimvar('z',length(p0));
    optimfun=fcn2optimexpr(GeneralEqmConditionsFn, z);
    prob = optimproblem("Objective",optimfun);
    z0.z=p0;
    [sol,GeneralEqmCondition]=solve(prob,z0);
    p_eqm_vec=sol.z;
    % Note, doesn't really work as automattic differentiation is only for
    % supported functions, and the objective here is not a supported function
elseif heteroagentoptions.fminalgo==3
    goal=zeros(length(p0),1);
    weight=ones(length(p0),1); % I already implement weights via heteroagentoptions
    [p_eqm_vec,GeneralEqmConditionsVec] = fgoalattain(GeneralEqmConditionsFn,p0,goal,weight);
    GeneralEqmCondition=sum(abs(GeneralEqmConditionsVec));
end

p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless

for ii=1:length(GEPriceParamNames)
    p_eqm.(GEPriceParamNames{ii})=p_eqm_vec(ii);
end

end