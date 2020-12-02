function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeights, n_d, n_a, n_z, N_j, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, SSvaluesFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)
% If n_p=0 then will use fminsearch to find the general equilibrium (find
% price vector that corresponds to GeneralEqmConditions=0). By setting n_p to
% nonzero it is assumed you want to use a grid on prices, which must then
% be passed in using heteroagentoptions.p_grid


% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);
N_p=prod(n_p);

l_p=length(GEPriceParamNames); % Otherwise get problem when not using p_grid
%l_p=length(n_p);

p_eqm_vec=nan; p_eqm_index=nan; GeneralEqmConditions=nan;

%% Check which options have been used, set all others to defaults 
if exist('vfoptions','var')==0
    vfoptions.parallel=1+(gpuDeviceCount>0);
    %If vfoptions is not given, just use all the defaults
    %Note that the defaults will be set when we call 'ValueFnIter...'
    %commands and the like, so no need to set them here except for a few.
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
end

if exist('simoptions','var')==0
    simoptions=struct(); % create a 'placeholder' simoptions that can be passed to subcodes
end

if exist('heteroagentoptions','var')==0
    heteroagentoptions.multiGEcriterion=1;
    heteroagentoptions.multiGEweights=ones(1,length(GeneralEqmEqns));
    heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm eqns
    heteroagentoptions.fminalgo=1;
    heteroagentoptions.verbose=0;
    heteroagentoptions.maxiter=1000;
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
    if isfield(heteroagentoptions,'toleranceGEprices')==0
        heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    end
    if isfield(heteroagentoptions,'toleranceGEcondns')==0
        heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm prices
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

if N_p~=0
    [p_eqm_vec,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_FHorz_pgrid(jequaloneDist,AgeWeights, n_d, n_a, n_z, N_j, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, SSvaluesFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions);
    return
end

%% Otherwise, use fminsearch to find the general equilibrium

GeneralEqmConditionsFn=@(GEprices) HeteroAgentStationaryEqm_Case1_FHorz_subfn(GEprices, jequaloneDist,AgeWeights, n_d, n_a, n_z, N_j, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, SSvaluesFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions)

GEprices0=nan(length(GEPriceParamNames),1);
for ii=1:length(GEPriceParamNames)
    GEprices0(ii)=Parameters.(GEPriceParamNames{ii});
end

fprintf('GEprices0 is: \n')
GEprices0

minoptions = optimset('TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns);
if heteroagentoptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
    heteroagentoptions.multiGEcriterion=0;
    [p_eqm_vec,GeneralEqmConditions]=fzero(GeneralEqmConditionsFn,GEprices0,minoptions);    
elseif heteroagentoptions.fminalgo==1
    [p_eqm_vec,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFn,GEprices0,minoptions);
else
    [p_eqm_vec,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFn,GEprices0,minoptions);
end

p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless

for ii=1:length(GEPriceParamNames)
    p_eqm.(GEPriceParamNames{ii})=p_eqm_vec(ii);
end

end