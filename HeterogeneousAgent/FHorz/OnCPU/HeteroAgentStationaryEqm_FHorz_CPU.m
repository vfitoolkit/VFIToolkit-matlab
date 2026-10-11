function [p_eqm, GeneralEqmConditions]=HeteroAgentStationaryEqm_FHorz_CPU(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)
% Outputs: [p_eqm, GeneralEqmConditions]

nGEprices=length(GEPriceParamNames);
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);
AggVarNames=fieldnames(FnsToEvaluate);

%%
vfoptions.parallel=1;
simoptions.parallel=1;
simoptions.outputasstructure=0;

heteroagentoptions.useintermediateEqns=0;

%%
jequaloneDist=gather(jequaloneDist);

heteroagentoptions.gridsinGE=0;

%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
if isstruct(FnsToEvaluate)
    l_d=length(n_d);
    if n_d(1)==0
        l_d=0;
    end
    l_a=length(n_a);
    l_aprime=l_a;
    l_z=length(n_z);
    if prod(n_z)==0
        l_z=0;
    end

    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_aprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_aprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
    end
    % Now have FnsToEvaluate as structure, FnsToEvaluateCell as cell
else
    % Do nothing
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
for pp=1:nGEParams
    GEparamsvec0(pp)=Parameters.(GEPriceParamNames{pp});
end

% If the parameter is constrained in some way then we need to transform it
[GEparamsvec0,heteroagentoptions]=ParameterConstraints_TransformParamsToUnconstrained(GEparamsvec0,0:1:nGEParams,GEPriceParamNames,heteroagentoptions,1);
% Also converts the constraints info in estimoptions to be a vector rather than by name.

%% Enough setup, Time to do the actual finding the HeteroAgentStationaryEqm:

if heteroagentoptions.maxiter>0 % Can use heteroagentoptions.maxiter=0 to just evaluate the current general eqm eqns

    %% Otherwise, use fminsearch to find the general equilibrium
    GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_FHorz_CPU_subfn(p, jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, GeneralEqmEqnsCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions, simoptions, vfoptions);

    % CPU: hardcodes fminalgo=1
    minoptions = optimset('TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns,'MaxIter',heteroagentoptions.maxiter,'MaxFunEvals',heteroagentoptions.maxiter); % same maxiter convention in every stationary GE command, MaxIter=MaxFunEvals=maxiter (until 2026-10-09 fminsearch was given no options here, so it ignored maxiter and both tolerances)
    [p_eqm_vec,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFnOpt,GEparamsvec0,minoptions);

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
    GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_FHorz_CPU_subfn(p, jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, GeneralEqmEqnsCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions, simoptions, vfoptions);
    GeneralEqmConditions=GeneralEqmConditionsFnOpt(p_eqm_vec);
end
if heteroagentoptions.outputGEstruct==1
    % put GeneralEqmConditions structure on cpu for purely cosmetic reasons
    GEeqnNames=fieldnames(GeneralEqmEqns);
    for gg=1:length(GEeqnNames)
        GeneralEqmConditions.(GEeqnNames{gg})=gather(GeneralEqmConditions.(GEeqnNames{gg}));
    end
end

%% Check that the general eqm conditions are actually satisfied
% None of the fminalgo routines verify the point they stopped at. The tolerances are passed to them
% as TolX/TolFun, but TolFun is a test on the CHANGE in the objective between iterations, not on its
% value, so an optimizer that stalls returns a p_eqm that is not a general eqm and says nothing.
% Only outputGEstruct 1 and 2 hold the conditions evaluated at the returned prices (outputGEstruct=0
% is the n_p grid, where GeneralEqmConditions covers the whole grid). maxiter=0 is the 'just evaluate
% the current general eqm eqns' mode, which is not a solve and so is not checked.
if (heteroagentoptions.outputGEstruct==1 || heteroagentoptions.outputGEstruct==2) && heteroagentoptions.maxiter>0
    if isstruct(GeneralEqmConditions)
        % Looped rather than cell2mat(struct2cell()), because a general eqm condition that depends on
        % permanent type holds one value per ptype, so the fields are not all the same size
        GEcondnsnames=fieldnames(GeneralEqmConditions);
        GEcondnsmax=0;
        for gg=1:length(GEcondnsnames)
            GEcondnsmax=max(GEcondnsmax,max(abs(GeneralEqmConditions.(GEcondnsnames{gg})(:))));
        end
    else
        GEcondnsmax=max(abs(GeneralEqmConditions(:)));
    end
    if GEcondnsmax>heteroagentoptions.toleranceGEcondns
        warning('HeteroAgentStationaryEqm_FHorz_CPU: the general eqm conditions are not all within heteroagentoptions.toleranceGEcondns (largest is %g, tolerance is %g)',GEcondnsmax,heteroagentoptions.toleranceGEcondns)
    end
end




end