function GeneralEqmConditions=HeteroAgentStationaryEqm_FHorz_PType2L_subfn(GEpricesvec, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions)
% Two-level PType GE objective. PTypeStructure is flat over (top,bottom) pairs:
%   PTypeStructure.N_pairs                number of (top,bottom) pairs
%   PTypeStructure.iistr{k}               pair key 'topname__ptypeNNN'
%   PTypeStructure.itop(k), ibot(k)       indexes into Names_i (top) and within-top bottom
%   PTypeStructure.(pairkey)              fully-peeled per-pair record (n_d, n_a, n_z, N_j,
%                                         z_gridvals_J, pi_z_J, d_grid, a_grid, ReturnFn,
%                                         ReturnFnParamNames, DiscountFactorParamNames,
%                                         Parameters, jequaloneDist, AgeWeightParamNames,
%                                         FnsToEvaluate, FnsToEvaluateParamNames,
%                                         FnsAndPTypeIndicator_ii, vfoptions, simoptions, ...)
%   PTypeStructure.ptweights(k,1)         topweight(itop(k))*bottomweight(itop(k),ibot(k))
% Inner loop is the same shape as HeteroAgentStationaryEqm_Case1_FHorz_PType_subfn:
% each pair drives ValueFnIter_Case1_FHorz / StationaryDist_FHorz_Case1 /
% EvalFnOnAgentDist_AggVars_FHorz_Case1.

heteroagentparamsvecindex=0:1:length(GEpricesvec);
[GEpricesvec,~]=ParameterConstraints_TransformParamsToOriginal(GEpricesvec,heteroagentparamsvecindex,GEPriceParamNames,heteroagentoptions);

if heteroagentoptions.verbose==2
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for pp=1:nGEprices
        fprintf('	%s: %8.4f \n',GEPriceParamNames{pp},GEpricesvec(pp))
    end
end

%%
for pp=1:nGEprices
    Parameters.(GEPriceParamNames{pp})=GEpricesvec(pp);
end


%%
AggVars_ConditionalOnPType=zeros(PTypeStructure.numFnsToEvaluate,PTypeStructure.N_pairs,'gpuArray');

for ii=1:PTypeStructure.N_pairs

    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEpricesvec(pp);
    end

    if heteroagentoptions.gridsinGE(ii)==1
        [PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).vfoptions]=ExogShockSetup_FHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_gridvals_J,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
        PTypeStructure.(iistr).simoptions.e_gridvals_J=PTypeStructure.(iistr).vfoptions.e_gridvals_J;
        PTypeStructure.(iistr).simoptions.pi_e_J=PTypeStructure.(iistr).vfoptions.pi_e_J;
    end

    if heteroagentoptions.gridsinGE_semiexo(ii)==1
        PTypeStructure.(iistr).vfoptions=SemiExogShockSetup_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
        PTypeStructure.(iistr).simoptions.semiz_gridvals_J=PTypeStructure.(iistr).vfoptions.semiz_gridvals_J;
        PTypeStructure.(iistr).simoptions.pi_semiz_J=PTypeStructure.(iistr).vfoptions.pi_semiz_J;
    end

    [~, Policy_ii]=ValueFnIter_Case1_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
    StationaryDist_ii=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).simoptions);
    AggVars_ii=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).simoptions);

    AggVars_ConditionalOnPType(PTypeStructure.(iistr).FnsAndPTypeIndicator_ii,ii)=AggVars_ii;
end
AggVars=gather(sum(AggVars_ConditionalOnPType.*PTypeStructure.ptweights',2));


%% Put GE parameters and AggVars in structure, so they can be used for intermediateEqns and GeneralEqmEqns
for aa=1:length(AggVarNames)
    Parameters.(AggVarNames{aa})=AggVars(aa);
end

%% Intermediate Eqns
if heteroagentoptions.useintermediateEqns==1
    intEqnnames=fieldnames(heteroagentoptions.intermediateEqns);
    intermediateEqnsVec=zeros(1,length(intEqnnames));
    for gg=1:length(intEqnnames)
        intermediateEqnsVec(gg)=real(GeneralEqmConditions_Case1_v3g(heteroagentoptions.intermediateEqnsCell{gg}, heteroagentoptions.intermediateEqnParamNames(gg).Names, Parameters));
        Parameters.(intEqnnames{gg})=intermediateEqnsVec(gg);
    end
end

%% Evaluate General Eqm Eqns
GeneralEqmConditionsVec=zeros(1,length(GEeqnNames));
for gg=1:length(GEeqnNames)
    GeneralEqmConditionsVec(gg)=real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
end


%% Output shape
if heteroagentoptions.outputGEform==0 % scalar
    if heteroagentoptions.multiGEcriterion==0
        GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
    elseif heteroagentoptions.multiGEcriterion==1
        GeneralEqmConditions=sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2));
    end
    if heteroagentoptions.outputgather==1
        GeneralEqmConditions=gather(GeneralEqmConditions);
    end
elseif heteroagentoptions.outputGEform==1 % vector
    GeneralEqmConditions=GeneralEqmConditionsVec;
    if heteroagentoptions.outputgather==1
        GeneralEqmConditions=gather(GeneralEqmConditions);
    end
elseif heteroagentoptions.outputGEform==2 % structure
    clear GeneralEqmConditions
    for gg=1:length(GEeqnNames)
        GeneralEqmConditions.(GEeqnNames{gg})=GeneralEqmConditionsVec(gg);
    end
end

%% Feedback on progress
if heteroagentoptions.verbose==1
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for pp=1:nGEprices
        fprintf(heteroagentoptions.verboseaccuracy1,GEPriceParamNames{pp},GEpricesvec(pp))
    end
end
if heteroagentoptions.verbose>=1
    fprintf('Current aggregate variables: \n')
    for aa=1:length(AggVarNames)
        fprintf(heteroagentoptions.verboseaccuracy1,AggVarNames{aa},AggVars(aa))
    end
    if heteroagentoptions.useintermediateEqns==1
        fprintf('Current intermediateEqn variables: \n')
        for aa=1:length(intEqnnames)
            fprintf(heteroagentoptions.verboseaccuracy1,intEqnnames{aa},intermediateEqnsVec(aa))
        end
    end
    fprintf('Current GeneralEqmEqns: \n')
    for gg=1:length(GEeqnNames)
        fprintf(heteroagentoptions.verboseaccuracy2,GEeqnNames{gg},GeneralEqmConditionsVec(gg))
    end
end


if heteroagentoptions.pricehistory==1
    load pricehistory.mat GEpricepath GEcondnpath itercount
    itercount=itercount+1;
    GEpricepath(:,itercount)=GEpricesvec;
    GEcondnpath(:,itercount)=GeneralEqmConditionsVec;
    save pricehistory.mat GEpricepath GEcondnpath itercount
end


end
