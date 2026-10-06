function [GEcondnPath,AggVarsPooledPath,AggVarsFullPath]=TransitionPath_FHorz_PType_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, PricePathSizeVec_ii, ParamPath, ParamPathNames, ParamPathSizeVec, T, FnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, transpathoptions, PTypeStructure)
% One iteration of the transition path with permanent types, at a given price path: each ptype is
% solved (value fn backwards, agent dist forwards, aggregate variables) by TransitionPath_FHorz_singlepathiter,
% exactly as a model without permanent types, then the aggregate variables are pooled across ptypes
% with the ptweights and the general eqm conditions are evaluated on them, period by period.
% This is a pure function of PricePathOld given the setup, which is what lets Anderson acceleration
% call it repeatedly.
%
% Output sizes:
%   GEcondnPath       is (T-1)-by-nGeneralEqmEqns_acrossptypes (a general eqm condition that depends on
%                     ptype, transpathoptions.GEptype, has N_i consecutive columns, one per ptype)
%   AggVarsPooledPath is (T-1)-by-numFnsToEvaluate
%   AggVarsFullPath   is (T-1)-by-numFnsToEvaluate-by-N_i, the aggregate variables conditional on ptype
%                     (zero where a FnsToEvaluate is not relevant to that ptype)
%
% PricePathOld contains every price, with N_i blocks of columns for a price that depends on ptype (each
% 1 column, or N_j for an age-dependent price, PTypeStructure.PricePathWidth_ii). Each ptype gets just its
% own block (RelevantPricePath), so PricePathSizeVec_ii describes what it sees. Each ptype has its own
% ParamPath (PTypeStructure.(iistr).ParamPath, ParamPathNames, ParamPathSizeVec). ParamPath,
% ParamPathNames and ParamPathSizeVec here are only the parameters common to all ptypes, which is what the
% general eqm conditions that do not depend on ptype see.

N_i=PTypeStructure.N_i;
Names_i=PTypeStructure.Names_i;

%% Solve each ptype, and keep its aggregate variables
AggVarsFullPath=zeros(T-1,PTypeStructure.numFnsToEvaluate,N_i,'gpuArray'); % Does not include period T
for ii=1:N_i
    iistr=Names_i{ii};

    % Some parts of PricePath may depend on ptype: get just the columns of the current ptype
    PricePathOld_ii=PricePathOld(:,PTypeStructure.(iistr).RelevantPricePath);

    % No general eqm conditions (empty GeneralEqmEqnsCell), and no _tminus1 of the aggregate variables:
    % both need the aggregates of every ptype, so are done below.
    [~,AggVarsPath_ii]=TransitionPath_FHorz_singlepathiter(PricePathOld_ii, PricePathNames, PricePathSizeVec_ii, PTypeStructure.(iistr).ParamPath, PTypeStructure.(iistr).ParamPathNames, PTypeStructure.(iistr).ParamPathSizeVec, T, PTypeStructure.(iistr).V_final, PTypeStructure.(iistr).AgentDist_initial, PTypeStructure.(iistr).jequalOneDist, ...
        PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).n_e,PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).N_d,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).N_semiz,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e, ...
        PTypeStructure.(iistr).l_d,PTypeStructure.(iistr).l_aprime,PTypeStructure.(iistr).l_a,PTypeStructure.(iistr).l_semiz,PTypeStructure.(iistr).l_z,PTypeStructure.(iistr).l_e, ...
        PTypeStructure.(iistr).d_gridvals, PTypeStructure.(iistr).aprime_gridvals,PTypeStructure.(iistr).a_gridvals,PTypeStructure.(iistr).a_grid,PTypeStructure.(iistr).semiz_gridvals_J,PTypeStructure.(iistr).z_gridvals_J,PTypeStructure.(iistr).e_gridvals_J,PTypeStructure.(iistr).semizze_gridvals_J_fastOLG, ...
        PTypeStructure.(iistr).pi_semiz_J, PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).pi_e_J,PTypeStructure.(iistr).pi_e_J_vf,PTypeStructure.(iistr).pi_semiz_J_sim,PTypeStructure.(iistr).pi_z_J_sim,PTypeStructure.(iistr).pi_e_J_sim, ...
        PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).FnsToEvaluateCell, PTypeStructure.(iistr).AggVarNames, PTypeStructure.(iistr).FnsToEvaluateParamNames, {}, struct('Names',{}), PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).AgeWeights_T, PTypeStructure.(iistr).ReturnFnParamNames, ...
        use_tminus1price, use_tminus1params, use_tplus1price, 0, 0, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, {}, [], [], ...
        PTypeStructure.(iistr).vfoptions, PTypeStructure.(iistr).simoptions, PTypeStructure.(iistr).transpathoptions, 1, PTypeStructure.(iistr).PolicyIndexesPath, PTypeStructure.(iistr).N_probs, PTypeStructure.(iistr).II1, PTypeStructure.(iistr).II2, PTypeStructure.(iistr).exceptlastj, PTypeStructure.(iistr).exceptfirstj, PTypeStructure.(iistr).justfirstj);

    % AggVarsPath_ii holds only the FnsToEvaluate relevant to this ptype, in this ptype's order
    for kk=1:PTypeStructure.numFnsToEvaluate
        if PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)>0
            AggVarsFullPath(:,kk,ii)=AggVarsPath_ii(:,PTypeStructure.(iistr).WhichFnsForCurrentPType(kk));
        end
    end
end

%% Pool the aggregate variables across ptypes
% Note: Cannot yet do transition paths in which the mass of each agent type changes.
% A FnsToEvaluate that is only relevant to some ptypes is zero for the others, so this is the
% ptweight-weighted sum over the ptypes it is relevant to (as in EvalFnOnTransPath_AggVars_Case1_FHorz_PType)
AggVarsPooledPath=sum(AggVarsFullPath.*shiftdim(gpuArray(PTypeStructure.ptweights(:)),-2),3);


%% Evaluate the general eqm conditions, period by period
GEcondnPath=zeros(T-1,sum(transpathoptions.GEptype==0)+N_i*sum(transpathoptions.GEptype==1),'gpuArray');

% Parameters holds the values common to all ptypes, Parameters_ii.(iistr) those of ptype iistr (for the
% general eqm conditions that are evaluated conditional on ptype, transpathoptions.GEptype)
Parameters=PTypeStructure.ParametersRaw;
Parameters_ii=struct();
for ii=1:N_i
    iistr=Names_i{ii};
    Parameters_ii.(iistr)=PTypeStructure.(iistr).Parameters;
end

% Initialise _tminus1 entries in Parameters (and Parameters_ii) from initialvalues (used at tt=1)
% An initial value can depend on ptype, as a structure or as a vector of length N_i
if use_tminus1price==1
    for pp=1:length(tminus1priceNames)
        temp=transpathoptions.initialvalues.(tminus1priceNames{pp});
        Parameters.([tminus1priceNames{pp},'_tminus1'])=temp;
        for ii=1:N_i
            iistr=Names_i{ii};
            if isstruct(temp)
                Parameters_ii.(iistr).([tminus1priceNames{pp},'_tminus1'])=temp.(iistr);
            elseif length(temp)==N_i
                Parameters_ii.(iistr).([tminus1priceNames{pp},'_tminus1'])=temp(ii);
            else
                Parameters_ii.(iistr).([tminus1priceNames{pp},'_tminus1'])=temp;
            end
        end
    end
end
if use_tminus1params==1
    for pp=1:length(tminus1paramNames)
        temp=transpathoptions.initialvalues.(tminus1paramNames{pp});
        Parameters.([tminus1paramNames{pp},'_tminus1'])=temp;
        for ii=1:N_i
            iistr=Names_i{ii};
            if isstruct(temp)
                Parameters_ii.(iistr).([tminus1paramNames{pp},'_tminus1'])=temp.(iistr);
            elseif length(temp)==N_i
                Parameters_ii.(iistr).([tminus1paramNames{pp},'_tminus1'])=temp(ii);
            else
                Parameters_ii.(iistr).([tminus1paramNames{pp},'_tminus1'])=temp;
            end
        end
    end
end
if use_tminus1AggVars==1
    for pp=1:length(tminus1AggVarsNames)
        temp=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
        if isstruct(temp) % the initial value of an aggregate variable conditional on ptype
            for ii=1:N_i
                iistr=Names_i{ii};
                Parameters_ii.(iistr).([tminus1AggVarsNames{pp},'_tminus1'])=temp.(iistr);
            end
        else
            Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=temp;
            for ii=1:N_i
                iistr=Names_i{ii};
                if length(temp)==N_i && N_i>1
                    Parameters_ii.(iistr).([tminus1AggVarsNames{pp},'_tminus1'])=temp(ii);
                else
                    Parameters_ii.(iistr).([tminus1AggVarsNames{pp},'_tminus1'])=temp;
                end
            end
        end
    end
end

for tt=1:T-1

    % Get t-1 PricePath, ParamPath and AggVars before we update them
    if tt>1
        if use_tminus1price==1
            for pp=1:length(tminus1priceNames)
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                for ii=1:N_i
                    iistr=Names_i{ii};
                    Parameters_ii.(iistr).([tminus1priceNames{pp},'_tminus1'])=Parameters_ii.(iistr).(tminus1priceNames{pp});
                end
            end
        end
        if use_tminus1params==1
            for pp=1:length(tminus1paramNames)
                Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
                for ii=1:N_i
                    iistr=Names_i{ii};
                    Parameters_ii.(iistr).([tminus1paramNames{pp},'_tminus1'])=Parameters_ii.(iistr).(tminus1paramNames{pp});
                end
            end
        end
        if use_tminus1AggVars==1
            for pp=1:length(tminus1AggVarsNames)
                % The AggVars have not yet been updated, so they still contain previous period values
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
                for ii=1:N_i
                    iistr=Names_i{ii};
                    Parameters_ii.(iistr).([tminus1AggVarsNames{pp},'_tminus1'])=Parameters_ii.(iistr).(tminus1AggVarsNames{pp});
                end
            end
        end
    end

    % Update current PricePath and ParamPath
    % Parameters gets the whole of a price that depends on ptype (all N_i blocks of it), Parameters_ii just
    % its own block. Each ptype gets its own ParamPath; Parameters the parameters common to all ptypes.
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathOld(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        for ii=1:N_i
            iistr=Names_i{ii};
            if PTypeStructure.PricePath_Idependsonptype(kk)==1
                Parameters_ii.(iistr).(PricePathNames{kk})=PricePathOld(tt,PricePathSizeVec(1,kk)+(ii-1)*PTypeStructure.PricePathWidth_ii(kk):PricePathSizeVec(1,kk)+ii*PTypeStructure.PricePathWidth_ii(kk)-1);
            else
                Parameters_ii.(iistr).(PricePathNames{kk})=Parameters.(PricePathNames{kk});
            end
        end
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    for ii=1:N_i
        iistr=Names_i{ii};
        for kk=1:length(PTypeStructure.(iistr).ParamPathNames)
            Parameters_ii.(iistr).(PTypeStructure.(iistr).ParamPathNames{kk})=PTypeStructure.(iistr).ParamPath(tt,PTypeStructure.(iistr).ParamPathSizeVec(1,kk):PTypeStructure.(iistr).ParamPathSizeVec(2,kk));
        end
    end

    % Current AggVars: pooled in Parameters, conditional on ptype in Parameters_ii
    for ff=1:length(FnNames)
        Parameters.(FnNames{ff})=AggVarsPooledPath(tt,ff);
        for ii=1:N_i
            iistr=Names_i{ii};
            Parameters_ii.(iistr).(FnNames{ff})=AggVarsFullPath(tt,ff,ii);
        end
    end
    % Expose per-type AggVars with _name suffix in global Parameters (as the stationary
    % GEptype solver does), so global intermediateEqns / GE eqns can reference LT_L, LT_H, ...
    for ff=1:length(FnNames)
        for ii=1:N_i
            Parameters.([FnNames{ff},'_',Names_i{ii}])=AggVarsFullPath(tt,ff,ii);
        end
    end

    % Get t+1 PricePath
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
            for ii=1:N_i
                iistr=Names_i{ii};
                if PTypeStructure.PricePath_Idependsonptype(kk)==1
                    Parameters_ii.(iistr).([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk)+(ii-1)*PTypeStructure.PricePathWidth_ii(kk):PricePathSizeVec(1,kk)+ii*PTypeStructure.PricePathWidth_ii(kk)-1);
                else
                    Parameters_ii.(iistr).([tplus1priceNames{pp},'_tplus1'])=Parameters.([tplus1priceNames{pp},'_tplus1']);
                end
            end
        end
    end

    % intermediateEqns: these have not been implemented in a manner that allows them to depend on per-ptype parameters, nor that lets them take different values for each ptype [NOTE: SHOULD be improved to allow this]
    if transpathoptions.useintermediateEqns==1
        intEqnnames=fieldnames(transpathoptions.intermediateEqns);
        for gg=1:length(intEqnnames)
            temp=real(GeneralEqmConditions_Case1_v3g(transpathoptions.intermediateEqnsCell{gg}, transpathoptions.intermediateEqnParamNames(gg).Names, Parameters));
            Parameters.(intEqnnames{gg})=temp;
            for ii=1:N_i
                Parameters_ii.(Names_i{ii}).(intEqnnames{gg})=temp;
            end
        end
    end

    %% General Eqm Eqns
    % Evaluated once, or once per ptype if the condition depends on ptype (transpathoptions.GEptype)
    gg_c=0;
    for gg=1:length(GeneralEqmEqnsCell)
        if transpathoptions.GEptype(gg)==0
            gg_c=gg_c+1;
            % Note: _v3 rather than _v3g, so on CPU rather than GPU
            GEcondnPath(tt,gg_c)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
        else
            for ii=1:N_i
                gg_c=gg_c+1;
                GEcondnPath(tt,gg_c)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters_ii.(Names_i{ii})));
            end
        end
        % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
    end
end

end
