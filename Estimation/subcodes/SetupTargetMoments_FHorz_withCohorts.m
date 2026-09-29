function [targetmomentvec,cohortmoments]=SetupTargetMoments_FHorz_withCohorts(TargetMoments,FnsToEvaluate,N_j,cohortagejshifter,useptype)
% Targets by cohort: TargetMoments.AgeConditionalStats.cohortC.(...), TargetMoments.AllStats.cohortC.(...) and
% TargetMoments.AutoCorr.cohortC.(...), where C=1,...,ncohorts indexes the entry ages in cohortagejshifter (an ordinal,
% not the age itself). Below the cohortC level the structure is exactly as without cohorts (fn.stat, or
% restriction.fn.stat). Age-conditional targets are N_j vectors and must be NaN before the cohort's entry age (the
% model has no mass there).
% Each cohort is parsed by SetupTargetMoments_FHorz() (useptype=0 or 1, as for the standard parser), and the moments
% are stacked cohort by cohort (within a cohort: AllStats, then AgeConditionalStats, then AutoCorr).
%
% AutoCorr targets (added 2026-09-29) are the outputs of EvalFnOnAgentDist_AutoCorrTransProbs_FHorz (or its PType
% wrapper): TargetMoments.AutoCorr.cohortC.fn.stat, or .cohortC.restriction.fn.stat (restriction is a name in
% simoptions.conditionalrestrictions), and with permanent types also .fn.ptype.stat and .restriction.fn.ptype.stat.
% stat is AutoCorrelation or AutoCovariance (the horizon-1 fields, vectors of length N_j-1), AutoCorrelation_kK or
% AutoCovariance_kK (horizon K, vectors of length N_j-K; the needed horizons are put into simoptions.timehorizons by
% the objective function), or Mean or StdDeviation (length N_j). Index jj of a horizon-K target is the pair of ages
% (jj, jj+K); entries with jj before the cohort's entry age must be NaN.
%
% cohortmoments holds, per cohort, everything CalibrateLifeCycleModel_withCohorts_objectivefn needs to place
% that cohort's moments: usingallstats(cc), usinglcp(cc), usingautocorr(cc), allstatmomentnames{cc},
% allstatcummomentsizes{cc}, AllStats_whichstats{cc}, FnsToEvaluate_AllStats{cc}, acsmomentnames{cc},
% acscummomentsizes{cc}, ACStats_whichstats{cc}, FnsToEvaluate_ACStats{cc}, acrmomentnames{cc} (rows of
% {restriction or '', fn, ptype or '', stat}), acrcummomentsizes{cc}, acrhorizons{cc} (the horizons K>=2 needed),
% FnsToEvaluate_AutoCorr{cc}, and offset(cc) (start of the cohort's segment of targetmomentvec).

ncohorts=length(cohortagejshifter);

% Only cohortC names are allowed at the top level
for ss={'AllStats','AgeConditionalStats','AutoCorr'}
    if isfield(TargetMoments,ss{1})
        temp=fieldnames(TargetMoments.(ss{1}));
        for tt=1:length(temp)
            if ~strncmp(temp{tt},'cohort',6)
                error(['estimoptions.cohortagejshifter is being used, so TargetMoments.',ss{1},' must only contain cohort1, cohort2, ..., but it contains ',temp{tt}])
            end
            if str2double(temp{tt}(7:end))>ncohorts
                error(['TargetMoments.',ss{1},'.',temp{tt},' but there are only ',num2str(ncohorts),' cohorts (length of estimoptions.cohortagejshifter)'])
            end
        end
    end
end
if isfield(TargetMoments,'CustomModelStats')
    error('TargetMoments.CustomModelStats is not implemented together with estimoptions.cohortagejshifter (if you want this, ask on the forum, discourse.vfitoolkit.com)')
end

FnNamesAll=fieldnames(FnsToEvaluate);

targetmomentvec=[];
cohortmoments=struct();
for cc=1:ncohorts
    cohortstr=['cohort',num2str(cc)];
    TargetMoments_cc=struct();
    if isfield(TargetMoments,'AllStats')
        if isfield(TargetMoments.AllStats,cohortstr)
            TargetMoments_cc.AllStats=TargetMoments.AllStats.(cohortstr);
        end
    end
    if isfield(TargetMoments,'AgeConditionalStats')
        if isfield(TargetMoments.AgeConditionalStats,cohortstr)
            TargetMoments_cc.AgeConditionalStats=TargetMoments.AgeConditionalStats.(cohortstr);
        end
    end
    cohortmoments.usingautocorr(cc)=0;
    if isfield(TargetMoments,'AutoCorr')
        if isfield(TargetMoments.AutoCorr,cohortstr)
            cohortmoments.usingautocorr(cc)=1;
        end
    end
    if isempty(fieldnames(TargetMoments_cc)) && cohortmoments.usingautocorr(cc)==0
        error(['estimoptions.cohortagejshifter is being used, so targets must be by cohort, but there are no targets for ',cohortstr])
    end
    [targetmomentvec_cc,cohortmoments.usingallstats(cc),cohortmoments.usinglcp(cc),~, cohortmoments.allstatmomentnames{cc},cohortmoments.allstatcummomentsizes{cc},cohortmoments.AllStats_whichstats{cc},cohortmoments.FnsToEvaluate_AllStats{cc}, cohortmoments.acsmomentnames{cc}, cohortmoments.acscummomentsizes{cc}, cohortmoments.ACStats_whichstats{cc},cohortmoments.FnsToEvaluate_ACStats{cc}, ~,~]=SetupTargetMoments_FHorz(TargetMoments_cc,FnsToEvaluate,useptype);
    % Age-conditional targets before the cohort's entry age must be NaN
    j0=cohortagejshifter(cc);
    if cohortmoments.usinglcp(cc)==1
        temp=targetmomentvec_cc(cohortmoments.allstatcummomentsizes{cc}(end)+1:cohortmoments.allstatcummomentsizes{cc}(end)+cohortmoments.acscummomentsizes{cc}(end)); % the age-conditional part
        if mod(length(temp),N_j)~=0
            error(['TargetMoments.AgeConditionalStats.',cohortstr,': age-conditional targets must be vectors of length N_j (NaN for ages not targeted)'])
        end
        temp=reshape(temp,N_j,[]); % (age, moment)
        if any(~isnan(temp(1:j0-1,:)),'all')
            error(['TargetMoments.AgeConditionalStats.',cohortstr,' has non-NaN targets at ages before the cohort enters (age ',num2str(j0),'); these must be NaN'])
        end
    end

    %% AutoCorr targets of this cohort
    if cohortmoments.usingautocorr(cc)==1
        TM=TargetMoments.AutoCorr.(cohortstr);
        % First list the (restriction, fn) nodes: a first-level name that is a FnsToEvaluate name is a function (no
        % restriction), otherwise it is a restriction whose fields must be FnsToEvaluate names
        nodes=cell(0,3); % {restriction or '', fn, the structure of stats below it}
        a1vec=fieldnames(TM);
        for a1=1:length(a1vec)
            if any(strcmp(a1vec{a1},FnNamesAll))
                nodes(end+1,:)={'',a1vec{a1},TM.(a1vec{a1})}; %#ok<AGROW>
            else
                if ~isstruct(TM.(a1vec{a1}))
                    error(['TargetMoments.AutoCorr.',cohortstr,'.',a1vec{a1},' is neither a FnsToEvaluate name nor a restriction (a structure of FnsToEvaluate names)'])
                end
                a2vec=fieldnames(TM.(a1vec{a1}));
                for a2=1:length(a2vec)
                    if ~any(strcmp(a2vec{a2},FnNamesAll))
                        error(['TargetMoments.AutoCorr.',cohortstr,'.',a1vec{a1},'.',a2vec{a2},': ',a2vec{a2},' is not a FnsToEvaluate name (below a restriction there must be a FnsToEvaluate name)'])
                    end
                    nodes(end+1,:)={a1vec{a1},a2vec{a2},TM.(a1vec{a1}).(a2vec{a2})}; %#ok<AGROW>
                end
            end
        end
        % Then the stats (or ptype then stats) below each node
        acrmomentnames=cell(0,4);
        acrmomentsizes=[];
        acrhorizons=[];
        targetmomentvec_acr=[];
        for nn=1:size(nodes,1)
            b1vec=fieldnames(nodes{nn,3});
            entries=cell(0,3); % {ptype or '', stat, value}
            for b1=1:length(b1vec)
                if isstruct(nodes{nn,3}.(b1vec{b1}))
                    if useptype==0
                        error(['TargetMoments.AutoCorr.',cohortstr,': ',b1vec{b1},' is a structure, but without permanent types the level below a FnsToEvaluate name must be a statistic'])
                    end
                    b2vec=fieldnames(nodes{nn,3}.(b1vec{b1}));
                    for b2=1:length(b2vec)
                        entries(end+1,:)={b1vec{b1},b2vec{b2},nodes{nn,3}.(b1vec{b1}).(b2vec{b2})}; %#ok<AGROW>
                    end
                else
                    entries(end+1,:)={'',b1vec{b1},nodes{nn,3}.(b1vec{b1})}; %#ok<AGROW>
                end
            end
            for ee=1:size(entries,1)
                statname=entries{ee,2};
                if strncmp(statname,'AutoCovariance',14) || strncmp(statname,'AutoCorrelation',15)
                    if strncmp(statname,'AutoCovariance',14)
                        suffix=statname(15:end);
                    else
                        suffix=statname(16:end);
                    end
                    if isempty(suffix)
                        kk=1;
                    elseif length(suffix)>2 && strcmp(suffix(1:2),'_k') && ~isnan(str2double(suffix(3:end))) && str2double(suffix(3:end))==round(str2double(suffix(3:end))) && str2double(suffix(3:end))>=1
                        kk=str2double(suffix(3:end));
                    else
                        error(['TargetMoments.AutoCorr.',cohortstr,': ',statname,' is not a recognised statistic (use AutoCorrelation, AutoCovariance, AutoCorrelation_kK, AutoCovariance_kK, Mean or StdDeviation)'])
                    end
                    if kk>N_j-1
                        error(['TargetMoments.AutoCorr.',cohortstr,': ',statname,' asks for a horizon larger than N_j-1'])
                    end
                    if kk>=2
                        acrhorizons=[acrhorizons,kk]; %#ok<AGROW>
                    end
                    expectedlength=N_j-kk;
                elseif strcmp(statname,'Mean') || strcmp(statname,'StdDeviation')
                    expectedlength=N_j;
                else
                    error(['TargetMoments.AutoCorr.',cohortstr,': ',statname,' is not a recognised statistic (use AutoCorrelation, AutoCovariance, AutoCorrelation_kK, AutoCovariance_kK, Mean or StdDeviation)'])
                end
                val=entries{ee,3};
                if length(val)~=expectedlength
                    error(['TargetMoments.AutoCorr.',cohortstr,': ',statname,' must be a vector of length ',num2str(expectedlength),' (NaN where not targeted); index jj of a horizon-K statistic is the pair of ages jj and jj+K'])
                end
                val=val(:); % column
                if any(~isnan(val(1:j0-1)))
                    error(['TargetMoments.AutoCorr.',cohortstr,': ',statname,' has non-NaN targets at ages before the cohort enters (age ',num2str(j0),'); these must be NaN'])
                end
                acrmomentnames(end+1,:)={nodes{nn,1},nodes{nn,2},entries{ee,1},statname}; %#ok<AGROW>
                acrmomentsizes(end+1)=length(val); %#ok<AGROW>
                targetmomentvec_acr=[targetmomentvec_acr; val]; %#ok<AGROW>
            end
        end
        cohortmoments.acrmomentnames{cc}=acrmomentnames;
        cohortmoments.acrcummomentsizes{cc}=cumsum(acrmomentsizes);
        cohortmoments.acrhorizons{cc}=unique(acrhorizons);
        FnsToEvaluate_AutoCorr=struct();
        fnsused=unique(nodes(:,2));
        for ff=1:length(fnsused)
            FnsToEvaluate_AutoCorr.(fnsused{ff})=FnsToEvaluate.(fnsused{ff});
        end
        cohortmoments.FnsToEvaluate_AutoCorr{cc}=FnsToEvaluate_AutoCorr;
        targetmomentvec_cc=[targetmomentvec_cc; targetmomentvec_acr];
    else
        % Placeholders
        cohortmoments.acrmomentnames{cc}=cell(1,4);
        cohortmoments.acrcummomentsizes{cc}=0;
        cohortmoments.acrhorizons{cc}=[];
        cohortmoments.FnsToEvaluate_AutoCorr{cc}=struct();
    end

    cohortmoments.offset(cc)=length(targetmomentvec);
    targetmomentvec=[targetmomentvec; targetmomentvec_cc];
end

end
