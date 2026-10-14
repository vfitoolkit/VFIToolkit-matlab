function [targetmomentvec,cohortmoments]=SetupTargetMoments_FHorz_withCohorts(TargetMoments,FnsToEvaluate,N_j,cohortagejshifter,useptype)
% Targets by cohort: TargetMoments.AgeConditionalStats.cohortC.(...), TargetMoments.AllStats.cohortC.(...) and
% TargetMoments.AutoCorr.cohortC.(...), where C=1,...,ncohorts indexes the entry ages in cohortagejshifter (an ordinal,
% not the age itself). Below the cohortC level the structure is exactly as without cohorts (fn.stat, or
% restriction.fn.stat). Age-conditional targets are N_j vectors and must be NaN before the cohort's entry age (the
% model has no mass there).
% Each cohort is parsed by SetupTargetMoments_FHorz() (useptype=0 or 1, as for the standard parser), and the moments
% are stacked cohort by cohort (within a cohort: AllStats, then AgeConditionalStats, then AutoCorr).
% AgeConditionalStats targets may name the window and log rows (fnname_log, fnname_MperiodAverage, ...; 2026-10-14): cohortmoments
% acsntimeperiods{cc}, acsntimeperiodstype{cc} and acslog{cc} (rows {restriction or '', FnsToEvaluate, M, 1 x N_j selection}) let the objective set
% simoptions.ntimeperiods, ntimeperiodstype and logwhichcombos; the last age of an M-period window target is M-1 ages after its start.
%
% AutoCorr targets (added 2026-09-29) are the outputs of EvalFnOnAgentDist_AutoCorrTransProbs_FHorz (or its PType
% wrapper): TargetMoments.AutoCorr.cohortC.fn.stat, or .cohortC.restriction.fn.stat (restriction is a name in
% simoptions.conditionalrestrictions), and with permanent types also .fn.ptype.stat and .restriction.fn.ptype.stat.
% stat is AutoCorrelation or AutoCovariance (the horizon-1 fields, vectors of length N_j-1), AutoCorrelation_horizonK or
% AutoCovariance_horizonK (horizon K, vectors of length N_j-K; the needed horizons are put into simoptions.timehorizons by
% the objective function), or Mean or StdDeviation (length N_j). Index jj of a horizon-K target is the pair of ages
% (jj, jj+K); entries with jj before the cohort's entry age must be NaN.
%
% cohortmoments holds, per cohort, everything CalibrateLifeCycleModel_withCohorts_objectivefn needs to place
% that cohort's moments: usingallstats(cc), usinglcp(cc), usingautocorr(cc), allstatmomentnames{cc},
% allstatcummomentsizes{cc}, AllStats_whichstats{cc}, FnsToEvaluate_AllStats{cc}, acsmomentnames{cc},
% acscummomentsizes{cc}, ACStats_whichstats{cc}, FnsToEvaluate_ACStats{cc}, acrmomentnames{cc} (rows of
% {restriction or '', fn, ptype or '', stat}), acrcummomentsizes{cc}, acrhorizons{cc} (the horizons needed, 1 included),
% FnsToEvaluate_AutoCorr{cc}, offset(cc) (start of the cohort's segment of targetmomentvec), and lastage(cc) (the last age any of
% the cohort's targets needs: the objective fn sets simoptions.lastDistAge to it so the cohort's agent distribution stops there;
% N_j when the cohort has AllStats targets, which pool every age).

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
        if any(~isnan(temp),'all')
            lastage_lcp=find(any(~isnan(temp),2),1,'last'); % the last targeted age
        else
            lastage_lcp=j0;
        end
        % The window and log rows of this cohort's AgeConditionalStats targets (fnname_log, fnname_MperiodAverage, fnname_MperiodSum,
        % fnname_MperiodAverage_log, fnname_MperiodSum_log; a window target is by start age): simoptions.ntimeperiods, ntimeperiodstype and
        % the start ages of the log targets (for simoptions.logwhichcombos), and an M-period window starting at the last targeted age runs M-1 ages further
        acsnames=cohortmoments.acsmomentnames{cc};
        acsM=ones(size(acsnames,1),1);
        acstype=cell(size(acsnames,1),1);
        acslog=cell(0,4); % {restriction or '', FnsToEvaluate, M, 1 x N_j selection}
        acslevel=cell(0,4);
        for mm=1:size(acsnames,1)
            names=acsnames(mm,:);
            names=names(~cellfun(@isempty,names));
            rowname=names{1};
            restr='';
            rbase=''; rM=1; rlog=0;
            for tries=1:2
                if any(strcmp(rowname,FnNamesAll))
                    rbase=rowname;
                else
                    tk=regexp(rowname,'^(.+)_(\d+)period(Average|Sum)(_log)?$','tokens','once');
                    if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                        rbase=tk{1}; rM=str2double(tk{2}); acstype{mm}=lower(tk{3}); rlog=~isempty(tk{4});
                    else
                        tk=regexp(rowname,'^(.+)_log$','tokens','once');
                        if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                            rbase=tk{1}; rlog=1;
                        end
                    end
                end
                if ~isempty(rbase) || length(names)<2
                    break
                end
                restr=names{1}; % the first name is a restriction, the row is the second
                rowname=names{2};
            end
            if isempty(rbase)
                continue % (SetupTargetMoments_FHorz has checked the names)
            end
            acsM(mm)=rM;
            val=TargetMoments_cc.AgeConditionalStats;
            for kk=1:length(names)
                val=val.(names{kk});
            end
            if isvector(val) && numel(val)==N_j
                sel=reshape(~isnan(val),[1,N_j]);
            elseif size(val,2)==N_j
                sel=any(~isnan(val),1);
            else
                sel=false(1,N_j);
            end
            if rM>1 && any(sel(N_j-rM+2:N_j))
                error(['TargetMoments.AgeConditionalStats.',cohortstr,': ',rowname,' is targeted at a start age after N_j-',num2str(rM-1),', where the ',num2str(rM),'-period window runs past age N_j (these entries must be NaN)'])
            end
            if rM>1 && any(sel)
                lastage_lcp=max(lastage_lcp,find(sel,1,'last')+rM-1);
            end
            if rlog==1
                acslog(end+1,:)={restr,rbase,rM,double(sel)}; %#ok<AGROW>
            else
                acslevel(end+1,:)={restr,rbase,rM,double(sel)}; %#ok<AGROW>
            end
        end
        wtypes=unique(acstype(acsM>1));
        if length(wtypes)>1
            error(['TargetMoments.AgeConditionalStats.',cohortstr,': the targets name both MperiodAverage and MperiodSum rows, but LifeCycleProfiles computes one or the other in a call (simoptions.ntimeperiodstype): use one of them for all the targets'])
        elseif isscalar(wtypes)
            cohortmoments.acsntimeperiodstype{cc}=wtypes{1};
        else
            cohortmoments.acsntimeperiodstype{cc}='average';
        end
        for l1=1:size(acslog,1)
            for l2=1:size(acslevel,1)
                if strcmp(acslog{l1,1},acslevel{l2,1}) && strcmp(acslog{l1,2},acslevel{l2,2}) && acslog{l1,3}==acslevel{l2,3} && any(acslog{l1,4} & acslevel{l2,4})
                    error(['TargetMoments.AgeConditionalStats.',cohortstr,': some targets ask for both a statistic of ',acslog{l1,2},' and of its log at the same age and restriction, but LifeCycleProfiles computes one or the other for each (simoptions.logwhichcombos): target only one of them at each age'])
                end
            end
        end
        cohortmoments.acsntimeperiods{cc}=unique(acsM(:)');
        cohortmoments.acslog{cc}=acslog;
    else
        cohortmoments.acsntimeperiods{cc}=1;
        cohortmoments.acsntimeperiodstype{cc}='average';
        cohortmoments.acslog{cc}=cell(0,4);
    end

    %% AutoCorr targets of this cohort
    if cohortmoments.usingautocorr(cc)==1
        TM=TargetMoments.AutoCorr.(cohortstr);
        % First list the (restriction, fn) nodes: a first-level name that is a FnsToEvaluate name is a function (no
        % restriction), otherwise it is a restriction whose fields must be FnsToEvaluate names
        % (a FnsToEvaluate name here may also be one of its rows of simoptions.ntimeperiods/logwhichcombos: fnname_log,
        % fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log)
        nodes=cell(0,7); % {restriction or '', row name, the structure of stats below it, the FnsToEvaluate, M, 'average'/'sum' (M>1), log}
        a1vec=fieldnames(TM);
        for a1=1:length(a1vec)
            if isstruct(TM.(a1vec{a1}))
                n_a2=length(fieldnames(TM.(a1vec{a1})));
            else
                n_a2=0;
            end
            for a2=0:n_a2 % a2=0: a1 as the row; a2>0: a1 as a restriction
                if a2==0
                    rowname=a1vec{a1};
                else
                    a2vec=fieldnames(TM.(a1vec{a1}));
                    rowname=a2vec{a2};
                end
                % the FnsToEvaluate, the number of periods and the log of the row
                rbase=''; rM=1; rtype=''; rlog=0;
                if any(strcmp(rowname,FnNamesAll))
                    rbase=rowname;
                else
                    tk=regexp(rowname,'^(.+)_(\d+)period(Average|Sum)(_log)?$','tokens','once');
                    if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                        rbase=tk{1}; rM=str2double(tk{2}); rtype=lower(tk{3}); rlog=~isempty(tk{4});
                        if rM<2
                            error(['TargetMoments.AutoCorr.',cohortstr,': ',rowname,' names a window of ',tk{2},' periods: a single period is the FnsToEvaluate name itself (',tk{1},', or ',tk{1},'_log)'])
                        end
                    else
                        tk=regexp(rowname,'^(.+)_log$','tokens','once');
                        if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                            rbase=tk{1}; rlog=1;
                        end
                    end
                end
                if a2==0
                    if ~isempty(rbase)
                        nodes(end+1,:)={'',rowname,TM.(a1vec{a1}),rbase,rM,rtype,rlog}; %#ok<AGROW>
                        break % a row, not a restriction
                    elseif ~isstruct(TM.(a1vec{a1}))
                        error(['TargetMoments.AutoCorr.',cohortstr,'.',a1vec{a1},' is neither a FnsToEvaluate name (or one of its rows fnname_log, fnname_MperiodAverage, ...) nor a restriction (a structure of FnsToEvaluate names)'])
                    end
                else
                    if isempty(rbase)
                        error(['TargetMoments.AutoCorr.',cohortstr,'.',a1vec{a1},'.',rowname,': ',rowname,' is not a FnsToEvaluate name, nor fnname_log, fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log of one (below a restriction there must be one)'])
                    end
                    nodes(end+1,:)={a1vec{a1},rowname,TM.(a1vec{a1}).(rowname),rbase,rM,rtype,rlog}; %#ok<AGROW>
                end
            end
        end
        % simoptions.ntimeperiods and simoptions.ntimeperiodstype of this cohort (one type per command call: the targets cannot mix MperiodAverage and MperiodSum)
        acrM=cell2mat(nodes(:,5));
        wtypes=unique(nodes(acrM>1,6));
        if length(wtypes)>1
            error(['TargetMoments.AutoCorr.',cohortstr,': the targets name both MperiodAverage and MperiodSum rows, but the command computes one or the other in a call (simoptions.ntimeperiodstype): use one of them for all the targets'])
        elseif isscalar(wtypes)
            cohortmoments.acrntimeperiodstype{cc}=wtypes{1};
        else
            cohortmoments.acrntimeperiodstype{cc}='average';
        end
        cohortmoments.acrntimeperiods{cc}=unique(acrM(:)');
        acrlog=cell(0,4); % {restriction or '', FnsToEvaluate, M, 1 x N_j selection}: the start ages of the targets of the log (simoptions.logwhichcombos)
        acrlevel=cell(0,4); % the same for the targets at the level (to refuse a target of both at one start age)
        % Then the stats (or ptype then stats) below each node
        acrmomentnames=cell(0,4);
        acrmomentsizes=[];
        acrhorizons=[];
        targetmomentvec_acr=[];
        lastage_acr=j0;
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
                    elseif length(suffix)>8 && strcmp(suffix(1:8),'_horizon') && ~isnan(str2double(suffix(9:end))) && str2double(suffix(9:end))==round(str2double(suffix(9:end))) && str2double(suffix(9:end))>=1
                        kk=str2double(suffix(9:end));
                    else
                        error(['TargetMoments.AutoCorr.',cohortstr,': ',statname,' is not a recognised statistic (use AutoCorrelation, AutoCovariance, AutoCorrelation_horizonK, AutoCovariance_horizonK, Mean or StdDeviation)'])
                    end
                    if kk>N_j-1
                        error(['TargetMoments.AutoCorr.',cohortstr,': ',statname,' asks for a horizon larger than N_j-1'])
                    end
                    if kk>=1
                        acrhorizons=[acrhorizons,kk]; %#ok<AGROW> the command computes exactly the horizons asked for (horizon 1 too)
                    end
                    expectedlength=N_j-kk;
                    kneeded=kk; % index jj of a horizon-K target is the pair (jj, jj+K), so age jj+K is needed
                elseif strcmp(statname,'Mean') || strcmp(statname,'StdDeviation')
                    expectedlength=N_j;
                    kneeded=0;
                else
                    error(['TargetMoments.AutoCorr.',cohortstr,': ',statname,' is not a recognised statistic (use AutoCorrelation, AutoCovariance, AutoCorrelation_horizonK, AutoCovariance_horizonK, Mean or StdDeviation)'])
                end
                val=entries{ee,3};
                if length(val)~=expectedlength
                    error(['TargetMoments.AutoCorr.',cohortstr,': ',statname,' must be a vector of length ',num2str(expectedlength),' (NaN where not targeted); index jj of a horizon-K statistic is the pair of ages jj and jj+K'])
                end
                val=val(:); % column
                if any(~isnan(val(1:j0-1)))
                    error(['TargetMoments.AutoCorr.',cohortstr,': ',statname,' has non-NaN targets at ages before the cohort enters (age ',num2str(j0),'); these must be NaN'])
                end
                if any(~isnan(val))
                    lastage_acr=max(lastage_acr,find(~isnan(val),1,'last')+kneeded+nodes{nn,5}-1); % an M-period window starting at the last age of the horizon runs M-1 ages further
                end
                acrmomentnames(end+1,:)={nodes{nn,1},nodes{nn,2},entries{ee,1},statname}; %#ok<AGROW>
                acrmomentsizes(end+1)=length(val); %#ok<AGROW>
                targetmomentvec_acr=[targetmomentvec_acr; val]; %#ok<AGROW>
                sel=zeros(1,N_j);
                sel(1:length(val))=~isnan(val'); % entry j selects start age j
                if nodes{nn,7}==1
                    acrlog(end+1,:)={nodes{nn,1},nodes{nn,4},nodes{nn,5},sel}; %#ok<AGROW>
                else
                    acrlevel(end+1,:)={nodes{nn,1},nodes{nn,4},nodes{nn,5},sel}; %#ok<AGROW>
                end
            end
        end
        % A target of a statistic and one of its log at the same start age and restriction cannot both be computed (simoptions.logwhichcombos)
        for l1=1:size(acrlog,1)
            for l2=1:size(acrlevel,1)
                if strcmp(acrlog{l1,1},acrlevel{l2,1}) && strcmp(acrlog{l1,2},acrlevel{l2,2}) && acrlog{l1,3}==acrlevel{l2,3} && any(acrlog{l1,4} & acrlevel{l2,4})
                    error(['TargetMoments.AutoCorr.',cohortstr,': some targets ask for both a statistic of ',acrlog{l1,2},' and of its log at the same start age and restriction, but the command computes one or the other for each (simoptions.logwhichcombos): target only one of them at each start age'])
                end
            end
        end
        cohortmoments.acrlog{cc}=acrlog;
        cohortmoments.acrmomentnames{cc}=acrmomentnames;
        cohortmoments.acrcummomentsizes{cc}=cumsum(acrmomentsizes);
        cohortmoments.acrhorizons{cc}=unique(acrhorizons);
        FnsToEvaluate_AutoCorr=struct();
        fnsused=unique(nodes(:,4)); % the FnsToEvaluate of the rows
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
        cohortmoments.acrntimeperiods{cc}=1;
        cohortmoments.acrntimeperiodstype{cc}='average';
        cohortmoments.acrlog{cc}=cell(0,4);
    end

    %% The last age this cohort's targets need (the objective fn stops the cohort's agent distribution there, simoptions.lastDistAge)
    lastage=j0;
    if cohortmoments.usingallstats(cc)==1
        lastage=N_j; % AllStats pool every age
    end
    if cohortmoments.usinglcp(cc)==1
        lastage=max(lastage,lastage_lcp);
    end
    if cohortmoments.usingautocorr(cc)==1
        lastage=max(lastage,lastage_acr);
    end
    cohortmoments.lastage(cc)=lastage;

    cohortmoments.offset(cc)=length(targetmomentvec);
    targetmomentvec=[targetmomentvec; targetmomentvec_cc];
end

end
