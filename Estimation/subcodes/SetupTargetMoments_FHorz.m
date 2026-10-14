function [targetmomentvec,usingallstats,usinglcp,usingcustomstats, allstatmomentnames,allstatcummomentsizes,AllStats_whichstats, FnsToEvaluate_AllStats, acsmomentnames, acscummomentsizes, ACStats_whichstats, FnsToEvaluate_ACStats,cmsmomentnames, cmscummomentsizes, selectors, usingautocorr, autocorrmomentnames, autocorrcummomentsizes, FnsToEvaluate_AutoCorr, autocorrtimehorizons, usingcrosssec, crosssecmomentnames, crossseccummomentsizes, FnsToEvaluate_CrossSec, usingagecrosssec, agecrosssecmomentnames, agecrossseccummomentsizes, FnsToEvaluate_AgeCrossSec]=SetupTargetMoments_FHorz(TargetMoments,FnsToEvaluate,useptype,N_j,simoptions,Names_i)
% useptype is 0 or 1
% N_j, simoptions and (with permanent types) Names_i are optional (the estimation commands do not yet pass them): when they are given, the last
% output 'selectors' holds the per-combination selectors for simoptions.whichcombos/whichstats of the stats commands:
% With useptype==1 the selectors carry a trailing type dimension of N_i+1 (one slot per permanent type in the order of Names_i, then the
% grouped stats): AllStats [nFns,1+nRestr,N_i+1] / [..,7], AgeConditionalStats [nFns,maxAgeGroups,1+nRestr,N_i+1] / [..,7], maxAgeGroups being
% the largest number of age groups over the ptypes (each ptype's groups shifted by its agejshifter), as LifeCycleProfiles_FHorz_Case1_PType lays them out.
%   selectors.AllStats.whichcombos  [nFns,1+nRestr]           selectors.AllStats.whichstats  [nFns,1+nRestr,7]
%   selectors.ACStats.whichcombos   [nFns,nAgeGroups,1+nRestr] selectors.ACStats.whichstats   [nFns,nAgeGroups,1+nRestr,7]
% with the functions in the order of FnsToEvaluate_AllStats/FnsToEvaluate_ACStats (the order of FnsToEvaluate) and the pages
% being the unrestricted stats and then simoptions.conditionalrestrictions in fieldnames order. An age-conditional target that is
% NaN at an age turns that age off for that statistic. Otherwise selectors is an empty struct.
% The 7-vectors AllStats_whichstats and ACStats_whichstats are the union over all targets (used when caliboptions.whichcombos=0
% is not set, i.e. by the estimation commands).
% Three more kinds of target stand alongside AllStats and AgeConditionalStats (2026-10-07; with permanent types since 2026-10-08): TargetMoments.AutoCorrTransProbs
% (EvalFnOnAgentDist_AutoCorrTransProbs_FHorz), .CrossSectionCovarCorr (EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz) and
% .AgeConditionalCrossSectionCovarCorr (the age-conditional version), each nested like its command's output. They enter targetmomentvec
% after AgeConditionalStats and before CustomModelStats, with their own using* flag, names table, cumulative sizes, reduced FnsToEvaluate
% and selector (selectors.AutoCorr/.CrossSec/.AgeCrossSec, whichcombos only: these commands have no per-combination whichstats);
% autocorrtimehorizons holds the horizons read off the AutoCorr target names (1 for one without a _horizonK suffix), for simoptions.timehorizons.
% AgeConditionalStats targets may also be of the rows of simoptions.ntimeperiods and simoptions.logwhichcombos (2026-10-14): fnname_log,
% fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log (a window target is by start age, 1 x N_j).
% selectors.ACStats then has haswindowlog=1, ntimeperiods (the rows of whichcombos/whichstats are in blocks of these), ntimeperiodstype,
% logwhichcombos (no type dimension) and agegroupings (1:N_j when the targets are all windows and simoptions.agegroupings is not
% 1:N_j, else []); when single-period targets by age groups and window targets are both present, split=1 and the window targets are a
% second LifeCycleProfiles call, selectors.ACStatsW (by start age, the same fields).

% Only calculate each of AllStats and LifeCycleProfiles when being used (so as faster when not using both)
if isfield(TargetMoments,'AllStats')
    usingallstats=1;
else
    usingallstats=0;
end
if isfield(TargetMoments,'AgeConditionalStats')
    usinglcp=1;
else
    usinglcp=0;
end
usingautocorr=double(isfield(TargetMoments,'AutoCorrTransProbs'));
usingcrosssec=double(isfield(TargetMoments,'CrossSectionCovarCorr'));
usingagecrosssec=double(isfield(TargetMoments,'AgeConditionalCrossSectionCovarCorr'));
if isfield(TargetMoments,'CustomModelStats')
    usingcustomstats=1;
else
    usingcustomstats=0;
end

temp=fieldnames(TargetMoments);
for a1=1:length(temp)
    if ~isempty(setdiff(temp{a1}, {'AllStats','AgeConditionalStats','AutoCorrTransProbs','CrossSectionCovarCorr','AgeConditionalCrossSectionCovarCorr','CustomModelStats'}))
        fprintf(' \n')
        fprintf(' \n')
        fprintf('The following error is because %s is a field in TargetMoments but does not fit allowed formats (e.g., AllStats, etc.) \n', temp{a1})
        error('TargetMoments contains a field with a problematic name (see line above)')
    end
end

selectors=struct(); % the per-combination selectors, built in the useptype==0 branch when N_j and simoptions are given
if useptype==0
    % Restriction names (page order of simoptions.whichcombos) and the number of age groups, when they were given
    if exist('simoptions','var') && isfield(simoptions,'conditionalrestrictions')
        RestrNames=fieldnames(simoptions.conditionalrestrictions);
    else
        RestrNames={};
    end
    nRestr=length(RestrNames);
    FnNamesAll=fieldnames(FnsToEvaluate);
    if exist('simoptions','var') && exist('N_j','var')
        buildselectors=1;
        if isfield(simoptions,'agegroupings')
            ngroups=length(simoptions.agegroupings);
        else
            ngroups=N_j;
        end
    else
        buildselectors=0;
    end
    targetmomentvec=[]; % Can't preallocate as have no idea how big this will be
    if (usingautocorr==1 || usingcrosssec==1 || usingagecrosssec==1) && buildselectors==0
        error('TargetMoments.AutoCorrTransProbs, .CrossSectionCovarCorr and .AgeConditionalCrossSectionCovarCorr need N_j and simoptions to be passed to SetupTargetMoments_FHorz (the calibration and estimation commands do; the three-input legacy call cannot take these targets)')
    end
    %% AllStats
    if usingallstats==1
        % Walk TargetMoments.AllStats: one row of names per target. Two levels is (fn,stat); three levels is
        % (restriction,fn,stat) or (fn,'MoreInequality',substat). The values are vectorised column-major with (:).
        allstatmomentnames=cell(0,3);
        allstatmomentsizes=[];
        a1vec=fieldnames(TargetMoments.AllStats);
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}));
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}));
                    for a3=1:length(a3vec)
                        temp=TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3});
                        if isstruct(temp)
                            error(['TargetMoments.AllStats.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},' is a structure: without permanent types a target has at most three levels (restriction.fn.stat or fn.MoreInequality.substat)'])
                        end
                        targetmomentvec=[targetmomentvec; temp(:)]; % append to end
                        allstatmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3}};
                        allstatmomentsizes(end+1)=numel(temp);
                    end
                else
                    temp=TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2});
                    targetmomentvec=[targetmomentvec; temp(:)]; % append to end
                    allstatmomentnames(end+1,:)={a1vec{a1},a2vec{a2},''};
                    allstatmomentsizes(end+1)=numel(temp);
                end
            end
        end
        allstatcummomentsizes=cumsum(allstatmomentsizes);
        % Classify each row as (fn, stat, page): page 1 is unrestricted, page 1+rr the rr-th conditional restriction
        nrows=size(allstatmomentnames,1);
        rowfn=cell(nrows,1);
        rowstat=cell(nrows,1);
        rowpage=zeros(nrows,1);
        for cc=1:nrows
            a1=allstatmomentnames{cc,1}; a2=allstatmomentnames{cc,2}; a3=allstatmomentnames{cc,3};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
            if isrestr && isfn
                error(['TargetMoments.AllStats.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
            elseif isrestr || (~isfn && ~exist('simoptions','var')) % (without simoptions a first-level name that is not a function is taken to be a restriction, as the estimation commands rely on; no page is needed as no selectors are built)
                if isempty(a3)
                    error(['TargetMoments.AllStats.',a1,'.',a2,': a target under a conditional restriction needs a function and then a statistic (e.g. TargetMoments.AllStats.',a1,'.',a2,'.Mean)'])
                end
                if isrestr
                    rowpage(cc)=1+find(strcmp(RestrNames,a1));
                else
                    rowpage(cc)=0;
                end
                rowfn{cc}=a2;
                rowstat{cc}=a3;
            elseif isfn
                rowpage(cc)=1;
                rowfn{cc}=a1;
                rowstat{cc}=a2; % (with three levels a2 is MoreInequality and a3 the sub-statistic)
            else
                error(['TargetMoments.AllStats.',a1,' is neither a FnsToEvaluate nor a conditional restriction in simoptions.conditionalrestrictions (restrictions must be in simoptions before CalibrateLifeCycleModel is called)'])
            end
            if ~any(strcmp(FnNamesAll,rowfn{cc}))
                error(['TargetMoments.AllStats: ',rowfn{cc},' is not one of the FnsToEvaluate'])
            end
        end
        % The functions AllStats has to evaluate (every function named by a target, in the order of FnsToEvaluate)
        FnsToEvaluate_AllStats=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowfn,FnNamesAll{ff}))
                FnsToEvaluate_AllStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesA=fieldnames(FnsToEvaluate_AllStats);
        % whichstats: the union over all targets (the 7-vector), and per (fn,page) [nFns,1+nRestr,7]
        AllStats_whichstats=zeros(7,1);
        if buildselectors==1
            AllStats_whichstatsArr=zeros(length(FnNamesA),1+nRestr,7);
        end
        for cc=1:nrows
            ff=find(strcmp(FnNamesA,rowfn{cc}));
            pp=rowpage(cc);
            temp=TargetMoments.AllStats; % the target's values: a target that is NaN in full selects nothing (it is dropped, with a warning below)
            for kk=1:size(allstatmomentnames,2)
                if ~isempty(allstatmomentnames{cc,kk})
                    temp=temp.(allstatmomentnames{cc,kk});
                end
            end
            rowallnan=all(isnan(temp(:)));
            % see StatsFromWeightedGrid: 1 Mean, 2 Median, 3 Variance/StdDeviation, 4 Gini(3)/LorenzCurve(1), 5 Minimum/Maximum, 6 Quantiles, 7 MoreInequality
            switch rowstat{cc}
                case 'Mean'
                    AllStats_whichstats(1)=1; sidx=1;
                case 'Median'
                    AllStats_whichstats(2)=1; sidx=2;
                case 'RatioMeanToMedian'
                    AllStats_whichstats(1:2)=1; sidx=[1,2];
                case {'Variance','StdDeviation'}
                    AllStats_whichstats(3)=1; sidx=3;
                case 'Gini'
                    if AllStats_whichstats(4)==0 % 3 is the Gini without the Lorenz curve; do not overwrite a 1 from a LorenzCurve target
                        AllStats_whichstats(4)=3;
                    end
                    sidx=4;
                case 'LorenzCurve'
                    AllStats_whichstats(4)=1; sidx=4;
                case {'Minimum','Maximum'}
                    AllStats_whichstats(5)=1; sidx=5;
                case {'QuantileCutoffs','QuantileMeans'}
                    AllStats_whichstats(6)=1; sidx=6;
                case 'MoreInequality'
                    AllStats_whichstats(7)=1; sidx=7;
                otherwise
                    error(['TargetMoments.AllStats: ',rowstat{cc},' is not a statistic that AllStats produces (Mean, Median, RatioMeanToMedian, Variance, StdDeviation, Gini, LorenzCurve, Minimum, Maximum, QuantileCutoffs, QuantileMeans, MoreInequality)'])
            end
            if buildselectors==1 && rowallnan==0 % (without simoptions the page of a restricted target is unknown and no selectors are built)
                for ss=sidx
                    if ss==4 % Gini/LorenzCurve: 3 is Gini only, 1 is the Lorenz curve (and Gini); a LorenzCurve target overrides a Gini-only 3
                        if strcmp(rowstat{cc},'LorenzCurve')
                            AllStats_whichstatsArr(ff,pp,4)=1;
                        elseif AllStats_whichstatsArr(ff,pp,4)==0
                            AllStats_whichstatsArr(ff,pp,4)=3;
                        end
                    else
                        AllStats_whichstatsArr(ff,pp,ss)=1;
                    end
                end
            end
        end
        if buildselectors==1
            AllStats_whichcombos=double(any(AllStats_whichstatsArr,3)); % [nFns,1+nRestr]
        else
            AllStats_whichstatsArr=[];
            AllStats_whichcombos=[];
        end
    else
        allstatmomentnames=cell(1,3);
        allstatcummomentsizes=0;
        AllStats_whichstats=zeros(7,1);
        FnsToEvaluate_AllStats=struct();
        AllStats_whichstatsArr=[];
        AllStats_whichcombos=[];
    end
    %% AgeConditionalStats (LifeCycleProfiles)
    if usinglcp==1
        % Same walk. A target is a vector over the age groups, or a matrix with the age groups as its columns
        % (LorenzCurve, QuantileCutoffs, QuantileMeans); either way it is vectorised column-major with (:), which
        % is the layout of the LifeCycleProfiles output. NaN in a target omits that entry (and that age is not computed).
        acsmomentnames=cell(0,3);
        acsmomentsizes=[];
        a1vec=fieldnames(TargetMoments.AgeConditionalStats);
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1}));
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}));
                    for a3=1:length(a3vec)
                        temp=TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3});
                        if isstruct(temp)
                            error(['TargetMoments.AgeConditionalStats.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},' is a structure: without permanent types a target has at most three levels (restriction.fn.stat or fn.MoreInequality.substat)'])
                        end
                        targetmomentvec=[targetmomentvec; temp(:)]; % append to end
                        acsmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3}};
                        acsmomentsizes(end+1)=numel(temp);
                    end
                else
                    temp=TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2});
                    targetmomentvec=[targetmomentvec; temp(:)]; % append to end
                    acsmomentnames(end+1,:)={a1vec{a1},a2vec{a2},''};
                    acsmomentsizes(end+1)=numel(temp);
                end
            end
        end
        acscummomentsizes=cumsum(acsmomentsizes);
        nrows=size(acsmomentnames,1);
        rowfn=cell(nrows,1);
        rowstat=cell(nrows,1);
        rowpage=zeros(nrows,1);
        for cc=1:nrows
            a1=acsmomentnames{cc,1}; a2=acsmomentnames{cc,2}; a3=acsmomentnames{cc,3};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
            if ~isfn % a window or log row of a FnsToEvaluate (simoptions.ntimeperiods, simoptions.logwhichcombos)
                tk=regexp(a1,'^(.+)_(\d+)period(Average|Sum)(_log)?$','tokens','once');
                if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                    isfn=true;
                else
                    tk=regexp(a1,'^(.+)_log$','tokens','once');
                    isfn=(~isempty(tk) && any(strcmp(FnNamesAll,tk{1})));
                end
            end
            if isrestr && isfn
                error(['TargetMoments.AgeConditionalStats.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
            elseif isrestr || (~isfn && ~exist('simoptions','var')) % (without simoptions a first-level name that is not a function is taken to be a restriction, as the estimation commands rely on; no page is needed as no selectors are built)
                if isempty(a3)
                    error(['TargetMoments.AgeConditionalStats.',a1,'.',a2,': a target under a conditional restriction needs a function and then a statistic (e.g. TargetMoments.AgeConditionalStats.',a1,'.',a2,'.Mean)'])
                end
                if isrestr
                    rowpage(cc)=1+find(strcmp(RestrNames,a1));
                else
                    rowpage(cc)=0;
                end
                rowfn{cc}=a2;
                rowstat{cc}=a3;
            elseif isfn
                rowpage(cc)=1;
                rowfn{cc}=a1;
                rowstat{cc}=a2;
            else
                error(['TargetMoments.AgeConditionalStats.',a1,' is neither a FnsToEvaluate (or one of its rows fnname_log, fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log) nor a conditional restriction in simoptions.conditionalrestrictions (restrictions must be in simoptions before CalibrateLifeCycleModel is called)'])
            end
        end
        % The output rows of LifeCycleProfiles: a FnsToEvaluate name, or (simoptions.ntimeperiods, simoptions.logwhichcombos) fnname_log,
        % fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log: rowbase is the FnsToEvaluate, rowM the
        % number of periods (1 for a single period), rowtype 'average' or 'sum' (M>1), rowlog 1 for the log. A window target is by start age (1 x N_j).
        rowbase=cell(nrows,1);
        rowM=ones(nrows,1);
        rowtype=cell(nrows,1);
        rowlog=zeros(nrows,1);
        for cc=1:nrows
            rowbase{cc}=rowfn{cc};
            if ~any(strcmp(FnNamesAll,rowfn{cc}))
                tk=regexp(rowfn{cc},'^(.+)_(\d+)period(Average|Sum)(_log)?$','tokens','once');
                if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                    rowbase{cc}=tk{1};
                    rowM(cc)=str2double(tk{2});
                    rowtype{cc}=lower(tk{3});
                    rowlog(cc)=~isempty(tk{4});
                    if rowM(cc)<2
                        error(['TargetMoments.AgeConditionalStats: ',rowfn{cc},' names a window of ',tk{2},' periods: a single period is the FnsToEvaluate name itself (',tk{1},', or ',tk{1},'_log)'])
                    end
                else
                    tk=regexp(rowfn{cc},'^(.+)_log$','tokens','once');
                    if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                        rowbase{cc}=tk{1};
                        rowlog(cc)=1;
                    else
                        error(['TargetMoments.AgeConditionalStats: ',rowfn{cc},' is not one of the FnsToEvaluate (nor fnname_log, fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log of one)'])
                    end
                end
            end
        end
        FnsToEvaluate_ACStats=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowbase,FnNamesAll{ff}))
                FnsToEvaluate_ACStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesL=fieldnames(FnsToEvaluate_ACStats);
        % simoptions.ntimeperiodstype (one per call, so the targets cannot mix MperiodAverage and MperiodSum)
        wtypes=unique(rowtype(rowM>1));
        if length(wtypes)>1
            error('TargetMoments.AgeConditionalStats: the targets name both MperiodAverage and MperiodSum rows, but LifeCycleProfiles computes one or the other in a call (simoptions.ntimeperiodstype): use one of them for all the targets')
        elseif isscalar(wtypes)
            ACStats_ntimeperiodstype=wtypes{1};
        else
            ACStats_ntimeperiodstype='average';
        end
        % The calls of LifeCycleProfiles: windows (M>1) are by start age, so they need agegroupings=1:1:N_j. When single-period targets
        % are by age groups (simoptions.agegroupings other than 1:1:N_j) and there are window targets too, the windows get a second call
        % (callW, by start age); otherwise one call does all (with agegroupings=1:1:N_j forced when there are only window targets).
        if buildselectors==1
            agedefault=(~isfield(simoptions,'agegroupings') || isequal(reshape(simoptions.agegroupings,1,[]),1:1:N_j));
        else
            agedefault=1;
        end
        ACStats_split=(any(rowM>1) && any(rowM==1) && ~agedefault);
        ACStats_haswindowlog=(any(rowM>1) || any(rowlog==1)); % some target is of a window or a log (the objective then sets simoptions.ntimeperiods, ntimeperiodstype, logwhichcombos)
        rowcall=ones(nrows,1); % 1 the main call, 2 the window call
        if ACStats_split
            rowcall(rowM>1)=2;
            ACStats_ntimeperiods=1;
            ACStatsW_ntimeperiods=unique(rowM(rowM>1)');
            ngroupsA=ngroups;
        else
            ACStats_ntimeperiods=unique(rowM(:)');
            ACStatsW_ntimeperiods=[];
            if any(rowM>1) && buildselectors==1 % (without the selectors, the legacy call of the cohort route, N_j is not given and no arrays are built)
                ngroupsA=N_j;
            elseif buildselectors==1
                ngroupsA=ngroups;
            else
                ngroupsA=0;
            end
        end
        nFnsL=length(FnNamesL);
        % whichstats: the union (7-vector), and per (row,agegroup,page) [nRows,ngroups,1+nRestr,7] with an age on only where the target
        % is not NaN; the rows are in blocks of the window lengths of the call (as simoptions.whichcombos with simoptions.ntimeperiods)
        ACStats_whichstats=zeros(7,1);
        if buildselectors==1
            ACStats_whichstatsArr=zeros(length(ACStats_ntimeperiods)*nFnsL,ngroupsA,1+nRestr,7);
            ACStats_logwhichcombos=zeros(length(ACStats_ntimeperiods)*nFnsL,ngroupsA,1+nRestr);
            ACStats_levelcombos=zeros(length(ACStats_ntimeperiods)*nFnsL,ngroupsA,1+nRestr);
            ACStatsW_whichstatsArr=zeros(length(ACStatsW_ntimeperiods)*nFnsL,N_j,1+nRestr,7);
            ACStatsW_logwhichcombos=zeros(length(ACStatsW_ntimeperiods)*nFnsL,N_j,1+nRestr);
            ACStatsW_levelcombos=zeros(length(ACStatsW_ntimeperiods)*nFnsL,N_j,1+nRestr);
        end
        for cc=1:nrows
            if rowcall(cc)==1
                ff=(find(ACStats_ntimeperiods==rowM(cc))-1)*nFnsL+find(strcmp(FnNamesL,rowbase{cc})); % the row: the block of its window length
                nslot=ngroupsA;
            else
                ff=(find(ACStatsW_ntimeperiods==rowM(cc))-1)*nFnsL+find(strcmp(FnNamesL,rowbase{cc}));
                nslot=N_j;
            end
            pp=rowpage(cc);
            if buildselectors==1
                if isempty(acsmomentnames{cc,3})
                    temp=TargetMoments.AgeConditionalStats.(acsmomentnames{cc,1}).(acsmomentnames{cc,2});
                else
                    temp=TargetMoments.AgeConditionalStats.(acsmomentnames{cc,1}).(acsmomentnames{cc,2}).(acsmomentnames{cc,3});
                end
                if isvector(temp) && numel(temp)==nslot
                    ageon=reshape(~isnan(temp),[1,nslot]);
                elseif ismatrix(temp) && size(temp,2)==nslot
                    ageon=any(~isnan(temp),1); % a matrix target: the age group is on if any entry of its column is targeted
                elseif rowM(cc)>1 || (rowcall(cc)==1 && nslot~=ngroups)
                    error(['TargetMoments.AgeConditionalStats target for ',rowfn{cc},' ',rowstat{cc},' has ',num2str(numel(temp)),' entries, but it is by start age (1 x ',num2str(N_j),', or a matrix with N_j columns): ',num2str(nslot),' are needed'])
                else
                    error(['TargetMoments.AgeConditionalStats target for ',rowfn{cc},' ',rowstat{cc},' has ',num2str(numel(temp)),' entries, but there are ',num2str(ngroups),' age groups (it must be a vector over the age groups, or a matrix with the age groups as columns)'])
                end
                if rowM(cc)>1 && any(ageon(N_j-rowM(cc)+2:N_j))
                    error(['TargetMoments.AgeConditionalStats: ',rowfn{cc},' is targeted at a start age after N_j-',num2str(rowM(cc)-1),', where the ',num2str(rowM(cc)),'-period window runs past age N_j (these entries must be NaN)'])
                end
                if rowcall(cc)==1
                    if rowlog(cc)==1
                        ACStats_logwhichcombos(ff,ageon,pp)=1;
                    else
                        ACStats_levelcombos(ff,ageon,pp)=1;
                    end
                else
                    if rowlog(cc)==1
                        ACStatsW_logwhichcombos(ff,ageon,pp)=1;
                    else
                        ACStatsW_levelcombos(ff,ageon,pp)=1;
                    end
                end
            else
                ageon=1;
            end
            switch rowstat{cc}
                case 'Mean'
                    ACStats_whichstats(1)=1; sidx=1;
                case 'Median'
                    ACStats_whichstats(2)=1; sidx=2;
                case 'RatioMeanToMedian'
                    ACStats_whichstats(1:2)=1; sidx=[1,2];
                case {'Variance','StdDeviation'}
                    ACStats_whichstats(3)=1; sidx=3;
                case 'Gini'
                    if ACStats_whichstats(4)==0 % 3 is the Gini without the Lorenz curve; do not overwrite a 2 from a LorenzCurve target
                        ACStats_whichstats(4)=3;
                    end
                    sidx=4;
                case 'LorenzCurve'
                    ACStats_whichstats(4)=2; sidx=4;
                case {'Minimum','Maximum'}
                    ACStats_whichstats(5)=1; sidx=5;
                case {'QuantileCutoffs','QuantileMeans'}
                    ACStats_whichstats(6)=1; sidx=6;
                case 'MoreInequality'
                    ACStats_whichstats(7)=1; sidx=7;
                otherwise
                    error(['TargetMoments.AgeConditionalStats: ',rowstat{cc},' is not a statistic that LifeCycleProfiles produces (Mean, Median, RatioMeanToMedian, Variance, StdDeviation, Gini, LorenzCurve, Minimum, Maximum, QuantileCutoffs, QuantileMeans, MoreInequality)'])
            end
            if buildselectors==1
                if rowcall(cc)==1
                    wsArr=ACStats_whichstatsArr;
                else
                    wsArr=ACStatsW_whichstatsArr;
                end
                for ss=sidx
                    if ss==4 % Gini/LorenzCurve: 3 is Gini only, 2 is the Lorenz curve (and Gini); a LorenzCurve target overrides a Gini-only 3
                        if strcmp(rowstat{cc},'LorenzCurve')
                            wsArr(ff,ageon,pp,4)=2;
                        else
                            current=wsArr(ff,:,pp,4);
                            current(ageon & current==0)=3;
                            wsArr(ff,:,pp,4)=current;
                        end
                    else
                        wsArr(ff,ageon,pp,ss)=1;
                    end
                end
                if rowcall(cc)==1
                    ACStats_whichstatsArr=wsArr;
                else
                    ACStatsW_whichstatsArr=wsArr;
                end
            end
        end
        if buildselectors==1
            ACStats_whichcombos=double(any(ACStats_whichstatsArr,4)); % [nRows,ngroups,1+nRestr]
            ACStatsW_whichcombos=double(any(ACStatsW_whichstatsArr,4));
            if any(ACStats_logwhichcombos(:) & ACStats_levelcombos(:)) || any(ACStatsW_logwhichcombos(:) & ACStatsW_levelcombos(:))
                error('TargetMoments.AgeConditionalStats: some targets ask for both a statistic and its log (fnname... and fnname..._log) at the same age and restriction, but LifeCycleProfiles computes one or the other for each (simoptions.logwhichcombos): target only one of them at each age')
            end
        else
            ACStats_whichstatsArr=[];
            ACStats_whichcombos=[];
        end
    else
        acsmomentnames=cell(1,3);
        acscummomentsizes=0;
        ACStats_whichstats=zeros(7,1);
        FnsToEvaluate_ACStats=struct();
        ACStats_whichstatsArr=[];
        ACStats_whichcombos=[];
        ACStats_ntimeperiods=1;
        ACStats_ntimeperiodstype='average';
        ACStats_split=0;
        ACStats_haswindowlog=0;
    end
    %% AutoCorrTransProbs (EvalFnOnAgentDist_AutoCorrTransProbs_FHorz): targets .(fn).(stat) and .(restriction).(fn).(stat); the stat is
    % Mean or StdDeviation (1 x N_j), or AutoCorrelation/AutoCovariance/PairMass/PairMean_j/PairMean_jplusk/PairStdDeviation_j/PairStdDeviation_jplusk
    % at horizon one (no suffix, 1 x N_j-1) or at horizon K (suffix _horizonK, 1 x N_j-K). TransitionProbs cannot be targeted (use CustomModelStats).
    % The selector is by start age: entry j of a horizon-K target selects start age j (Mean/StdDeviation: age j). The horizons (1 for a
    % target without a suffix) are read off the names and returned in autocorrtimehorizons for simoptions.timehorizons.
    if usingautocorr==1
        autocorrmomentnames=cell(0,3);
        autocorrmomentsizes=[];
        a1vec=fieldnames(TargetMoments.AutoCorrTransProbs);
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AutoCorrTransProbs.(a1vec{a1}));
            for a2=1:length(a2vec)
                temp=TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2});
                if isstruct(temp)
                    a3vec=fieldnames(temp);
                    for a3=1:length(a3vec)
                        if isstruct(temp.(a3vec{a3}))
                            error(['TargetMoments.AutoCorrTransProbs.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},' is a structure: a target has at most three levels (restriction.fn.stat)'])
                        end
                        targetmomentvec=[targetmomentvec; reshape(temp.(a3vec{a3}),[],1)];
                        autocorrmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3}};
                        autocorrmomentsizes(end+1)=numel(temp.(a3vec{a3}));
                    end
                else
                    targetmomentvec=[targetmomentvec; temp(:)];
                    autocorrmomentnames(end+1,:)={a1vec{a1},a2vec{a2},''};
                    autocorrmomentsizes(end+1)=numel(temp);
                end
            end
        end
        autocorrcummomentsizes=cumsum(autocorrmomentsizes);
        nrows=size(autocorrmomentnames,1);
        rowfn=cell(nrows,1);
        rowstat=cell(nrows,1);
        rowpage=zeros(nrows,1);
        % The output rows of the command: a FnsToEvaluate name, or (simoptions.ntimeperiods, simoptions.logwhichcombos) fnname_log,
        % fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log: rowbase is the FnsToEvaluate,
        % rowM the number of periods (1 for a single period), rowtype 'average' or 'sum' (M>1), rowlog 1 for the log
        rowbase=cell(nrows,1);
        rowM=ones(nrows,1);
        rowtype=cell(nrows,1);
        rowlog=zeros(nrows,1);
        for cc=1:nrows
            a1=autocorrmomentnames{cc,1}; a2=autocorrmomentnames{cc,2}; a3=autocorrmomentnames{cc,3};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
            if ~isfn % a window or log row of a FnsToEvaluate
                tk=regexp(a1,'^(.+)_(\d+)period(Average|Sum)(_log)?$','tokens','once');
                if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                    isfn=true;
                else
                    tk=regexp(a1,'^(.+)_log$','tokens','once');
                    isfn=(~isempty(tk) && any(strcmp(FnNamesAll,tk{1})));
                end
            end
            if isrestr && isfn
                error(['TargetMoments.AutoCorrTransProbs.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
            elseif isrestr
                if isempty(a3)
                    error(['TargetMoments.AutoCorrTransProbs.',a1,'.',a2,': a target under a conditional restriction needs a function and then a statistic'])
                end
                rowpage(cc)=1+find(strcmp(RestrNames,a1));
                rowfn{cc}=a2;
                rowstat{cc}=a3;
            elseif isfn
                if ~isempty(a3)
                    error(['TargetMoments.AutoCorrTransProbs.',a1,'.',a2,'.',a3,': a target is .(fn).(stat) or .(restriction).(fn).(stat)'])
                end
                rowpage(cc)=1;
                rowfn{cc}=a1;
                rowstat{cc}=a2;
            else
                error(['TargetMoments.AutoCorrTransProbs.',a1,' is neither a FnsToEvaluate (or one of its rows fnname_log, fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log) nor a conditional restriction in simoptions.conditionalrestrictions'])
            end
            rowbase{cc}=rowfn{cc};
            if ~any(strcmp(FnNamesAll,rowfn{cc}))
                tk=regexp(rowfn{cc},'^(.+)_(\d+)period(Average|Sum)(_log)?$','tokens','once');
                if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                    rowbase{cc}=tk{1};
                    rowM(cc)=str2double(tk{2});
                    rowtype{cc}=lower(tk{3});
                    rowlog(cc)=~isempty(tk{4});
                    if rowM(cc)<2
                        error(['TargetMoments.AutoCorrTransProbs: ',rowfn{cc},' names a window of ',tk{2},' periods: a single period is the FnsToEvaluate name itself (',tk{1},', or ',tk{1},'_log)'])
                    end
                else
                    tk=regexp(rowfn{cc},'^(.+)_log$','tokens','once');
                    if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                        rowbase{cc}=tk{1};
                        rowlog(cc)=1;
                    else
                        error(['TargetMoments.AutoCorrTransProbs: ',rowfn{cc},' is not one of the FnsToEvaluate (nor fnname_log, fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log of one)'])
                    end
                end
            end
        end
        FnsToEvaluate_AutoCorr=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowbase,FnNamesAll{ff}))
                FnsToEvaluate_AutoCorr.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesAC=fieldnames(FnsToEvaluate_AutoCorr);
        % simoptions.ntimeperiods (the window lengths the targets name, 1 for a single period) and simoptions.ntimeperiodstype
        % (one per command call, so the targets cannot mix MperiodAverage and MperiodSum)
        AutoCorr_ntimeperiods=unique(rowM(:)');
        wtypes=unique(rowtype(rowM>1));
        if length(wtypes)>1
            error('TargetMoments.AutoCorrTransProbs: the targets name both MperiodAverage and MperiodSum rows, but the command computes one or the other in a call (simoptions.ntimeperiodstype): use one of them for all the targets')
        elseif isscalar(wtypes)
            AutoCorr_ntimeperiodstype=wtypes{1};
        else
            AutoCorr_ntimeperiodstype='average';
        end
        nblocksAC=length(AutoCorr_ntimeperiods);
        autocorrtimehorizons=[];
        if buildselectors==1
            % rows in blocks of the window lengths (as simoptions.whichcombos with simoptions.ntimeperiods), and the log combinations
            AutoCorr_whichcombos=zeros(nblocksAC*length(FnNamesAC),N_j,1+nRestr);
            AutoCorr_logwhichcombos=zeros(nblocksAC*length(FnNamesAC),N_j,1+nRestr);
            AutoCorr_levelcombos=zeros(nblocksAC*length(FnNamesAC),N_j,1+nRestr);
        else
            AutoCorr_whichcombos=[];
            AutoCorr_logwhichcombos=0;
        end
        for cc=1:nrows
            stat=rowstat{cc};
            tk=regexp(stat,'_horizon(\d+)$','tokens','once'); % the horizon suffix
            if ~isempty(tk)
                K=str2double(tk{1});
                base=stat(1:end-length(tk{1})-8); % strip _horizonK
            else
                K=1;
                base=stat;
            end
            if any(strcmp(stat,{'Mean','StdDeviation'}))
                K=0;
            elseif ~any(strcmp(base,{'AutoCorrelation','AutoCovariance','PairMass','PairMean_j','PairMean_jplusk','PairStdDeviation_j','PairStdDeviation_jplusk'}))
                error(['TargetMoments.AutoCorrTransProbs: ',stat,' is not a targetable output (Mean, StdDeviation, AutoCorrelation, AutoCovariance and the Pair byproducts, each with an optional _horizonK suffix for horizon K>1; TransitionProbs cannot be targeted, use CustomModelStats)'])
            end
            if K>=1
                autocorrtimehorizons=[autocorrtimehorizons,K]; % the command computes exactly the horizons asked for (horizon 1 too)
            end
            if buildselectors==1
                if isempty(autocorrmomentnames{cc,3})
                    temp=TargetMoments.AutoCorrTransProbs.(autocorrmomentnames{cc,1}).(autocorrmomentnames{cc,2});
                else
                    temp=TargetMoments.AutoCorrTransProbs.(autocorrmomentnames{cc,1}).(autocorrmomentnames{cc,2}).(autocorrmomentnames{cc,3});
                end
                if ~(isvector(temp) && numel(temp)==N_j-K)
                    error(['TargetMoments.AutoCorrTransProbs target ',rowfn{cc},' ',stat,' has ',num2str(numel(temp)),' entries, but it is 1 x ',num2str(N_j-K),' (N_j minus the horizon)'])
                end
                ff=(find(AutoCorr_ntimeperiods==rowM(cc))-1)*length(FnNamesAC)+find(strcmp(FnNamesAC,rowbase{cc})); % the row: the block of its window length
                pp=rowpage(cc);
                sel=double(reshape(~isnan(temp),1,[]));
                AutoCorr_whichcombos(ff,1:N_j-K,pp)=max(AutoCorr_whichcombos(ff,1:N_j-K,pp),sel); % entry j selects start age j
                if rowlog(cc)==1
                    AutoCorr_logwhichcombos(ff,1:N_j-K,pp)=max(AutoCorr_logwhichcombos(ff,1:N_j-K,pp),sel);
                else
                    AutoCorr_levelcombos(ff,1:N_j-K,pp)=max(AutoCorr_levelcombos(ff,1:N_j-K,pp),sel);
                end
            end
        end
        if buildselectors==1 && any(AutoCorr_logwhichcombos(:) & AutoCorr_levelcombos(:))
            error('TargetMoments.AutoCorrTransProbs: some targets ask for both a statistic and its log (fnname... and fnname..._log) at the same start age and restriction, but the command computes one or the other for each (simoptions.logwhichcombos): target only one of them at each start age')
        end
        autocorrtimehorizons=unique(autocorrtimehorizons);
    else
        autocorrmomentnames=cell(1,3);
        autocorrcummomentsizes=0;
        FnsToEvaluate_AutoCorr=struct();
        autocorrtimehorizons=[];
        AutoCorr_whichcombos=[];
        AutoCorr_ntimeperiods=1;
        AutoCorr_ntimeperiodstype='average';
        AutoCorr_logwhichcombos=0;
    end
    %% CrossSectionCovarCorr (EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz) and AgeConditionalCrossSectionCovarCorr
    % (EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz): targets .(fn1).CovarianceWith.(fn2), .(fn1).CorrelationWith.(fn2),
    % .(fn).Mean, .(fn).StdDeviation, or the matrices .CovarianceMatrix / .CorrelationMatrix; each also under a conditional restriction as
    % .(restriction).(fn1).CovarianceWith.(fn2), .(restriction).(fn).Mean, .(restriction).CovarianceMatrix etc. (the commands compute the
    % restricted stats since 2026-10-07, output under .(restriction)). The self-correlation .(fn).(fn) is always one and cannot be targeted.
    % Scalars and [nFns,nFns] for the plain command; 1 x (number of age groups) and [nFns,nFns,number of age groups] for the age-conditional
    % one, the age axis as the command lays it out (simoptions.agegroupings, else N_j). NaN omits an entry (a pair, an age group, a matrix
    % entry). The selector is pair-shaped per page, [nFns,nFns,1+nRestr] and [nFns,nFns,ngroups,1+nRestr] (page 1 unrestricted, then the
    % restrictions in the fieldnames order of simoptions.conditionalrestrictions), symmetric, the diagonal being the own stats. A matrix
    % target is indexed over every FnsToEvaluate, so it makes the reduced set the full one.
    for xx=1:2
        if xx==1
            xf='CrossSectionCovarCorr'; usingx=usingcrosssec; nage=1;
        else
            xf='AgeConditionalCrossSectionCovarCorr'; usingx=usingagecrosssec;
            if buildselectors==1
                nage=ngroups;
            else
                nage=1;
            end
        end
        if usingx==1
            xnames=cell(0,4);
            xsizes=[];
            a1vec=fieldnames(TargetMoments.(xf));
            for a1=1:length(a1vec)
                temp1=TargetMoments.(xf).(a1vec{a1});
                if ~isstruct(temp1) % one name: a matrix
                    targetmomentvec=[targetmomentvec; temp1(:)];
                    xnames(end+1,:)={a1vec{a1},'','',''};
                    xsizes(end+1)=numel(temp1);
                else
                    a2vec=fieldnames(temp1);
                    for a2=1:length(a2vec)
                        temp2=temp1.(a2vec{a2});
                        if ~isstruct(temp2)
                            targetmomentvec=[targetmomentvec; temp2(:)];
                            xnames(end+1,:)={a1vec{a1},a2vec{a2},'',''};
                            xsizes(end+1)=numel(temp2);
                        else
                            a3vec=fieldnames(temp2);
                            for a3=1:length(a3vec)
                                temp3=temp2.(a3vec{a3});
                                if ~isstruct(temp3)
                                    targetmomentvec=[targetmomentvec; temp3(:)];
                                    xnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},''};
                                    xsizes(end+1)=numel(temp3);
                                else
                                    a4vec=fieldnames(temp3);
                                    for a4=1:length(a4vec)
                                        temp4=temp3.(a4vec{a4});
                                        if isstruct(temp4)
                                            error(['TargetMoments.',xf,'.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},'.',a4vec{a4},' is a structure: a target has at most four levels (restriction.fn1.CovarianceWith.fn2)'])
                                        end
                                        targetmomentvec=[targetmomentvec; temp4(:)];
                                        xnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4}};
                                        xsizes(end+1)=numel(temp4);
                                    end
                                end
                            end
                        end
                    end
                end
            end
            xcumsizes=cumsum(xsizes);
            nrows=size(xnames,1);
            rowfn1=cell(nrows,1);
            rowfn2=cell(nrows,1);
            rowpage=zeros(nrows,1);
            rowmatrix=zeros(nrows,1);
            for cc=1:nrows
                names=xnames(cc,:);
                names=names(~cellfun(@isempty,names));
                a1=names{1};
                isrestr=any(strcmp(RestrNames,a1));
                isfn=any(strcmp(FnNamesAll,a1));
                if isrestr && isfn
                    error(['TargetMoments.',xf,'.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
                elseif isrestr
                    rowpage(cc)=1+find(strcmp(RestrNames,a1));
                    rest=names(2:end);
                    if isempty(rest)
                        error(['TargetMoments.',xf,'.',a1,': a target under a conditional restriction needs a function (then Mean, StdDeviation, CovarianceWith.(fn2) or CorrelationWith.(fn2)) or CovarianceMatrix/CorrelationMatrix'])
                    end
                else
                    rowpage(cc)=1;
                    rest=names;
                end
                b1=rest{1};
                if isscalar(rest) % a matrix
                    if ~any(strcmp(b1,{'CovarianceMatrix','CorrelationMatrix'}))
                        error(['TargetMoments.',xf,'.',strjoin(names,'.'),': a target with no statistic must be CovarianceMatrix or CorrelationMatrix'])
                    end
                    rowmatrix(cc)=1;
                elseif ~any(strcmp(FnNamesAll,b1))
                    error(['TargetMoments.',xf,'.',strjoin(names,'.'),': ',b1,' is not one of the FnsToEvaluate (nor a conditional restriction in simoptions.conditionalrestrictions, nor CovarianceMatrix/CorrelationMatrix)'])
                elseif any(strcmp(rest{2},{'CovarianceWith','CorrelationWith'}))
                    if numel(rest)~=3 || ~any(strcmp(FnNamesAll,rest{3}))
                        error(['TargetMoments.',xf,'.',strjoin(names,'.'),': ',rest{2},' needs a second FnsToEvaluate name'])
                    end
                    rowfn1{cc}=b1;
                    rowfn2{cc}=rest{3};
                elseif numel(rest)==2 && any(strcmp(rest{2},{'Mean','StdDeviation'}))
                    rowfn1{cc}=b1;
                    rowfn2{cc}=b1;
                elseif numel(rest)==2 && strcmp(rest{2},b1)
                    error(['TargetMoments.',xf,'.',strjoin(names,'.'),': the self-correlation is always one and cannot be targeted'])
                else
                    error(['TargetMoments.',xf,'.',strjoin(names,'.'),': the targets are Mean, StdDeviation, CovarianceWith.(fn2), CorrelationWith.(fn2), or CovarianceMatrix/CorrelationMatrix, each optionally under a conditional restriction'])
                end
            end
            FnsX=struct();
            for ff=1:length(FnNamesAll)
                if any(rowmatrix==1) || any(strcmp(rowfn1,FnNamesAll{ff})) || any(strcmp(rowfn2,FnNamesAll{ff}))
                    FnsX.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
                end
            end
            FnNamesX=fieldnames(FnsX);
            if buildselectors==1
                wcX=zeros(length(FnNamesX),length(FnNamesX),nage,1+nRestr);
                for cc=1:nrows
                    temp=TargetMoments.(xf);
                    for kk=1:size(xnames,2)
                        if ~isempty(xnames{cc,kk})
                            temp=temp.(xnames{cc,kk});
                        end
                    end
                    pp=rowpage(cc);
                    if rowmatrix(cc)==1 % a matrix: non-NaN entries select the pairs
                        if ~isequal(size(temp,1:3),[length(FnNamesX),length(FnNamesX),nage])
                            error(['TargetMoments.',xf,' target ',strjoin(xnames(cc,~cellfun(@isempty,xnames(cc,:))),'.'),' must be of size [',num2str(length(FnNamesX)),',',num2str(length(FnNamesX)),',',num2str(nage),'] (number of FnsToEvaluate, twice, number of age groups), NaN where not wanted'])
                        end
                        on=double(~isnan(temp));
                        wcX(:,:,:,pp)=max(wcX(:,:,:,pp),max(on,permute(on,[2,1,3]))); % symmetric: the commands read the upper triangle, a lower-triangle entry alone must still select the pair
                    else
                        if ~(isvector(temp) && numel(temp)==nage)
                            error(['TargetMoments.',xf,' target ',strjoin(xnames(cc,~cellfun(@isempty,xnames(cc,:))),'.'),' has ',num2str(numel(temp)),' entries, but there are ',num2str(nage),' age groups'])
                        end
                        f1=find(strcmp(FnNamesX,rowfn1{cc}));
                        f2=find(strcmp(FnNamesX,rowfn2{cc}));
                        on=double(reshape(~isnan(temp),[1,1,nage]));
                        wcX(f1,f2,:,pp)=max(wcX(f1,f2,:,pp),on);
                        wcX(f2,f1,:,pp)=max(wcX(f2,f1,:,pp),on);
                    end
                end
                if xx==1
                    wcX=reshape(wcX,[length(FnNamesX),length(FnNamesX),1+nRestr]); % (nage is one)
                end
            else
                wcX=[];
            end
        else
            xnames=cell(1,4);
            xcumsizes=0;
            FnsX=struct();
            wcX=[];
        end
        if xx==1
            crosssecmomentnames=xnames; crossseccummomentsizes=xcumsizes; FnsToEvaluate_CrossSec=FnsX; CrossSec_whichcombos=wcX;
        else
            agecrosssecmomentnames=xnames; agecrossseccummomentsizes=xcumsizes; FnsToEvaluate_AgeCrossSec=FnsX; AgeCrossSec_whichcombos=wcX;
        end
    end
    %% A target that is NaN in full is dropped, with a warning (NaN is for omitting some entries of a target, ages or matrix entries; a
    % target omitted in full should not be listed, and with caliboptions.whichcombos=1 nothing is computed for it)
    nanchk_kinds={'AllStats','AgeConditionalStats','AutoCorrTransProbs','CrossSectionCovarCorr','AgeConditionalCrossSectionCovarCorr'};
    nanchk_using=[usingallstats,usinglcp,usingautocorr,usingcrosssec,usingagecrosssec];
    nanchk_sofar=0;
    for nanchk_kk=1:5
        if nanchk_using(nanchk_kk)==1
            if nanchk_kk==1
                nanchk_names=allstatmomentnames; nanchk_cum=allstatcummomentsizes;
            elseif nanchk_kk==2
                nanchk_names=acsmomentnames; nanchk_cum=acscummomentsizes;
            elseif nanchk_kk==3
                nanchk_names=autocorrmomentnames; nanchk_cum=autocorrcummomentsizes;
            elseif nanchk_kk==4
                nanchk_names=crosssecmomentnames; nanchk_cum=crossseccummomentsizes;
            else
                nanchk_names=agecrosssecmomentnames; nanchk_cum=agecrossseccummomentsizes;
            end
            for cc=1:size(nanchk_names,1)
                if cc==1
                    nanchk_idx=nanchk_sofar+1:nanchk_sofar+nanchk_cum(1);
                else
                    nanchk_idx=nanchk_sofar+nanchk_cum(cc-1)+1:nanchk_sofar+nanchk_cum(cc);
                end
                if all(isnan(targetmomentvec(nanchk_idx)))
                    warning(['TargetMoments.',nanchk_kinds{nanchk_kk},'.',strjoin(nanchk_names(cc,~cellfun(@isempty,nanchk_names(cc,:))),'.'),' is NaN in full: a target that is omitted in full should not be listed (NaN is for omitting some of its entries); it is dropped'])
                end
            end
            nanchk_sofar=nanchk_sofar+nanchk_cum(end);
        end
    end
    %% The selectors for simoptions.whichcombos/whichstats (used by CalibrateLifeCycleModel when caliboptions.whichcombos=1)
    selectors=struct();
    if buildselectors==1
        selectors.AllStats.whichcombos=AllStats_whichcombos;
        selectors.AllStats.whichstats=AllStats_whichstatsArr;
        selectors.ACStats.whichcombos=ACStats_whichcombos;
        selectors.ACStats.whichstats=ACStats_whichstatsArr;
        % The window and log rows of the AgeConditionalStats targets (only when some target names one): the rows of whichcombos/whichstats
        % are in blocks of ntimeperiods; agegroupings is what the call must use ([]: simoptions' own); with split=1 the window targets
        % are in a second call, ACStatsW (by start age), as the single-period ones are by age groups
        selectors.ACStats.haswindowlog=ACStats_haswindowlog;
        if ACStats_haswindowlog==1
            selectors.ACStats.ntimeperiods=ACStats_ntimeperiods;
            selectors.ACStats.ntimeperiodstype=ACStats_ntimeperiodstype;
            selectors.ACStats.logwhichcombos=ACStats_logwhichcombos;
            if any(ACStats_ntimeperiods>1) && ~agedefault
                selectors.ACStats.agegroupings=1:1:N_j;
            else
                selectors.ACStats.agegroupings=[];
            end
            selectors.ACStats.split=ACStats_split;
            if ACStats_split==1
                selectors.ACStatsW.whichcombos=ACStatsW_whichcombos;
                selectors.ACStatsW.whichstats=ACStatsW_whichstatsArr;
                selectors.ACStatsW.ntimeperiods=ACStatsW_ntimeperiods;
                selectors.ACStatsW.ntimeperiodstype=ACStats_ntimeperiodstype;
                selectors.ACStatsW.logwhichcombos=ACStatsW_logwhichcombos;
                selectors.ACStatsW.agegroupings=1:1:N_j;
            end
        end
        selectors.AutoCorr.whichcombos=AutoCorr_whichcombos;
        selectors.AutoCorr.ntimeperiods=AutoCorr_ntimeperiods; % the window lengths the targets name (simoptions.ntimeperiods; the rows of whichcombos are in blocks of these)
        selectors.AutoCorr.ntimeperiodstype=AutoCorr_ntimeperiodstype; % simoptions.ntimeperiodstype
        selectors.AutoCorr.logwhichcombos=AutoCorr_logwhichcombos; % simoptions.logwhichcombos (the targets named ..._log)
        selectors.CrossSec.whichcombos=CrossSec_whichcombos;
        selectors.AgeCrossSec.whichcombos=AgeCrossSec_whichcombos;
    end
elseif useptype==1
    % With permanent types a target has two to four names: fn.stat (grouped), fn.type.stat, restriction.fn.stat (grouped),
    % restriction.fn.type.stat, and fn.MoreInequality.substat (or fn.type.MoreInequality.substat). The selectors carry a trailing
    % type dimension of N_i+1: one slot per permanent type in the order of Names_i, then the grouped stats (see the PType stats
    % commands). A grouped target selects the grouped slot; the commands then compute every ptype's stats for that combination too,
    % as the grouped Mean, StdDeviation, Minimum and Maximum are built from them.
    if ~exist('Names_i','var')
        Names_i={};
    end
    N_i=length(Names_i);
    if exist('simoptions','var') && isfield(simoptions,'conditionalrestrictions')
        RestrNames=fieldnames(simoptions.conditionalrestrictions);
    else
        RestrNames={};
    end
    nRestr=length(RestrNames);
    FnNamesAll=fieldnames(FnsToEvaluate);
    knownstats={'Mean','Median','RatioMeanToMedian','Variance','StdDeviation','Gini','LorenzCurve','Minimum','Maximum','QuantileCutoffs','QuantileMeans','MoreInequality'};
    if exist('simoptions','var') && exist('N_j','var') && N_i>0
        buildselectors=1;
        % The age axis of LifeCycleProfiles_FHorz_Case1_PType: the largest number of age groups over the ptypes (per-type agegroupings
        % or per-type N_j), each ptype's age groups sitting at positions shifted by its agejshifter (relative to the smallest)
        ngroups_i=zeros(N_i,1);
        for ii=1:N_i
            if isfield(simoptions,'agegroupings')
                if isstruct(simoptions.agegroupings)
                    ngroups_i(ii)=length(simoptions.agegroupings.(Names_i{ii}));
                else
                    ngroups_i(ii)=length(simoptions.agegroupings);
                end
            elseif isstruct(N_j)
                ngroups_i(ii)=N_j.(Names_i{ii});
            else
                ngroups_i(ii)=N_j;
            end
        end
        maxngroups=max(ngroups_i);
        ageshift=zeros(N_i,1);
        if isfield(simoptions,'agejshifter')
            if isstruct(simoptions.agejshifter)
                for ii=1:N_i
                    ageshift(ii)=simoptions.agejshifter.(Names_i{ii});
                end
            elseif ~isscalar(simoptions.agejshifter)
                ageshift=reshape(simoptions.agejshifter,[N_i,1]);
            end
            ageshift=ageshift-min(ageshift);
        end
    else
        buildselectors=0;
    end
    targetmomentvec=[]; % Can't preallocate as have no idea how big this will be
    %% AllStats
    if usingallstats==1
        allstatmomentnames=cell(0,4);
        allstatmomentsizes=[];
        a1vec=fieldnames(TargetMoments.AllStats);
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}));
            for a2=1:length(a2vec)
                temp2=TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2});
                if isstruct(temp2)
                    a3vec=fieldnames(temp2);
                    for a3=1:length(a3vec)
                        temp3=temp2.(a3vec{a3});
                        if isstruct(temp3)
                            a4vec=fieldnames(temp3);
                            for a4=1:length(a4vec)
                                temp4=temp3.(a4vec{a4});
                                if isstruct(temp4)
                                    error(['TargetMoments.AllStats.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},'.',a4vec{a4},' is a structure: a target has at most four levels (restriction.fn.type.stat)'])
                                end
                                targetmomentvec=[targetmomentvec; temp4(:)]; % append to end
                                allstatmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4}};
                                allstatmomentsizes(end+1)=numel(temp4);
                            end
                        else
                            targetmomentvec=[targetmomentvec; temp3(:)]; % append to end
                            allstatmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},''};
                            allstatmomentsizes(end+1)=numel(temp3);
                        end
                    end
                else
                    targetmomentvec=[targetmomentvec; temp2(:)]; % append to end
                    allstatmomentnames(end+1,:)={a1vec{a1},a2vec{a2},'',''};
                    allstatmomentsizes(end+1)=numel(temp2);
                end
            end
        end
        allstatcummomentsizes=cumsum(allstatmomentsizes);
        % Classify each row as (fn, stat, page, slot): page 1 unrestricted or 1+rr; slot ii a ptype, N_i+1 the grouped stats
        nrows=size(allstatmomentnames,1);
        rowfn=cell(nrows,1);
        rowstat=cell(nrows,1);
        rowpage=zeros(nrows,1);
        rowslot=zeros(nrows,1);
        for cc=1:nrows
            names=allstatmomentnames(cc,:);
            names=names(~cellfun(@isempty,names));
            a1=names{1};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
            if isrestr && isfn
                error(['TargetMoments.AllStats.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
            elseif isrestr || (~isfn && ~exist('simoptions','var')) % (without simoptions a first-level name that is not a function is taken to be a restriction, as the estimation commands rely on)
                if isrestr
                    rowpage(cc)=1+find(strcmp(RestrNames,a1));
                else
                    rowpage(cc)=0;
                end
                if length(names)<3
                    error(['TargetMoments.AllStats.',a1,'.',names{2},': a target under a conditional restriction needs a function and then a statistic'])
                end
                rowfn{cc}=names{2};
                rest=names(3:end);
            elseif isfn
                rowpage(cc)=1;
                rowfn{cc}=a1;
                rest=names(2:end);
            else
                error(['TargetMoments.AllStats.',a1,' is neither a FnsToEvaluate nor a conditional restriction in simoptions.conditionalrestrictions'])
            end
            if ~any(strcmp(FnNamesAll,rowfn{cc}))
                error(['TargetMoments.AllStats: ',rowfn{cc},' is not one of the FnsToEvaluate'])
            end
            if isempty(rest)
                error(['TargetMoments.AllStats.',a1,': a target needs a statistic (e.g. .Mean)'])
            end
            if N_i>0 && any(strcmp(Names_i,rest{1})) % a permanent type: its own stats
                rowslot(cc)=find(strcmp(Names_i,rest{1}));
                if length(rest)<2
                    error(['TargetMoments.AllStats: the target for ',rowfn{cc},' of type ',rest{1},' needs a statistic (e.g. .Mean)'])
                end
                rowstat{cc}=rest{2}; % (a third element is a MoreInequality sub-statistic)
            elseif N_i>0 % the grouped stats
                rowslot(cc)=N_i+1;
                rowstat{cc}=rest{1}; % (a second element is a MoreInequality sub-statistic)
            else % without Names_i (the estimation commands' three-input call) the type level cannot be told apart: take the first known statistic name
                rowslot(cc)=0;
                rowstat{cc}='';
                for kk=1:length(rest)
                    if any(strcmp(knownstats,rest{kk}))
                        rowstat{cc}=rest{kk};
                        break
                    end
                end
            end
            if any(strcmp(knownstats,rowfn{cc})) || (rowslot(cc)>0 && rowslot(cc)<=N_i && any(strcmp(knownstats,Names_i{rowslot(cc)})))
                error('TargetMoments.AllStats: a FnsToEvaluate or permanent type is named like a statistic, so the target cannot be read')
            end
        end
        FnsToEvaluate_AllStats=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowfn,FnNamesAll{ff}))
                FnsToEvaluate_AllStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesA=fieldnames(FnsToEvaluate_AllStats);
        AllStats_whichstats=zeros(7,1);
        if buildselectors==1
            AllStats_whichstatsArr=zeros(length(FnNamesA),1+nRestr,N_i+1,7);
        end
        for cc=1:nrows
            ff=find(strcmp(FnNamesA,rowfn{cc}));
            pp=rowpage(cc);
            ss_slot=rowslot(cc);
            temp=TargetMoments.AllStats; % the target's values: a target that is NaN in full selects nothing (it is dropped, with a warning below)
            for kk=1:size(allstatmomentnames,2)
                if ~isempty(allstatmomentnames{cc,kk})
                    temp=temp.(allstatmomentnames{cc,kk});
                end
            end
            rowallnan=all(isnan(temp(:)));
            switch rowstat{cc} % see StatsFromWeightedGrid: 1 Mean, 2 Median, 3 Variance/StdDeviation, 4 Gini(3)/LorenzCurve(1), 5 Minimum/Maximum, 6 Quantiles, 7 MoreInequality
                case 'Mean'
                    AllStats_whichstats(1)=1; sidx=1;
                case 'Median'
                    AllStats_whichstats(2)=1; sidx=2;
                case 'RatioMeanToMedian'
                    AllStats_whichstats(1:2)=1; sidx=[1,2];
                case {'Variance','StdDeviation'}
                    AllStats_whichstats(3)=1; sidx=3;
                case 'Gini'
                    if AllStats_whichstats(4)==0
                        AllStats_whichstats(4)=3;
                    end
                    sidx=4;
                case 'LorenzCurve'
                    AllStats_whichstats(4)=1; sidx=4;
                case {'Minimum','Maximum'}
                    AllStats_whichstats(5)=1; sidx=5;
                case {'QuantileCutoffs','QuantileMeans'}
                    AllStats_whichstats(6)=1; sidx=6;
                case 'MoreInequality'
                    AllStats_whichstats(7)=1; sidx=7;
                otherwise
                    error(['TargetMoments.AllStats: ',rowstat{cc},' is not a statistic that AllStats produces (Mean, Median, RatioMeanToMedian, Variance, StdDeviation, Gini, LorenzCurve, Minimum, Maximum, QuantileCutoffs, QuantileMeans, MoreInequality)'])
            end
            if buildselectors==1 && rowallnan==0
                for ss=sidx
                    if ss==4 % Gini/LorenzCurve: 3 is Gini only, 1 is the Lorenz curve (and Gini); a LorenzCurve target overrides a Gini-only 3
                        if strcmp(rowstat{cc},'LorenzCurve')
                            AllStats_whichstatsArr(ff,pp,ss_slot,4)=1;
                        elseif AllStats_whichstatsArr(ff,pp,ss_slot,4)==0
                            AllStats_whichstatsArr(ff,pp,ss_slot,4)=3;
                        end
                    else
                        AllStats_whichstatsArr(ff,pp,ss_slot,ss)=1;
                    end
                end
            end
        end
        if buildselectors==1
            AllStats_whichcombos=double(any(AllStats_whichstatsArr,4)); % [nFns,1+nRestr,N_i+1]
        else
            AllStats_whichstatsArr=[];
            AllStats_whichcombos=[];
        end
    else
        allstatmomentnames=cell(1,4);
        allstatcummomentsizes=0;
        AllStats_whichstats=zeros(7,1);
        FnsToEvaluate_AllStats=struct();
        AllStats_whichstatsArr=[];
        AllStats_whichcombos=[];
    end
    %% AgeConditionalStats (LifeCycleProfiles)
    if usinglcp==1
        acsmomentnames=cell(0,4);
        acsmomentsizes=[];
        a1vec=fieldnames(TargetMoments.AgeConditionalStats);
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1}));
            for a2=1:length(a2vec)
                temp2=TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2});
                if isstruct(temp2)
                    a3vec=fieldnames(temp2);
                    for a3=1:length(a3vec)
                        temp3=temp2.(a3vec{a3});
                        if isstruct(temp3)
                            a4vec=fieldnames(temp3);
                            for a4=1:length(a4vec)
                                temp4=temp3.(a4vec{a4});
                                if isstruct(temp4)
                                    error(['TargetMoments.AgeConditionalStats.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},'.',a4vec{a4},' is a structure: a target has at most four levels (restriction.fn.type.stat)'])
                                end
                                targetmomentvec=[targetmomentvec; temp4(:)]; % append to end
                                acsmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4}};
                                acsmomentsizes(end+1)=numel(temp4);
                            end
                        else
                            targetmomentvec=[targetmomentvec; temp3(:)]; % append to end
                            acsmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},''};
                            acsmomentsizes(end+1)=numel(temp3);
                        end
                    end
                else
                    targetmomentvec=[targetmomentvec; temp2(:)]; % append to end
                    acsmomentnames(end+1,:)={a1vec{a1},a2vec{a2},'',''};
                    acsmomentsizes(end+1)=numel(temp2);
                end
            end
        end
        acscummomentsizes=cumsum(acsmomentsizes);
        nrows=size(acsmomentnames,1);
        rowfn=cell(nrows,1);
        rowstat=cell(nrows,1);
        rowpage=zeros(nrows,1);
        rowslot=zeros(nrows,1);
        rowbase=cell(nrows,1); % the FnsToEvaluate of the row (fnname, or the fnname of fnname_log, fnname_MperiodAverage, ...)
        rowM=ones(nrows,1); % its number of periods (1 for a single period)
        rowtype=cell(nrows,1); % average or sum (M>1)
        rowlog=zeros(nrows,1); % 1 for the log
        for cc=1:nrows
            names=acsmomentnames(cc,:);
            names=names(~cellfun(@isempty,names));
            a1=names{1};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
            if ~isfn % a window or log row of a FnsToEvaluate (simoptions.ntimeperiods, simoptions.logwhichcombos)
                tk=regexp(a1,'^(.+)_(\d+)period(Average|Sum)(_log)?$','tokens','once');
                if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                    isfn=true;
                else
                    tk=regexp(a1,'^(.+)_log$','tokens','once');
                    isfn=(~isempty(tk) && any(strcmp(FnNamesAll,tk{1})));
                end
            end
            if isrestr && isfn
                error(['TargetMoments.AgeConditionalStats.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
            elseif isrestr || (~isfn && ~exist('simoptions','var'))
                if isrestr
                    rowpage(cc)=1+find(strcmp(RestrNames,a1));
                else
                    rowpage(cc)=0;
                end
                if length(names)<3
                    error(['TargetMoments.AgeConditionalStats.',a1,'.',names{2},': a target under a conditional restriction needs a function and then a statistic'])
                end
                rowfn{cc}=names{2};
                rest=names(3:end);
            elseif isfn
                rowpage(cc)=1;
                rowfn{cc}=a1;
                rest=names(2:end);
            else
                error(['TargetMoments.AgeConditionalStats.',a1,' is neither a FnsToEvaluate (or one of its rows fnname_log, fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log) nor a conditional restriction in simoptions.conditionalrestrictions'])
            end
            % the FnsToEvaluate of the row, its number of periods, and whether it is the log (as without permanent types)
            rowbase{cc}=rowfn{cc};
            if ~any(strcmp(FnNamesAll,rowfn{cc}))
                tk=regexp(rowfn{cc},'^(.+)_(\d+)period(Average|Sum)(_log)?$','tokens','once');
                if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                    rowbase{cc}=tk{1};
                    rowM(cc)=str2double(tk{2});
                    rowtype{cc}=lower(tk{3});
                    rowlog(cc)=~isempty(tk{4});
                    if rowM(cc)<2
                        error(['TargetMoments.AgeConditionalStats: ',rowfn{cc},' names a window of ',tk{2},' periods: a single period is the FnsToEvaluate name itself (',tk{1},', or ',tk{1},'_log)'])
                    end
                else
                    tk=regexp(rowfn{cc},'^(.+)_log$','tokens','once');
                    if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                        rowbase{cc}=tk{1};
                        rowlog(cc)=1;
                    else
                        error(['TargetMoments.AgeConditionalStats: ',rowfn{cc},' is not one of the FnsToEvaluate (nor fnname_log, fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log of one)'])
                    end
                end
            end
            if isempty(rest)
                error(['TargetMoments.AgeConditionalStats.',a1,': a target needs a statistic (e.g. .Mean)'])
            end
            if N_i>0 && any(strcmp(Names_i,rest{1}))
                rowslot(cc)=find(strcmp(Names_i,rest{1}));
                if length(rest)<2
                    error(['TargetMoments.AgeConditionalStats: the target for ',rowfn{cc},' of type ',rest{1},' needs a statistic (e.g. .Mean)'])
                end
                rowstat{cc}=rest{2};
            elseif N_i>0
                rowslot(cc)=N_i+1;
                rowstat{cc}=rest{1};
            else
                rowslot(cc)=0;
                rowstat{cc}='';
                for kk=1:length(rest)
                    if any(strcmp(knownstats,rest{kk}))
                        rowstat{cc}=rest{kk};
                        break
                    end
                end
            end
            if any(strcmp(knownstats,rowfn{cc})) || (rowslot(cc)>0 && rowslot(cc)<=N_i && any(strcmp(knownstats,Names_i{rowslot(cc)})))
                error('TargetMoments.AgeConditionalStats: a FnsToEvaluate or permanent type is named like a statistic, so the target cannot be read')
            end
        end
        FnsToEvaluate_ACStats=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowbase,FnNamesAll{ff}))
                FnsToEvaluate_ACStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesL=fieldnames(FnsToEvaluate_ACStats);
        % simoptions.ntimeperiodstype (one per call, so the targets cannot mix MperiodAverage and MperiodSum)
        wtypes=unique(rowtype(rowM>1));
        if length(wtypes)>1
            error('TargetMoments.AgeConditionalStats: the targets name both MperiodAverage and MperiodSum rows, but LifeCycleProfiles computes one or the other in a call (simoptions.ntimeperiodstype): use one of them for all the targets')
        elseif isscalar(wtypes)
            ACStats_ntimeperiodstype=wtypes{1};
        else
            ACStats_ntimeperiodstype='average';
        end
        % The calls of LifeCycleProfiles_FHorz_Case1_PType (as without permanent types): windows are by start age (agegroupings=1:1:N_j);
        % when the single-period targets are by age groups and there are window targets too, the windows get a second call (callW)
        if buildselectors==1
            if any(rowM>1)
                if isstruct(N_j) || any(ageshift~=0) || (isfield(simoptions,'agegroupings') && isstruct(simoptions.agegroupings))
                    error('TargetMoments.AgeConditionalStats: targets of multi-period sums/averages (fnname_MperiodAverage, ...) need the same N_j for every permanent type, no agejshifter, and agegroupings that do not differ by permanent type')
                end
            end
            agedefault=(~isfield(simoptions,'agegroupings') || (~isstruct(simoptions.agegroupings) && ~isstruct(N_j) && isequal(reshape(simoptions.agegroupings,1,[]),1:1:N_j)));
        else
            agedefault=1;
        end
        ACStats_split=(any(rowM>1) && any(rowM==1) && ~agedefault);
        ACStats_haswindowlog=(any(rowM>1) || any(rowlog==1)); % some target is of a window or a log (the objective then sets simoptions.ntimeperiods, ntimeperiodstype, logwhichcombos)
        rowcall=ones(nrows,1); % 1 the main call, 2 the window call
        if ACStats_split
            rowcall(rowM>1)=2;
            ACStats_ntimeperiods=1;
            ACStatsW_ntimeperiods=unique(rowM(rowM>1)');
            ngroupsA=maxngroups;
        else
            ACStats_ntimeperiods=unique(rowM(:)');
            ACStatsW_ntimeperiods=[];
            if any(rowM>1) && buildselectors==1 % (without the selectors, the legacy call of the cohort route, N_j is not given and no arrays are built)
                ngroupsA=N_j;
            elseif buildselectors==1
                ngroupsA=maxngroups;
            else
                ngroupsA=0;
            end
        end
        nFnsL=length(FnNamesL);
        ACStats_whichstats=zeros(7,1);
        if buildselectors==1
            % rows in blocks of the window lengths of the call; logwhichcombos has no type dimension (it applies to every ptype and the grouped stats)
            ACStats_whichstatsArr=zeros(length(ACStats_ntimeperiods)*nFnsL,ngroupsA,1+nRestr,N_i+1,7);
            ACStats_logwhichcombos=zeros(length(ACStats_ntimeperiods)*nFnsL,ngroupsA,1+nRestr);
            ACStats_levelcombos=zeros(length(ACStats_ntimeperiods)*nFnsL,ngroupsA,1+nRestr);
            ACStatsW_whichstatsArr=zeros(length(ACStatsW_ntimeperiods)*nFnsL,N_j,1+nRestr,N_i+1,7);
            ACStatsW_logwhichcombos=zeros(length(ACStatsW_ntimeperiods)*nFnsL,N_j,1+nRestr);
            ACStatsW_levelcombos=zeros(length(ACStatsW_ntimeperiods)*nFnsL,N_j,1+nRestr);
        end
        for cc=1:nrows
            if rowcall(cc)==1
                ff=(find(ACStats_ntimeperiods==rowM(cc))-1)*nFnsL+find(strcmp(FnNamesL,rowbase{cc})); % the row: the block of its window length
                ngc=ngroupsA;
            else
                ff=(find(ACStatsW_ntimeperiods==rowM(cc))-1)*nFnsL+find(strcmp(FnNamesL,rowbase{cc}));
                ngc=N_j;
            end
            pp=rowpage(cc);
            ss_slot=rowslot(cc);
            if buildselectors==1
                % the target entries of this row: a vector over this slot's age groups, or a matrix with them as columns; an age group is on where the target is not NaN
                names=acsmomentnames(cc,:);
                names=names(~cellfun(@isempty,names));
                temp=TargetMoments.AgeConditionalStats;
                for kk=1:length(names)
                    temp=temp.(names{kk});
                end
                if rowM(cc)>1 || ngc~=maxngroups % a window (or every row with only window targets): by start age
                    nslot=N_j;
                    shift=0;
                elseif ss_slot<=N_i
                    nslot=ngroups_i(ss_slot);
                    shift=ageshift(ss_slot);
                else
                    nslot=maxngroups;
                    shift=0;
                end
                if isvector(temp) && numel(temp)==nslot
                    ageon=reshape(~isnan(temp),[1,nslot]);
                elseif ismatrix(temp) && size(temp,2)==nslot
                    ageon=any(~isnan(temp),1);
                elseif rowM(cc)>1
                    error(['TargetMoments.AgeConditionalStats target for ',rowfn{cc},' ',rowstat{cc},' has ',num2str(numel(temp)),' entries, but it is by start age (1 x ',num2str(N_j),', or a matrix with N_j columns)'])
                else
                    error(['TargetMoments.AgeConditionalStats target for ',rowfn{cc},' ',rowstat{cc},' has ',num2str(numel(temp)),' entries, but there are ',num2str(nslot),' age groups for it (it must be a vector over the age groups, or a matrix with the age groups as columns)'])
                end
                agepos=(1:nslot)+shift; % this slot's age groups on the common age axis
                keep=(agepos<=ngc);
                ageonpos=false(1,ngc);
                ageonpos(agepos(keep))=ageon(keep);
                if rowM(cc)>1 && any(ageonpos(N_j-rowM(cc)+2:N_j))
                    error(['TargetMoments.AgeConditionalStats: ',rowfn{cc},' is targeted at a start age after N_j-',num2str(rowM(cc)-1),', where the ',num2str(rowM(cc)),'-period window runs past age N_j (these entries must be NaN)'])
                end
                if rowcall(cc)==1
                    if rowlog(cc)==1
                        ACStats_logwhichcombos(ff,ageonpos,pp)=1;
                    else
                        ACStats_levelcombos(ff,ageonpos,pp)=1;
                    end
                else
                    if rowlog(cc)==1
                        ACStatsW_logwhichcombos(ff,ageonpos,pp)=1;
                    else
                        ACStatsW_levelcombos(ff,ageonpos,pp)=1;
                    end
                end
            end
            switch rowstat{cc}
                case 'Mean'
                    ACStats_whichstats(1)=1; sidx=1;
                case 'Median'
                    ACStats_whichstats(2)=1; sidx=2;
                case 'RatioMeanToMedian'
                    ACStats_whichstats(1:2)=1; sidx=[1,2];
                case {'Variance','StdDeviation'}
                    ACStats_whichstats(3)=1; sidx=3;
                case 'Gini'
                    if ACStats_whichstats(4)==0
                        ACStats_whichstats(4)=3;
                    end
                    sidx=4;
                case 'LorenzCurve'
                    ACStats_whichstats(4)=2; sidx=4;
                case {'Minimum','Maximum'}
                    ACStats_whichstats(5)=1; sidx=5;
                case {'QuantileCutoffs','QuantileMeans'}
                    ACStats_whichstats(6)=1; sidx=6;
                case 'MoreInequality'
                    ACStats_whichstats(7)=1; sidx=7;
                otherwise
                    error(['TargetMoments.AgeConditionalStats: ',rowstat{cc},' is not a statistic that LifeCycleProfiles produces (Mean, Median, RatioMeanToMedian, Variance, StdDeviation, Gini, LorenzCurve, Minimum, Maximum, QuantileCutoffs, QuantileMeans, MoreInequality)'])
            end
            if buildselectors==1
                if rowcall(cc)==1
                    wsArr=ACStats_whichstatsArr;
                else
                    wsArr=ACStatsW_whichstatsArr;
                end
                for ss=sidx
                    if ss==4 % Gini/LorenzCurve: 3 is Gini only, 2 the Lorenz curve (and Gini); a LorenzCurve target overrides a Gini-only 3
                        if strcmp(rowstat{cc},'LorenzCurve')
                            wsArr(ff,ageonpos,pp,ss_slot,4)=2;
                        else
                            current=reshape(wsArr(ff,:,pp,ss_slot,4),[1,ngc]);
                            current(ageonpos & current==0)=3;
                            wsArr(ff,:,pp,ss_slot,4)=current;
                        end
                    else
                        wsArr(ff,ageonpos,pp,ss_slot,ss)=1;
                    end
                end
                if rowcall(cc)==1
                    ACStats_whichstatsArr=wsArr;
                else
                    ACStatsW_whichstatsArr=wsArr;
                end
            end
        end
        if buildselectors==1
            ACStats_whichcombos=double(any(ACStats_whichstatsArr,5)); % [nRows,maxngroups,1+nRestr,N_i+1]
            ACStatsW_whichcombos=double(any(ACStatsW_whichstatsArr,5));
            if any(ACStats_logwhichcombos(:) & ACStats_levelcombos(:)) || any(ACStatsW_logwhichcombos(:) & ACStatsW_levelcombos(:))
                error('TargetMoments.AgeConditionalStats: some targets ask for both a statistic and its log (fnname... and fnname..._log) at the same age and restriction (for any permanent type or the grouped stats), but LifeCycleProfiles computes one or the other for each (simoptions.logwhichcombos has no type dimension): target only one of them at each age')
            end
        else
            ACStats_whichstatsArr=[];
            ACStats_whichcombos=[];
        end
    else
        acsmomentnames=cell(1,4);
        acscummomentsizes=0;
        ACStats_whichstats=zeros(7,1);
        FnsToEvaluate_ACStats=struct();
        ACStats_whichstatsArr=[];
        ACStats_whichcombos=[];
        ACStats_ntimeperiods=1;
        ACStats_ntimeperiodstype='average';
        ACStats_split=0;
        ACStats_haswindowlog=0;
    end
    %% AutoCorrTransProbs (EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType): targets .(fn).(stat) and .(restriction).(fn).(stat) (the grouped
    % output) and .(fn).(type).(stat) and .(restriction).(fn).(type).(stat) (a permanent type's own output), nested as the command's output.
    % The stat is Mean or StdDeviation (1 x N_j), or AutoCorrelation/AutoCovariance/PairMass/PairMean_j/PairMean_jplusk/PairStdDeviation_j/
    % PairStdDeviation_jplusk at horizon one (no suffix, 1 x N_j-1) or at horizon K (suffix _horizonK, 1 x N_j-K). TransitionProbs cannot be
    % targeted (use CustomModelStats). The selector is [nFns,N_j,1+nRestr,N_i+1]: entry j of a horizon-K target selects start age j
    % (Mean/StdDeviation: age j) in the target's slot, a type's slot or the grouped slot N_i+1 (which makes the command compute every
    % type at that start age, as the grouped output is pooled from them). The horizons (1 for a target without a suffix) are read off
    % the names and returned in autocorrtimehorizons for simoptions.timehorizons. N_j must be common to the types and there can be no agejshifter (the PType
    % command does not implement them with simoptions.whichcombos).
    if usingautocorr==1
        if buildselectors==0
            error('TargetMoments.AutoCorrTransProbs needs N_j, simoptions and Names_i to be passed to SetupTargetMoments_FHorz (CalibrateLifeCycleModel_PType does; the estimation commands do not yet support these targets)')
        end
        if isstruct(N_j) || any(ageshift~=0)
            error('TargetMoments.AutoCorrTransProbs with permanent types needs N_j common to the types and no simoptions.agejshifter (EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType does not implement them with simoptions.whichcombos)')
        end
        autocorrmomentnames=cell(0,4);
        autocorrmomentsizes=[];
        a1vec=fieldnames(TargetMoments.AutoCorrTransProbs);
        for a1=1:length(a1vec)
            temp1=TargetMoments.AutoCorrTransProbs.(a1vec{a1});
            if ~isstruct(temp1)
                error(['TargetMoments.AutoCorrTransProbs.',a1vec{a1},': a target is .(fn).(stat), .(fn).(type).(stat), .(restriction).(fn).(stat) or .(restriction).(fn).(type).(stat)'])
            end
            a2vec=fieldnames(temp1);
            for a2=1:length(a2vec)
                temp2=temp1.(a2vec{a2});
                if isstruct(temp2)
                    a3vec=fieldnames(temp2);
                    for a3=1:length(a3vec)
                        temp3=temp2.(a3vec{a3});
                        if isstruct(temp3)
                            a4vec=fieldnames(temp3);
                            for a4=1:length(a4vec)
                                temp4=temp3.(a4vec{a4});
                                if isstruct(temp4)
                                    error(['TargetMoments.AutoCorrTransProbs.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},'.',a4vec{a4},' is a structure: a target has at most four levels (restriction.fn.type.stat)'])
                                end
                                targetmomentvec=[targetmomentvec; temp4(:)];
                                autocorrmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4}};
                                autocorrmomentsizes(end+1)=numel(temp4);
                            end
                        else
                            targetmomentvec=[targetmomentvec; temp3(:)];
                            autocorrmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},''};
                            autocorrmomentsizes(end+1)=numel(temp3);
                        end
                    end
                else
                    targetmomentvec=[targetmomentvec; temp2(:)];
                    autocorrmomentnames(end+1,:)={a1vec{a1},a2vec{a2},'',''};
                    autocorrmomentsizes(end+1)=numel(temp2);
                end
            end
        end
        autocorrcummomentsizes=cumsum(autocorrmomentsizes);
        % Classify each row as (fn, stat, page, slot)
        nrows=size(autocorrmomentnames,1);
        rowfn=cell(nrows,1);
        rowstat=cell(nrows,1);
        rowpage=zeros(nrows,1);
        rowslot=zeros(nrows,1);
        % The output rows of the command (as without permanent types): rowbase the FnsToEvaluate, rowM the number of periods, rowtype
        % 'average' or 'sum', rowlog 1 for the log
        rowbase=cell(nrows,1);
        rowM=ones(nrows,1);
        rowtype=cell(nrows,1);
        rowlog=zeros(nrows,1);
        for cc=1:nrows
            names=autocorrmomentnames(cc,:);
            names=names(~cellfun(@isempty,names));
            a1=names{1};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
            if ~isfn % a window or log row of a FnsToEvaluate
                tk=regexp(a1,'^(.+)_(\d+)period(Average|Sum)(_log)?$','tokens','once');
                if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                    isfn=true;
                else
                    tk=regexp(a1,'^(.+)_log$','tokens','once');
                    isfn=(~isempty(tk) && any(strcmp(FnNamesAll,tk{1})));
                end
            end
            if isrestr && isfn
                error(['TargetMoments.AutoCorrTransProbs.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
            elseif isrestr
                rowpage(cc)=1+find(strcmp(RestrNames,a1));
                if length(names)<3
                    error(['TargetMoments.AutoCorrTransProbs.',a1,'.',names{2},': a target under a conditional restriction needs a function and then a statistic'])
                end
                rowfn{cc}=names{2};
                rest=names(3:end);
            elseif isfn
                rowpage(cc)=1;
                rowfn{cc}=a1;
                rest=names(2:end);
            else
                error(['TargetMoments.AutoCorrTransProbs.',a1,' is neither a FnsToEvaluate (or one of its rows fnname_log, fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log) nor a conditional restriction in simoptions.conditionalrestrictions'])
            end
            rowbase{cc}=rowfn{cc};
            if ~any(strcmp(FnNamesAll,rowfn{cc}))
                tk=regexp(rowfn{cc},'^(.+)_(\d+)period(Average|Sum)(_log)?$','tokens','once');
                if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                    rowbase{cc}=tk{1};
                    rowM(cc)=str2double(tk{2});
                    rowtype{cc}=lower(tk{3});
                    rowlog(cc)=~isempty(tk{4});
                    if rowM(cc)<2
                        error(['TargetMoments.AutoCorrTransProbs: ',rowfn{cc},' names a window of ',tk{2},' periods: a single period is the FnsToEvaluate name itself (',tk{1},', or ',tk{1},'_log)'])
                    end
                else
                    tk=regexp(rowfn{cc},'^(.+)_log$','tokens','once');
                    if ~isempty(tk) && any(strcmp(FnNamesAll,tk{1}))
                        rowbase{cc}=tk{1};
                        rowlog(cc)=1;
                    else
                        error(['TargetMoments.AutoCorrTransProbs: ',rowfn{cc},' is not one of the FnsToEvaluate (nor fnname_log, fnname_MperiodAverage, fnname_MperiodSum, fnname_MperiodAverage_log, fnname_MperiodSum_log of one)'])
                    end
                end
            end
            if isempty(rest)
                error(['TargetMoments.AutoCorrTransProbs: the target for ',rowfn{cc},' needs a statistic (e.g. .AutoCorrelation)'])
            end
            if any(strcmp(Names_i,rest{1})) % a permanent type: its own output
                rowslot(cc)=find(strcmp(Names_i,rest{1}));
                if length(rest)~=2
                    error(['TargetMoments.AutoCorrTransProbs: the target for ',rowfn{cc},' of type ',rest{1},' is .(fn).(type).(stat), one statistic name after the type'])
                end
                rowstat{cc}=rest{2};
            else % the grouped output
                rowslot(cc)=N_i+1;
                if length(rest)~=1
                    error(['TargetMoments.AutoCorrTransProbs: the target for ',rowfn{cc},' is .(fn).(stat) or .(fn).(type).(stat) (',rest{1},' is neither a permanent type nor the last name)'])
                end
                rowstat{cc}=rest{1};
            end
        end
        FnsToEvaluate_AutoCorr=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowbase,FnNamesAll{ff}))
                FnsToEvaluate_AutoCorr.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesAC=fieldnames(FnsToEvaluate_AutoCorr);
        % simoptions.ntimeperiods and simoptions.ntimeperiodstype (one per command call: the targets cannot mix MperiodAverage and MperiodSum)
        AutoCorr_ntimeperiods=unique(rowM(:)');
        wtypes=unique(rowtype(rowM>1));
        if length(wtypes)>1
            error('TargetMoments.AutoCorrTransProbs: the targets name both MperiodAverage and MperiodSum rows, but the command computes one or the other in a call (simoptions.ntimeperiodstype): use one of them for all the targets')
        elseif isscalar(wtypes)
            AutoCorr_ntimeperiodstype=wtypes{1};
        else
            AutoCorr_ntimeperiodstype='average';
        end
        nblocksAC=length(AutoCorr_ntimeperiods);
        autocorrtimehorizons=[];
        % rows in blocks of the window lengths; logwhichcombos has no type dimension (the PType command applies it to every type and
        % the grouped stats), so a log target of one type or the grouped stats makes that combination the log for all of them
        AutoCorr_whichcombos=zeros(nblocksAC*length(FnNamesAC),N_j,1+nRestr,N_i+1);
        AutoCorr_logwhichcombos=zeros(nblocksAC*length(FnNamesAC),N_j,1+nRestr);
        AutoCorr_levelcombos=zeros(nblocksAC*length(FnNamesAC),N_j,1+nRestr);
        for cc=1:nrows
            stat=rowstat{cc};
            tk=regexp(stat,'_horizon(\d+)$','tokens','once'); % the horizon suffix
            if ~isempty(tk)
                K=str2double(tk{1});
                base=stat(1:end-length(tk{1})-8); % strip _horizonK
            else
                K=1;
                base=stat;
            end
            if any(strcmp(stat,{'Mean','StdDeviation'}))
                K=0;
            elseif ~any(strcmp(base,{'AutoCorrelation','AutoCovariance','PairMass','PairMean_j','PairMean_jplusk','PairStdDeviation_j','PairStdDeviation_jplusk'}))
                error(['TargetMoments.AutoCorrTransProbs: ',stat,' is not a targetable output (Mean, StdDeviation, AutoCorrelation, AutoCovariance and the Pair byproducts, each with an optional _horizonK suffix for horizon K>1; TransitionProbs cannot be targeted, use CustomModelStats)'])
            end
            if K>=1
                autocorrtimehorizons=[autocorrtimehorizons,K]; % the command computes exactly the horizons asked for (horizon 1 too)
            end
            names=autocorrmomentnames(cc,:);
            names=names(~cellfun(@isempty,names));
            temp=TargetMoments.AutoCorrTransProbs;
            for kk=1:length(names)
                temp=temp.(names{kk});
            end
            if ~(isvector(temp) && numel(temp)==N_j-K)
                error(['TargetMoments.AutoCorrTransProbs target ',rowfn{cc},' ',stat,' has ',num2str(numel(temp)),' entries, but it is 1 x ',num2str(N_j-K),' (N_j minus the horizon)'])
            end
            ff=(find(AutoCorr_ntimeperiods==rowM(cc))-1)*length(FnNamesAC)+find(strcmp(FnNamesAC,rowbase{cc})); % the row: the block of its window length
            pp=rowpage(cc);
            ss_slot=rowslot(cc);
            sel=double(reshape(~isnan(temp),1,[]));
            AutoCorr_whichcombos(ff,1:N_j-K,pp,ss_slot)=max(AutoCorr_whichcombos(ff,1:N_j-K,pp,ss_slot),sel); % entry j selects start age j
            if rowlog(cc)==1
                AutoCorr_logwhichcombos(ff,1:N_j-K,pp)=max(AutoCorr_logwhichcombos(ff,1:N_j-K,pp),sel);
            else
                AutoCorr_levelcombos(ff,1:N_j-K,pp)=max(AutoCorr_levelcombos(ff,1:N_j-K,pp),sel);
            end
        end
        if any(AutoCorr_logwhichcombos(:) & AutoCorr_levelcombos(:))
            error('TargetMoments.AutoCorrTransProbs: some targets ask for both a statistic and its log (fnname... and fnname..._log) at the same start age and restriction (for any permanent type or the grouped stats), but the command computes one or the other for each (simoptions.logwhichcombos): target only one of them at each start age')
        end
        autocorrtimehorizons=unique(autocorrtimehorizons);
    else
        autocorrmomentnames=cell(1,4);
        autocorrcummomentsizes=0;
        FnsToEvaluate_AutoCorr=struct();
        autocorrtimehorizons=[];
        AutoCorr_whichcombos=[];
        AutoCorr_ntimeperiods=1;
        AutoCorr_ntimeperiodstype='average';
        AutoCorr_logwhichcombos=0;
    end
    %% CrossSectionCovarCorr (EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz_PType) and AgeConditionalCrossSectionCovarCorr
    % (EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz_PType), nested as the commands' outputs: grouped targets
    % .(fn).Mean, .(fn).StdDeviation, .(fn1).CovarianceWith.(fn2), .(fn1).CorrelationWith.(fn2), .CovarianceMatrix, .CorrelationMatrix;
    % per-type targets .(fn).(type).Mean, .(fn).(type).StdDeviation, .(fn1).(type).CovarianceWith.(fn2), .(fn1).(type).CorrelationWith.(fn2);
    % each also under a conditional restriction as .(restriction).(...) (up to five names: restriction.fn1.type.CovarianceWith.fn2).
    % (The self-correlation .(fn).(fn) is always one and cannot be targeted; nor can the per-type matrices CovarianceMatrix_ptype/
    % CorrelationMatrix_ptype, which the commands shrink to the functions evaluated for the type: use the per-type pairs.) Scalars and
    % [nFns,nFns] for the plain command; 1 x (number of age groups) and [nFns,nFns,number of age groups] for the age-conditional one
    % (simoptions.agegroupings, else N_j; common to the types, no agejshifter). NaN omits an entry. The selector is pair-shaped per page
    % and slot, [nFns,nFns,1+nRestr,N_i+1] and [nFns,nFns,ngroups,1+nRestr,N_i+1] (page 1 unrestricted, then the restrictions in the
    % fieldnames order of simoptions.conditionalrestrictions; without restrictions the page dimension is dropped, as the commands read
    % it), symmetric, the diagonal being the own stats; a matrix target is indexed over every FnsToEvaluate, so it makes the reduced
    % set the full one.
    for xx=1:2
        if xx==1
            xf='CrossSectionCovarCorr'; usingx=usingcrosssec; nage=1;
        else
            xf='AgeConditionalCrossSectionCovarCorr'; usingx=usingagecrosssec; nage=1;
            if usingx==1 && buildselectors==1
                if isstruct(N_j) || any(ageshift~=0) || (isfield(simoptions,'agegroupings') && isstruct(simoptions.agegroupings))
                    error('TargetMoments.AgeConditionalCrossSectionCovarCorr with permanent types needs N_j and simoptions.agegroupings common to the types and no simoptions.agejshifter (EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz_PType does not implement them)')
                end
                if isfield(simoptions,'agegroupings')
                    nage=length(simoptions.agegroupings);
                else
                    nage=N_j;
                end
            end
        end
        if usingx==1
            if buildselectors==0
                error(['TargetMoments.',xf,' needs N_j, simoptions and Names_i to be passed to SetupTargetMoments_FHorz (CalibrateLifeCycleModel_PType does; the estimation commands do not yet support these targets)'])
            end
            xnames=cell(0,5);
            xsizes=[];
            a1vec=fieldnames(TargetMoments.(xf));
            for a1=1:length(a1vec)
                temp1=TargetMoments.(xf).(a1vec{a1});
                if ~isstruct(temp1) % one name: a grouped matrix
                    targetmomentvec=[targetmomentvec; temp1(:)];
                    xnames(end+1,:)={a1vec{a1},'','','',''};
                    xsizes(end+1)=numel(temp1);
                else
                    a2vec=fieldnames(temp1);
                    for a2=1:length(a2vec)
                        temp2=temp1.(a2vec{a2});
                        if ~isstruct(temp2)
                            targetmomentvec=[targetmomentvec; temp2(:)];
                            xnames(end+1,:)={a1vec{a1},a2vec{a2},'','',''};
                            xsizes(end+1)=numel(temp2);
                        else
                            a3vec=fieldnames(temp2);
                            for a3=1:length(a3vec)
                                temp3=temp2.(a3vec{a3});
                                if ~isstruct(temp3)
                                    targetmomentvec=[targetmomentvec; temp3(:)];
                                    xnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},'',''};
                                    xsizes(end+1)=numel(temp3);
                                else
                                    a4vec=fieldnames(temp3);
                                    for a4=1:length(a4vec)
                                        temp4=temp3.(a4vec{a4});
                                        if ~isstruct(temp4)
                                            targetmomentvec=[targetmomentvec; temp4(:)];
                                            xnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4},''};
                                            xsizes(end+1)=numel(temp4);
                                        else
                                            a5vec=fieldnames(temp4);
                                            for a5=1:length(a5vec)
                                                temp5=temp4.(a5vec{a5});
                                                if isstruct(temp5)
                                                    error(['TargetMoments.',xf,'.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},'.',a4vec{a4},'.',a5vec{a5},' is a structure: a target has at most five levels (restriction.fn1.type.CovarianceWith.fn2)'])
                                                end
                                                targetmomentvec=[targetmomentvec; temp5(:)];
                                                xnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4},a5vec{a5}};
                                                xsizes(end+1)=numel(temp5);
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            xcumsizes=cumsum(xsizes);
            nrows=size(xnames,1);
            rowfn1=cell(nrows,1);
            rowfn2=cell(nrows,1);
            rowpage=zeros(nrows,1);
            rowslot=zeros(nrows,1);
            rowmatrix=zeros(nrows,1);
            for cc=1:nrows
                names=xnames(cc,:);
                names=names(~cellfun(@isempty,names));
                a1=names{1};
                isrestr=any(strcmp(RestrNames,a1));
                isfn=any(strcmp(FnNamesAll,a1));
                if isrestr && isfn
                    error(['TargetMoments.',xf,'.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
                elseif isrestr
                    rowpage(cc)=1+find(strcmp(RestrNames,a1));
                    rest=names(2:end);
                    if isempty(rest)
                        error(['TargetMoments.',xf,'.',a1,': a target under a conditional restriction needs a function (then a type and/or Mean, StdDeviation, CovarianceWith.(fn2) or CorrelationWith.(fn2)) or CovarianceMatrix/CorrelationMatrix'])
                    end
                else
                    rowpage(cc)=1;
                    rest=names;
                end
                b1=rest{1};
                if isscalar(rest) % a grouped matrix
                    if ~any(strcmp(b1,{'CovarianceMatrix','CorrelationMatrix'}))
                        error(['TargetMoments.',xf,'.',strjoin(names,'.'),': a target with no statistic must be CovarianceMatrix or CorrelationMatrix'])
                    end
                    rowmatrix(cc)=1; rowslot(cc)=N_i+1;
                elseif any(strcmp(b1,{'CovarianceMatrix_ptype','CorrelationMatrix_ptype'}))
                    error(['TargetMoments.',xf,'.',strjoin(names,'.'),': the per-type matrices cannot be targeted (the commands shrink them to the functions evaluated for the type); target the per-type pairs .(fn1).(type).CovarianceWith.(fn2) instead'])
                elseif ~any(strcmp(FnNamesAll,b1))
                    error(['TargetMoments.',xf,'.',strjoin(names,'.'),': ',b1,' is not one of the FnsToEvaluate (nor a conditional restriction in simoptions.conditionalrestrictions, nor CovarianceMatrix/CorrelationMatrix)'])
                else
                    if any(strcmp(Names_i,rest{2})) % per type: .(fn).(type).(...)
                        rowslot(cc)=find(strcmp(Names_i,rest{2}));
                        rest2=rest(3:end);
                    else
                        rowslot(cc)=N_i+1;
                        rest2=rest(2:end);
                    end
                    if isempty(rest2)
                        error(['TargetMoments.',xf,'.',strjoin(names,'.'),': the target needs Mean, StdDeviation, CovarianceWith.(fn2) or CorrelationWith.(fn2) (after the type name for a per-type target)'])
                    elseif any(strcmp(rest2{1},{'CovarianceWith','CorrelationWith'}))
                        if numel(rest2)~=2 || ~any(strcmp(FnNamesAll,rest2{2}))
                            error(['TargetMoments.',xf,'.',strjoin(names,'.'),': ',rest2{1},' needs a second FnsToEvaluate name'])
                        end
                        rowfn1{cc}=b1; rowfn2{cc}=rest2{2};
                    elseif isscalar(rest2) && any(strcmp(rest2{1},{'Mean','StdDeviation'}))
                        rowfn1{cc}=b1; rowfn2{cc}=b1;
                    elseif isscalar(rest2) && strcmp(rest2{1},b1)
                        error(['TargetMoments.',xf,'.',strjoin(names,'.'),': the self-correlation is always one and cannot be targeted'])
                    else
                        error(['TargetMoments.',xf,'.',strjoin(names,'.'),': the targets are Mean, StdDeviation, CovarianceWith.(fn2), CorrelationWith.(fn2) (grouped, or after a type name), or the grouped CovarianceMatrix/CorrelationMatrix, each optionally under a conditional restriction'])
                    end
                end
            end
            FnsX=struct();
            for ff=1:length(FnNamesAll)
                if any(rowmatrix==1) || any(strcmp(rowfn1,FnNamesAll{ff})) || any(strcmp(rowfn2,FnNamesAll{ff}))
                    FnsX.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
                end
            end
            FnNamesX=fieldnames(FnsX);
            wcX=zeros(length(FnNamesX),length(FnNamesX),nage,1+nRestr,N_i+1);
            for cc=1:nrows
                names=xnames(cc,:);
                names=names(~cellfun(@isempty,names));
                temp=TargetMoments.(xf);
                for kk=1:length(names)
                    temp=temp.(names{kk});
                end
                pp=rowpage(cc);
                ss_slot=rowslot(cc);
                if rowmatrix(cc)==1 % a matrix: non-NaN entries select the pairs (symmetrised: the commands read the upper triangle)
                    if ~isequal(size(temp,1:3),[length(FnNamesX),length(FnNamesX),nage])
                        error(['TargetMoments.',xf,' target ',strjoin(names,'.'),' must be of size [',num2str(length(FnNamesX)),',',num2str(length(FnNamesX)),',',num2str(nage),'] (number of FnsToEvaluate, twice, number of age groups), NaN where not wanted'])
                    end
                    on=double(~isnan(temp));
                    wcX(:,:,:,pp,ss_slot)=max(wcX(:,:,:,pp,ss_slot),max(on,permute(on,[2,1,3])));
                else
                    if ~(isvector(temp) && numel(temp)==nage)
                        error(['TargetMoments.',xf,' target ',strjoin(names,'.'),' has ',num2str(numel(temp)),' entries, but there are ',num2str(nage),' age groups'])
                    end
                    f1=find(strcmp(FnNamesX,rowfn1{cc}));
                    f2=find(strcmp(FnNamesX,rowfn2{cc}));
                    on=double(reshape(~isnan(temp),[1,1,nage]));
                    wcX(f1,f2,:,pp,ss_slot)=max(wcX(f1,f2,:,pp,ss_slot),on);
                    wcX(f2,f1,:,pp,ss_slot)=max(wcX(f2,f1,:,pp,ss_slot),on);
                end
            end
            % the shapes the commands read: the page dimension only with restrictions, nage only for the age-conditional command
            if xx==1
                if nRestr==0
                    wcX=reshape(wcX,[length(FnNamesX),length(FnNamesX),N_i+1]);
                else
                    wcX=reshape(wcX,[length(FnNamesX),length(FnNamesX),1+nRestr,N_i+1]);
                end
            elseif nRestr==0
                wcX=reshape(wcX,[length(FnNamesX),length(FnNamesX),nage,N_i+1]);
            end
        else
            xnames=cell(1,5);
            xcumsizes=0;
            FnsX=struct();
            wcX=[];
        end
        if xx==1
            crosssecmomentnames=xnames; crossseccummomentsizes=xcumsizes; FnsToEvaluate_CrossSec=FnsX; CrossSec_whichcombos=wcX;
        else
            agecrosssecmomentnames=xnames; agecrossseccummomentsizes=xcumsizes; FnsToEvaluate_AgeCrossSec=FnsX; AgeCrossSec_whichcombos=wcX;
        end
    end
    %% A target that is NaN in full is dropped, with a warning (NaN is for omitting some entries of a target, ages or matrix entries; a
    % target omitted in full should not be listed, and with caliboptions.whichcombos=1 nothing is computed for it)
    nanchk_kinds={'AllStats','AgeConditionalStats','AutoCorrTransProbs','CrossSectionCovarCorr','AgeConditionalCrossSectionCovarCorr'};
    nanchk_using=[usingallstats,usinglcp,usingautocorr,usingcrosssec,usingagecrosssec];
    nanchk_sofar=0;
    for nanchk_kk=1:5
        if nanchk_using(nanchk_kk)==1
            if nanchk_kk==1
                nanchk_names=allstatmomentnames; nanchk_cum=allstatcummomentsizes;
            elseif nanchk_kk==2
                nanchk_names=acsmomentnames; nanchk_cum=acscummomentsizes;
            elseif nanchk_kk==3
                nanchk_names=autocorrmomentnames; nanchk_cum=autocorrcummomentsizes;
            elseif nanchk_kk==4
                nanchk_names=crosssecmomentnames; nanchk_cum=crossseccummomentsizes;
            else
                nanchk_names=agecrosssecmomentnames; nanchk_cum=agecrossseccummomentsizes;
            end
            for cc=1:size(nanchk_names,1)
                if cc==1
                    nanchk_idx=nanchk_sofar+1:nanchk_sofar+nanchk_cum(1);
                else
                    nanchk_idx=nanchk_sofar+nanchk_cum(cc-1)+1:nanchk_sofar+nanchk_cum(cc);
                end
                if all(isnan(targetmomentvec(nanchk_idx)))
                    warning(['TargetMoments.',nanchk_kinds{nanchk_kk},'.',strjoin(nanchk_names(cc,~cellfun(@isempty,nanchk_names(cc,:))),'.'),' is NaN in full: a target that is omitted in full should not be listed (NaN is for omitting some of its entries); it is dropped'])
                end
            end
            nanchk_sofar=nanchk_sofar+nanchk_cum(end);
        end
    end
    %% The selectors (used by CalibrateLifeCycleModel_PType when caliboptions.whichcombos=1)
    selectors=struct();
    if buildselectors==1
        selectors.AllStats.whichcombos=AllStats_whichcombos;
        selectors.AllStats.whichstats=AllStats_whichstatsArr;
        selectors.ACStats.whichcombos=ACStats_whichcombos;
        selectors.ACStats.whichstats=ACStats_whichstatsArr;
        % The window and log rows of the AgeConditionalStats targets (as without permanent types; logwhichcombos has no type dimension)
        selectors.ACStats.haswindowlog=ACStats_haswindowlog;
        if ACStats_haswindowlog==1
            selectors.ACStats.ntimeperiods=ACStats_ntimeperiods;
            selectors.ACStats.ntimeperiodstype=ACStats_ntimeperiodstype;
            selectors.ACStats.logwhichcombos=ACStats_logwhichcombos;
            if any(ACStats_ntimeperiods>1) && ~agedefault
                selectors.ACStats.agegroupings=1:1:N_j;
            else
                selectors.ACStats.agegroupings=[];
            end
            selectors.ACStats.split=ACStats_split;
            if ACStats_split==1
                selectors.ACStatsW.whichcombos=ACStatsW_whichcombos;
                selectors.ACStatsW.whichstats=ACStatsW_whichstatsArr;
                selectors.ACStatsW.ntimeperiods=ACStatsW_ntimeperiods;
                selectors.ACStatsW.ntimeperiodstype=ACStats_ntimeperiodstype;
                selectors.ACStatsW.logwhichcombos=ACStatsW_logwhichcombos;
                selectors.ACStatsW.agegroupings=1:1:N_j;
            end
        end
        selectors.AutoCorr.whichcombos=AutoCorr_whichcombos;
        selectors.AutoCorr.ntimeperiods=AutoCorr_ntimeperiods; % the window lengths the targets name (simoptions.ntimeperiods; the rows of whichcombos are in blocks of these)
        selectors.AutoCorr.ntimeperiodstype=AutoCorr_ntimeperiodstype; % simoptions.ntimeperiodstype
        selectors.AutoCorr.logwhichcombos=AutoCorr_logwhichcombos; % simoptions.logwhichcombos (the targets named ..._log; no type dimension)
        selectors.CrossSec.whichcombos=CrossSec_whichcombos;
        selectors.AgeCrossSec.whichcombos=AgeCrossSec_whichcombos;
    end
end


%% Custom Model Stats
if usingcustomstats==1
    % You just have to give a dedicated name to each CustomModelStat
    cmsmomentnames=fieldnames(TargetMoments.CustomModelStats);
    cmsmomentsizes=0;
    for a1=1:length(cmsmomentnames)
        if size(TargetMoments.CustomModelStats.(cmsmomentnames{a1}),2)==1 % already column vector
            targetmomentvec=[targetmomentvec; TargetMoments.CustomModelStats.(cmsmomentnames{a1})]; % append to end
        else
            targetmomentvec=[targetmomentvec; TargetMoments.CustomModelStats.(cmsmomentnames{a1})']; % transpose, then append to end
        end

        cmsmomentsizes(a1)=length(TargetMoments.CustomModelStats.(cmsmomentnames{a1}));
    end
    cmscummomentsizes=cumsum(cmsmomentsizes); % Note: this is zero if CustomModelStats is unused
else
    % Placeholders
    cmsmomentnames=cell(1,1);
    cmscummomentsizes=0;
end

end
