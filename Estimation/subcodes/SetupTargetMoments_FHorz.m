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
% Three more kinds of target stand alongside AllStats and AgeConditionalStats (2026-10-07; single-type only so far): TargetMoments.AutoCorrTransProbs
% (EvalFnOnAgentDist_AutoCorrTransProbs_FHorz), .CrossSectionCovarCorr (EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz) and
% .AgeConditionalCrossSectionCovarCorr (the age-conditional version), each nested like its command's output. They enter targetmomentvec
% after AgeConditionalStats and before CustomModelStats, with their own using* flag, names table, cumulative sizes, reduced FnsToEvaluate
% and selector (selectors.AutoCorr/.CrossSec/.AgeCrossSec, whichcombos only: these commands have no per-combination whichstats);
% autocorrtimehorizons holds the horizons K>=2 read off the AutoCorr target names, for simoptions.timehorizons.

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
        error('TargetMoments.AutoCorrTransProbs, .CrossSectionCovarCorr and .AgeConditionalCrossSectionCovarCorr need N_j and simoptions to be passed to SetupTargetMoments_FHorz (CalibrateLifeCycleModel does; the estimation commands do not yet support these targets)')
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
            if buildselectors==1 % (without simoptions the page of a restricted target is unknown and no selectors are built)
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
                error(['TargetMoments.AgeConditionalStats.',a1,' is neither a FnsToEvaluate nor a conditional restriction in simoptions.conditionalrestrictions (restrictions must be in simoptions before CalibrateLifeCycleModel is called)'])
            end
            if ~any(strcmp(FnNamesAll,rowfn{cc}))
                error(['TargetMoments.AgeConditionalStats: ',rowfn{cc},' is not one of the FnsToEvaluate'])
            end
        end
        FnsToEvaluate_ACStats=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowfn,FnNamesAll{ff}))
                FnsToEvaluate_ACStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesL=fieldnames(FnsToEvaluate_ACStats);
        % whichstats: the union (7-vector), and per (fn,agegroup,page) [nFns,ngroups,1+nRestr,7] with an age on only where the target is not NaN
        ACStats_whichstats=zeros(7,1);
        if buildselectors==1
            ACStats_whichstatsArr=zeros(length(FnNamesL),ngroups,1+nRestr,7);
        end
        for cc=1:nrows
            ff=find(strcmp(FnNamesL,rowfn{cc}));
            pp=rowpage(cc);
            if buildselectors==1
                if isempty(acsmomentnames{cc,3})
                    temp=TargetMoments.AgeConditionalStats.(acsmomentnames{cc,1}).(acsmomentnames{cc,2});
                else
                    temp=TargetMoments.AgeConditionalStats.(acsmomentnames{cc,1}).(acsmomentnames{cc,2}).(acsmomentnames{cc,3});
                end
                if isvector(temp) && numel(temp)==ngroups
                    ageon=reshape(~isnan(temp),[1,ngroups]);
                elseif ismatrix(temp) && size(temp,2)==ngroups
                    ageon=any(~isnan(temp),1); % a matrix target: the age group is on if any entry of its column is targeted
                else
                    error(['TargetMoments.AgeConditionalStats target for ',rowfn{cc},' ',rowstat{cc},' has ',num2str(numel(temp)),' entries, but there are ',num2str(ngroups),' age groups (it must be a vector over the age groups, or a matrix with the age groups as columns)'])
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
                for ss=sidx
                    if ss==4 % Gini/LorenzCurve: 3 is Gini only, 2 is the Lorenz curve (and Gini); a LorenzCurve target overrides a Gini-only 3
                        if strcmp(rowstat{cc},'LorenzCurve')
                            ACStats_whichstatsArr(ff,ageon,pp,4)=2;
                        else
                            current=ACStats_whichstatsArr(ff,:,pp,4);
                            current(ageon & current==0)=3;
                            ACStats_whichstatsArr(ff,:,pp,4)=current;
                        end
                    else
                        ACStats_whichstatsArr(ff,ageon,pp,ss)=1;
                    end
                end
            end
        end
        if buildselectors==1
            ACStats_whichcombos=double(any(ACStats_whichstatsArr,4)); % [nFns,ngroups,1+nRestr]
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
    end
    %% AutoCorrTransProbs (EvalFnOnAgentDist_AutoCorrTransProbs_FHorz): targets .(fn).(stat) and .(restriction).(fn).(stat); the stat is
    % Mean or StdDeviation (1 x N_j), or AutoCorrelation/AutoCovariance/PairMass/PairMean_j/PairMean_jplusk/PairStdDeviation_j/PairStdDeviation_jplusk
    % at horizon one (no suffix, 1 x N_j-1) or at horizon K (suffix _kK, 1 x N_j-K). TransitionProbs cannot be targeted (use CustomModelStats).
    % The selector is by start age: entry j of a horizon-K target selects start age j (Mean/StdDeviation: age j). Horizons K>=2 are read off
    % the names and returned in autocorrtimehorizons for simoptions.timehorizons.
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
        for cc=1:nrows
            a1=autocorrmomentnames{cc,1}; a2=autocorrmomentnames{cc,2}; a3=autocorrmomentnames{cc,3};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
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
                error(['TargetMoments.AutoCorrTransProbs.',a1,' is neither a FnsToEvaluate nor a conditional restriction in simoptions.conditionalrestrictions'])
            end
            if ~any(strcmp(FnNamesAll,rowfn{cc}))
                error(['TargetMoments.AutoCorrTransProbs: ',rowfn{cc},' is not one of the FnsToEvaluate'])
            end
        end
        FnsToEvaluate_AutoCorr=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowfn,FnNamesAll{ff}))
                FnsToEvaluate_AutoCorr.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesAC=fieldnames(FnsToEvaluate_AutoCorr);
        autocorrtimehorizons=[];
        if buildselectors==1
            AutoCorr_whichcombos=zeros(length(FnNamesAC),N_j,1+nRestr);
        else
            AutoCorr_whichcombos=[];
        end
        for cc=1:nrows
            stat=rowstat{cc};
            tk=regexp(stat,'_k(\d+)$','tokens','once'); % the horizon suffix
            if ~isempty(tk)
                K=str2double(tk{1});
                base=stat(1:end-length(tk{1})-2);
            else
                K=1;
                base=stat;
            end
            if any(strcmp(stat,{'Mean','StdDeviation'}))
                K=0;
            elseif ~any(strcmp(base,{'AutoCorrelation','AutoCovariance','PairMass','PairMean_j','PairMean_jplusk','PairStdDeviation_j','PairStdDeviation_jplusk'}))
                error(['TargetMoments.AutoCorrTransProbs: ',stat,' is not a targetable output (Mean, StdDeviation, AutoCorrelation, AutoCovariance and the Pair byproducts, each with an optional _kK horizon suffix; TransitionProbs cannot be targeted, use CustomModelStats)'])
            end
            if K>=2
                autocorrtimehorizons=[autocorrtimehorizons,K];
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
                ff=find(strcmp(FnNamesAC,rowfn{cc}));
                pp=rowpage(cc);
                AutoCorr_whichcombos(ff,1:N_j-K,pp)=max(AutoCorr_whichcombos(ff,1:N_j-K,pp),double(reshape(~isnan(temp),1,[]))); % entry j selects start age j
            end
        end
        autocorrtimehorizons=unique(autocorrtimehorizons);
    else
        autocorrmomentnames=cell(1,3);
        autocorrcummomentsizes=0;
        FnsToEvaluate_AutoCorr=struct();
        autocorrtimehorizons=[];
        AutoCorr_whichcombos=[];
    end
    %% CrossSectionCovarCorr (EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz) and AgeConditionalCrossSectionCovarCorr
    % (EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz): targets .(fn1).CovarianceWith.(fn2), .(fn1).CorrelationWith.(fn2),
    % .(fn).Mean, .(fn).StdDeviation, or the matrices .CovarianceMatrix / .CorrelationMatrix (the self-correlation .(fn).(fn) is always one and cannot be targeted). Scalars and
    % [nFns,nFns] for the plain command; 1 x (number of age groups) and [nFns,nFns,number of age groups] for the age-conditional one, the age
    % axis as the command lays it out (simoptions.agegroupings, else N_j). NaN omits an entry (a pair, an age group, a matrix entry).
    % The selector is pair-shaped, [nFns,nFns(,ngroups)], symmetric, the diagonal being the own stats. A matrix target is indexed over
    % every FnsToEvaluate, so it makes the reduced set the full one. Neither command implements conditional restrictions.
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
            xnames=cell(0,3);
            xsizes=[];
            a1vec=fieldnames(TargetMoments.(xf));
            for a1=1:length(a1vec)
                temp1=TargetMoments.(xf).(a1vec{a1});
                if ~isstruct(temp1) % one name: a matrix
                    targetmomentvec=[targetmomentvec; temp1(:)];
                    xnames(end+1,:)={a1vec{a1},'',''};
                    xsizes(end+1)=numel(temp1);
                else
                    a2vec=fieldnames(temp1);
                    for a2=1:length(a2vec)
                        temp2=temp1.(a2vec{a2});
                        if ~isstruct(temp2)
                            targetmomentvec=[targetmomentvec; temp2(:)];
                            xnames(end+1,:)={a1vec{a1},a2vec{a2},''};
                            xsizes(end+1)=numel(temp2);
                        else
                            a3vec=fieldnames(temp2);
                            for a3=1:length(a3vec)
                                if isstruct(temp2.(a3vec{a3}))
                                    error(['TargetMoments.',xf,'.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},' is a structure: a target has at most three levels (fn1.CovarianceWith.fn2)'])
                                end
                                targetmomentvec=[targetmomentvec; reshape(temp2.(a3vec{a3}),[],1)];
                                xnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3}};
                                xsizes(end+1)=numel(temp2.(a3vec{a3}));
                            end
                        end
                    end
                end
            end
            xcumsizes=cumsum(xsizes);
            nrows=size(xnames,1);
            rowfn1=cell(nrows,1);
            rowfn2=cell(nrows,1);
            usematrix=0;
            for cc=1:nrows
                a1=xnames{cc,1}; a2=xnames{cc,2}; a3=xnames{cc,3};
                if isempty(a2)
                    if ~any(strcmp(a1,{'CovarianceMatrix','CorrelationMatrix'}))
                        error(['TargetMoments.',xf,'.',a1,': a one-level target must be CovarianceMatrix or CorrelationMatrix'])
                    end
                    usematrix=1;
                elseif ~any(strcmp(FnNamesAll,a1))
                    error(['TargetMoments.',xf,'.',a1,' is not one of the FnsToEvaluate'])
                elseif any(strcmp(a2,{'CovarianceWith','CorrelationWith'}))
                    if isempty(a3) || ~any(strcmp(FnNamesAll,a3))
                        error(['TargetMoments.',xf,'.',a1,'.',a2,' needs a second FnsToEvaluate name'])
                    end
                    rowfn1{cc}=a1;
                    rowfn2{cc}=a3;
                elseif any(strcmp(a2,{'Mean','StdDeviation'})) && isempty(a3)
                    rowfn1{cc}=a1;
                    rowfn2{cc}=a1;
                else
                    error(['TargetMoments.',xf,'.',a1,'.',a2,': the targets are Mean, StdDeviation, CovarianceWith.(fn2), CorrelationWith.(fn2), or CovarianceMatrix/CorrelationMatrix (the self-correlation is always one and cannot be targeted)'])
                end
            end
            FnsX=struct();
            for ff=1:length(FnNamesAll)
                if usematrix==1 || any(strcmp(rowfn1,FnNamesAll{ff})) || any(strcmp(rowfn2,FnNamesAll{ff}))
                    FnsX.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
                end
            end
            FnNamesX=fieldnames(FnsX);
            if buildselectors==1
                wcX=zeros(length(FnNamesX),length(FnNamesX),nage);
                for cc=1:nrows
                    temp=TargetMoments.(xf);
                    for kk=1:3
                        if ~isempty(xnames{cc,kk})
                            temp=temp.(xnames{cc,kk});
                        end
                    end
                    if isempty(xnames{cc,2}) % a matrix: non-NaN entries select the pairs
                        if ~isequal(size(temp,1:3),[length(FnNamesX),length(FnNamesX),nage])
                            error(['TargetMoments.',xf,'.',xnames{cc,1},' must be of size [',num2str(length(FnNamesX)),',',num2str(length(FnNamesX)),',',num2str(nage),'] (number of FnsToEvaluate, twice, number of age groups), NaN where not wanted'])
                        end
                        on=double(~isnan(temp));
                        wcX=max(wcX,max(on,permute(on,[2,1,3]))); % symmetric: the commands read the upper triangle, a lower-triangle entry alone must still select the pair
                    else
                        if ~(isvector(temp) && numel(temp)==nage)
                            error(['TargetMoments.',xf,' target ',xnames{cc,1},' ',xnames{cc,2},' has ',num2str(numel(temp)),' entries, but there are ',num2str(nage),' age groups'])
                        end
                        f1=find(strcmp(FnNamesX,rowfn1{cc}));
                        f2=find(strcmp(FnNamesX,rowfn2{cc}));
                        on=double(reshape(~isnan(temp),[1,1,nage]));
                        wcX(f1,f2,:)=max(wcX(f1,f2,:),on);
                        wcX(f2,f1,:)=max(wcX(f2,f1,:),on);
                    end
                end
            else
                wcX=[];
            end
        else
            xnames=cell(1,3);
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
    %% The selectors for simoptions.whichcombos/whichstats (used by CalibrateLifeCycleModel when caliboptions.whichcombos=1)
    selectors=struct();
    if buildselectors==1
        selectors.AllStats.whichcombos=AllStats_whichcombos;
        selectors.AllStats.whichstats=AllStats_whichstatsArr;
        selectors.ACStats.whichcombos=ACStats_whichcombos;
        selectors.ACStats.whichstats=ACStats_whichstatsArr;
        selectors.AutoCorr.whichcombos=AutoCorr_whichcombos;
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
    if usingautocorr==1 || usingcrosssec==1 || usingagecrosssec==1
        error('TargetMoments.AutoCorrTransProbs, .CrossSectionCovarCorr and .AgeConditionalCrossSectionCovarCorr are not yet supported with permanent types')
    end
    autocorrmomentnames=cell(1,3); autocorrcummomentsizes=0; FnsToEvaluate_AutoCorr=struct(); autocorrtimehorizons=[];
    crosssecmomentnames=cell(1,3); crossseccummomentsizes=0; FnsToEvaluate_CrossSec=struct();
    agecrosssecmomentnames=cell(1,3); agecrossseccummomentsizes=0; FnsToEvaluate_AgeCrossSec=struct();
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
            if buildselectors==1
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
        for cc=1:nrows
            names=acsmomentnames(cc,:);
            names=names(~cellfun(@isempty,names));
            a1=names{1};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
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
                error(['TargetMoments.AgeConditionalStats.',a1,' is neither a FnsToEvaluate nor a conditional restriction in simoptions.conditionalrestrictions'])
            end
            if ~any(strcmp(FnNamesAll,rowfn{cc}))
                error(['TargetMoments.AgeConditionalStats: ',rowfn{cc},' is not one of the FnsToEvaluate'])
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
            if any(strcmp(rowfn,FnNamesAll{ff}))
                FnsToEvaluate_ACStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesL=fieldnames(FnsToEvaluate_ACStats);
        ACStats_whichstats=zeros(7,1);
        if buildselectors==1
            ACStats_whichstatsArr=zeros(length(FnNamesL),maxngroups,1+nRestr,N_i+1,7);
        end
        for cc=1:nrows
            ff=find(strcmp(FnNamesL,rowfn{cc}));
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
                if ss_slot<=N_i
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
                else
                    error(['TargetMoments.AgeConditionalStats target for ',rowfn{cc},' ',rowstat{cc},' has ',num2str(numel(temp)),' entries, but there are ',num2str(nslot),' age groups for it (it must be a vector over the age groups, or a matrix with the age groups as columns)'])
                end
                agepos=(1:nslot)+shift; % this slot's age groups on the common age axis
                keep=(agepos<=maxngroups);
                ageonpos=false(1,maxngroups);
                ageonpos(agepos(keep))=ageon(keep);
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
                for ss=sidx
                    if ss==4 % Gini/LorenzCurve: 3 is Gini only, 2 the Lorenz curve (and Gini); a LorenzCurve target overrides a Gini-only 3
                        if strcmp(rowstat{cc},'LorenzCurve')
                            ACStats_whichstatsArr(ff,ageonpos,pp,ss_slot,4)=2;
                        else
                            current=reshape(ACStats_whichstatsArr(ff,:,pp,ss_slot,4),[1,maxngroups]);
                            current(ageonpos & current==0)=3;
                            ACStats_whichstatsArr(ff,:,pp,ss_slot,4)=current;
                        end
                    else
                        ACStats_whichstatsArr(ff,ageonpos,pp,ss_slot,ss)=1;
                    end
                end
            end
        end
        if buildselectors==1
            ACStats_whichcombos=double(any(ACStats_whichstatsArr,5)); % [nFns,maxngroups,1+nRestr,N_i+1]
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
    end
    %% The selectors (used by CalibrateLifeCycleModel_PType when caliboptions.whichcombos=1)
    selectors=struct();
    if buildselectors==1
        selectors.AllStats.whichcombos=AllStats_whichcombos;
        selectors.AllStats.whichstats=AllStats_whichstatsArr;
        selectors.ACStats.whichcombos=ACStats_whichcombos;
        selectors.ACStats.whichstats=ACStats_whichstatsArr;
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
