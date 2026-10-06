function [targetmomentvec,usingallstats,usinglcp,usingcustomstats, allstatmomentnames,allstatcummomentsizes,AllStats_whichstats, FnsToEvaluate_AllStats, acsmomentnames, acscummomentsizes, ACStats_whichstats, FnsToEvaluate_ACStats,cmsmomentnames, cmscummomentsizes, selectors]=SetupTargetMoments_FHorz(TargetMoments,FnsToEvaluate,useptype,N_j,simoptions)
% useptype is 0 or 1
% N_j and simoptions are optional (the estimation commands do not yet pass them): when both are given and useptype==0, the last
% output 'selectors' holds the per-combination selectors for simoptions.whichcombos/whichstats of the stats commands:
%   selectors.AllStats.whichcombos  [nFns,1+nRestr]           selectors.AllStats.whichstats  [nFns,1+nRestr,7]
%   selectors.ACStats.whichcombos   [nFns,nAgeGroups,1+nRestr] selectors.ACStats.whichstats   [nFns,nAgeGroups,1+nRestr,7]
% with the functions in the order of FnsToEvaluate_AllStats/FnsToEvaluate_ACStats (the order of FnsToEvaluate) and the pages
% being the unrestricted stats and then simoptions.conditionalrestrictions in fieldnames order. An age-conditional target that is
% NaN at an age turns that age off for that statistic. Otherwise selectors is an empty struct.
% The 7-vectors AllStats_whichstats and ACStats_whichstats are the union over all targets (used when caliboptions.whichcombos=0
% is not set, i.e. by the estimation commands).

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
if isfield(TargetMoments,'CustomModelStats')
    usingcustomstats=1;
else
    usingcustomstats=0;
end

temp=fieldnames(TargetMoments);
for a1=1:length(temp)
    if ~isempty(setdiff(temp{a1}, {'AllStats','AgeConditionalStats','CustomModelStats'}))
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
    %% The selectors for simoptions.whichcombos/whichstats (used by CalibrateLifeCycleModel when caliboptions.whichcombos=1)
    selectors=struct();
    if buildselectors==1
        selectors.AllStats.whichcombos=AllStats_whichcombos;
        selectors.AllStats.whichstats=AllStats_whichstatsArr;
        selectors.ACStats.whichcombos=ACStats_whichcombos;
        selectors.ACStats.whichstats=ACStats_whichstatsArr;
    end
elseif useptype==1
    % PType means we need the third level a3vec
    % conditionalrestrictions means we need the third level a3vec
    % To allow conditionalrestrictions and ptype at once we go to fourth level a4vec

    % Get all of the moments out of TargetMoments and make them into a vector
    % Also, store all the names
    targetmomentvec=[]; % Can't preallocate as have no idea how big this will be
    % Ends up a column vector (create row vector, then transpose)

    % First, do those in AllStats
    if usingallstats==1
        allstatmomentnames=cell(1,4);
        allstatmomentcounter=0;
        allstatmomentsizes=0;
        a1vec=fieldnames(TargetMoments.AllStats); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}));% These will be Mean, etc
                    for a3=1:length(a3vec)
                        if isstruct(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}))
                            a4vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a3}).(a3vec{a3}));% These will be Mean, etc. Only relevant when ptype & conditionalrestrictions together.
                            for a4=1:length(a4vec)
                                allstatmomentcounter=allstatmomentcounter+1;
                                if size(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}),2)==1 % already column vector
                                    targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4})]; % append to end
                                else
                                    targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4})']; % transpose, then append to end
                                end
                                allstatmomentnames(allstatmomentcounter,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4}};
                                allstatmomentsizes(allstatmomentcounter)=length(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}));
                            end
                        else
                            allstatmomentcounter=allstatmomentcounter+1;
                            if size(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}),2)==1 % already column vector
                                targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})]; % append to end
                            else
                                targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})']; % transpose, then append to end
                            end
                            allstatmomentnames(allstatmomentcounter,1:3)={a1vec{a1},a2vec{a2},a3vec{a3}};
                            allstatmomentsizes(allstatmomentcounter)=length(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));
                        end
                    end
                else
                    a3vec={};
                    allstatmomentcounter=allstatmomentcounter+1;
                    if size(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}),2)==1 % already column vector
                        targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2})]; % append to end
                    else
                        targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2})']; % transpose, then append to end
                    end
                    allstatmomentnames(allstatmomentcounter,1:2)={a1vec{a1},a2vec{a2}};
                    allstatmomentsizes(allstatmomentcounter)=length(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}));
                end
            end
        end
        allstatcummomentsizes=cumsum(allstatmomentsizes); % Note: this is zero is AllStats is unused
        % To do AllStats faster, we use simoptions.whichstats so that we only compute the stats we want.
        AllStats_whichstats=zeros(7,1);
        for aa=2:4
            if any(strcmp(allstatmomentnames(:,aa),'Mean'))
                AllStats_whichstats(1)=1;
            end
            if any(strcmp(allstatmomentnames(:,aa),'Median'))
                AllStats_whichstats(2)=1;
            end
            if any(strcmp(allstatmomentnames(:,aa),'RatioMeanToMedian'))
                AllStats_whichstats(1)=1;
                AllStats_whichstats(2)=1;
            end
            if any(strcmp(allstatmomentnames(:,aa),'Variance')) || any(strcmp(allstatmomentnames(:,aa),'StdDeviation'))
                AllStats_whichstats(3)=1;
            end
            if any(strcmp(allstatmomentnames(:,aa),'Gini'))
                if AllStats_whichstats(4)==0 % Avoid overwriting if it is 1 from LorenzCurve
                    AllStats_whichstats(4)=3;
                end
            end
            if any(strcmp(allstatmomentnames(:,aa),'LorenzCurve'))
                AllStats_whichstats(4)=1;
            end
            if any(strcmp(allstatmomentnames(:,aa),'Maximum')) || any(strcmp(allstatmomentnames(:,aa),'Minimum'))
                AllStats_whichstats(5)=1;
            end
            if any(strcmp(allstatmomentnames(:,aa),'QuantileCutoffs')) || any(strcmp(allstatmomentnames(:,aa),'QuantileMeans'))
                AllStats_whichstats(6)=1; % quantiles are whichstats(6) (5 is Minimum/Maximum)
            end
            if any(strcmp(allstatmomentnames(:,aa),'MoreInequality'))
                AllStats_whichstats(7)=1;
            end
        end
        % To do AllStats faster, just evaluate the relevant functions
        FnsToEvaluate_AllStats=struct();
        % Put a1vec and a2vec together, then find just those which are in FnsToEvaluate
        a123vec={}; for cc=1:size(allstatmomentnames,1), for kk=1:size(allstatmomentnames,2), if ischar(allstatmomentnames{cc,kk}) && ~isempty(allstatmomentnames{cc,kk}), a123vec{end+1,1}=allstatmomentnames{cc,kk}; end, end, end % every name in every target (not just the last target's levels)
        a123vec=intersect(a123vec,fieldnames(FnsToEvaluate));
        for ff=1:length(a123vec)
            FnsToEvaluate_AllStats.(a123vec{ff})=FnsToEvaluate.(a123vec{ff});
        end
        % % all stats should be of length 1 [actually, no, they might be, e.g., QuantileMeans]
        % for ii=1:length(allstatmomentsizes)
        %     if allstatmomentsizes(ii)~=1
        %         errorstr=['Target Age-Conditional Stats must be of length() N_j (if you want to ignore some ages, use NaN for those ages); problem is with ', allstatmomentsizes{ii,1}, ' ', allstatmomentsizes{ii,2}, ' ',allstatmomentsizes{ii,3},' \n'];
        %         error(errorstr)
        %     end
        % end
    else
        % Placeholders
        allstatmomentnames=cell(1,3);
        allstatcummomentsizes=0;
        AllStats_whichstats=zeros(7,1);
        FnsToEvaluate_AllStats=struct();
    end





    % Second, do those in AgeConditionalStats
    if usinglcp==1
        acsmomentnames=cell(1,4);
        acsmomentcounter=0;
        acsmomentsizes=0;
        a1vec=fieldnames(TargetMoments.AgeConditionalStats); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}));% These will be Mean, etc. Only relevant when ptype or conditionalrestrictions.
                    for a3=1:length(a3vec)
                        if isstruct(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}))
                            a4vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a3}).(a3vec{a3}));% These will be Mean, etc. Only relevant when ptype & conditionalrestrictions together.
                            for a4=1:length(a4vec)
                                acsmomentcounter=acsmomentcounter+1;
                                if size(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}),2)==1 % already column vector
                                    targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4})]; % append to end
                                else
                                    targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4})']; % transpose, then append to end
                                end
                                acsmomentnames(acsmomentcounter,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4}};
                                acsmomentsizes(acsmomentcounter)=length(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}));
                            end
                        else
                            acsmomentcounter=acsmomentcounter+1;
                            if size(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}),2)==1 % already column vector
                                targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})]; % append to end
                            else
                                targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})']; % transpose, then append to end
                            end
                            acsmomentnames(acsmomentcounter,1:3)={a1vec{a1},a2vec{a2},a3vec{a3}};
                            acsmomentsizes(acsmomentcounter)=length(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));
                        end
                    end
                else
                    a3vec={};
                    acsmomentcounter=acsmomentcounter+1;
                    if size(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}),2)==1 % already column vector
                        targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2})]; % append to end
                    else
                        targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2})']; % transpose, then append to end
                    end
                    acsmomentnames(acsmomentcounter,1:2)={a1vec{a1},a2vec{a2}};
                    acsmomentsizes(acsmomentcounter)=length(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}));
                end
            end
        end
        acscummomentsizes=cumsum(acsmomentsizes); % Note: this is zero is AgeConditionalStats is unused
        % To do AgeConditionalStats faster, we use simoptions.whichstats so that we only compute the stats we want.
        ACStats_whichstats=zeros(7,1);
        for aa=2:4
            if any(strcmp(acsmomentnames(:,aa),'Mean'))
                ACStats_whichstats(1)=1;
            end
            if any(strcmp(acsmomentnames(:,aa),'Median'))
                ACStats_whichstats(2)=1;
            end
            if any(strcmp(acsmomentnames(:,aa),'RatioMeanToMedian'))
                ACStats_whichstats(1)=1;
                ACStats_whichstats(2)=1;
            end
            if any(strcmp(acsmomentnames(:,aa),'Variance')) || any(strcmp(acsmomentnames(:,aa),'StdDeviation'))
                ACStats_whichstats(3)=1;
            end
            if any(strcmp(acsmomentnames(:,aa),'Gini'))
                if ACStats_whichstats(4)==0 % Avoid overwriting if it is 1 from LorenzCurve
                    ACStats_whichstats(4)=3;
                end
            end
            if any(strcmp(acsmomentnames(:,aa),'LorenzCurve'))
                ACStats_whichstats(4)=2;
            end
            if any(strcmp(acsmomentnames(:,aa),'Maximum')) || any(strcmp(acsmomentnames(:,aa),'Minimum'))
                ACStats_whichstats(5)=1;
            end
            if any(strcmp(acsmomentnames(:,aa),'QuantileCutoffs')) || any(strcmp(acsmomentnames(:,aa),'QuantileMeans'))
                ACStats_whichstats(6)=1; % quantiles are whichstats(6) (5 is Minimum/Maximum)
            end
            if any(strcmp(acsmomentnames(:,aa),'MoreInequality'))
                ACStats_whichstats(7)=1;
            end
        end
        % To do AgeConditionalStats faster, just evaluate the relevant functions
        FnsToEvaluate_ACStats=struct();
        % Put a1vec and a2vec together, then find just those which are in FnsToEvaluate
        a123vec={}; for cc=1:size(acsmomentnames,1), for kk=1:size(acsmomentnames,2), if ischar(acsmomentnames{cc,kk}) && ~isempty(acsmomentnames{cc,kk}), a123vec{end+1,1}=acsmomentnames{cc,kk}; end, end, end % every name in every target (not just the last target's levels)
        a123vec=intersect(a123vec,fieldnames(FnsToEvaluate));
        for ff=1:length(a123vec)
            FnsToEvaluate_ACStats.(a123vec{ff})=FnsToEvaluate.(a123vec{ff});
        end
        % % age-conditional stats should be of length N_j [actually, no, they might be, e.g., QuantileMeans]
        % for ii=1:length(acsmomentsizes)
        %     if acsmomentsizes(ii)~=N_j
        %         errorstr=['Target Age-Conditional Stats must be of length() N_j (if you want to ignore some ages, use NaN for those ages); problem is with ', acsmomentnames{ii,1}, ' ', acsmomentnames{ii,2}, ' ',acsmomentnames{ii,3},' \n'];
        %         error(errorstr)
        %     end
        % end
    else
        % Placeholders
        acsmomentnames=cell(1,4);
        acscummomentsizes=0;
        ACStats_whichstats=zeros(7,1);
        FnsToEvaluate_ACStats=struct();
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
