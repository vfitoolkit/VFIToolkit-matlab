function [targetmomentvec,usingallstats,usingautocorr,usingcrosssec,usingcustomstats, allstatmomentnames,allstatcummomentsizes,AllStats_whichstats, FnsToEvaluate_AllStats, autocorrmomentnames, autocorrcummomentsizes, AutoCorrStats_whichstats, FnsToEvaluate_AutoCorrStats, crosssecmomentnames, crossseccummomentsizes, CrossSecStats_whichstats, FnsToEvaluate_CrossSecStats,cmsmomentnames, cmscummomentsizes, selectors, autocorrtimehorizons]=SetupTargetMoments_InfHorz(TargetMoments,FnsToEvaluate,useptype,simoptions,Names_i)
% Turns TargetMoments into the target vector and the bookkeeping the InfHorz calibration objective needs.
% useptype is 0 or 1. simoptions (optional; the calibration commands pass it) gives the conditional restrictions, whose names the
% targets may use, and switches on the building of 'selectors'; Names_i (optional, with useptype=1) the permanent types, whose names
% the targets may use (fn.type.stat), and which the selectors' trailing type dimension follows.
% Also divides FnsToEvaluate up into separate versions for each command (in FnsToEvaluate order).
%
% TargetMoments.AllStats.(fn).(stat), .(fn).MoreInequality.(substat), .(restriction).(fn).(stat)
% TargetMoments.AutoCorrTransProbs.(fn).(stat), .(restriction).(fn).(stat): stat is Mean, StdDeviation, AutoCorrelation, AutoCovariance or a
%     Pair byproduct (PairMass, PairMean_t, PairMean_tplusk, PairStdDeviation_t, PairStdDeviation_tplusk), each with an optional _kK
%     horizon suffix (K>=2; the command reports horizon K under .tperiodsK, the objective reads it from there). Scalars. TransitionProbs
%     cannot be targeted.
% TargetMoments.CrossSectionCovarCorr.(fn1).CovarianceWith.(fn2), .(fn1).CorrelationWith.(fn2), .(fn).Mean, .(fn).StdDeviation, the matrices
%     .CovarianceMatrix/.CorrelationMatrix ([nFns,nFns] over the reduced FnsToEvaluate, NaN omits an entry), each also under a
%     .(restriction). The self-correlation .(fn).CorrelationWith.(fn) is always one and cannot be targeted.
% TargetMoments.CustomModelStats.(name)
% NaN in a target omits that entry.
%
% Output 'selectors' holds the per-combination selectors for simoptions.whichcombos/whichstats of the three commands (used by
% CalibrateInfHorzAgentModel when caliboptions.whichcombos=1); it is an empty structure when simoptions was not given:
%   selectors.AllStats.whichcombos  [nFns,1+nRestr]        selectors.AllStats.whichstats  [nFns,1+nRestr,7]
%   selectors.AutoCorr.whichcombos  [nFns,1+nRestr]        (no per-combination whichstats: the command reports its byproducts)
%   selectors.CrossSec.whichcombos  [nFns,nFns,1+nRestr]   (pair-shaped, symmetric; the diagonal is a function's own Mean/StdDeviation)
% pages are the unrestricted targets first, then the restrictions in the fieldnames order of simoptions.conditionalrestrictions; nFns
% is the number of functions in that command's reduced FnsToEvaluate. 'autocorrtimehorizons' are the K>=2 the AutoCorr targets name.
% The 7-vectors AllStats_whichstats (and the 4-vectors AutoCorrStats_whichstats, CrossSecStats_whichstats, which the two commands do
% not read) are the union over all targets, used when caliboptions.whichcombos=0.
%
% Rewritten 2026-10-08 for the single-type case (CalibInfHorzTests): restricted AutoCorr/CrossSec targets, horizon suffixes, matrix
% targets, the selectors, validation of the names; and four defects: quantile targets switched on Minimum/Maximum (entry 5) instead
% of the quantiles (entry 6); the reduced FnsToEvaluate were built from the LAST first-level name's second level (so they depended on
% the field order of TargetMoments) and came out alphabetical (intersect sorts); CovarianceMatrix/CorrelationMatrix were refused.

% Only calculate each of AllStats and AutoCorrTransProbs and CrossSectionCovarCorr when being used (so as faster when not using them all)
if isfield(TargetMoments,'AllStats')
    usingallstats=1;
else
    usingallstats=0;
end
if isfield(TargetMoments,'AutoCorrTransProbs')
    usingautocorr=1;
else
    usingautocorr=0;
end
if isfield(TargetMoments,'CrossSectionCovarCorr')
    usingcrosssec=1;
else
    usingcrosssec=0;
end
if isfield(TargetMoments,'CustomModelStats')
    usingcustomstats=1;
else
    usingcustomstats=0;
end

temp=fieldnames(TargetMoments);
for a1=1:length(temp)
    if ~any(strcmp(temp{a1},{'AllStats','AutoCorrTransProbs','CrossSectionCovarCorr','CustomModelStats'}))
        fprintf(' \n')
        fprintf(' \n')
        fprintf('The following error is because %s is a field in TargetMoments but does not fit allowed formats (AllStats, AutoCorrTransProbs, CrossSectionCovarCorr, CustomModelStats) \n', temp{a1})
        error('TargetMoments contains a field with a problematic name (see line above)')
    end
end

if useptype==0
    % Restriction names (the page order of simoptions.whichcombos), when simoptions was given
    if exist('simoptions','var') && isfield(simoptions,'conditionalrestrictions')
        RestrNames=fieldnames(simoptions.conditionalrestrictions);
    else
        RestrNames={};
    end
    nRestr=length(RestrNames);
    FnNamesAll=fieldnames(FnsToEvaluate);
    buildselectors=double(exist('simoptions','var')==1);

    targetmomentvec=[]; % Can't preallocate as have no idea how big this will be (ends up a column vector)

    %% AllStats: targets .(fn).(stat), .(fn).MoreInequality.(substat), .(restriction).(fn).(stat)
    % Without simoptions a first-level name that is not a function is taken to be a restriction (as before), with simoptions it must be one.
    if usingallstats==1
        allstatmomentnames=cell(0,3);
        allstatmomentsizes=[];
        a1vec=fieldnames(TargetMoments.AllStats);
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}));
            for a2=1:length(a2vec)
                temp=TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2});
                if isstruct(temp)
                    a3vec=fieldnames(temp);
                    for a3=1:length(a3vec)
                        if isstruct(temp.(a3vec{a3}))
                            error(['TargetMoments.AllStats.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},' is a structure: a target has at most three levels (fn.stat, fn.MoreInequality.substat, or restriction.fn.stat)'])
                        end
                        targetmomentvec=[targetmomentvec; reshape(temp.(a3vec{a3}),[],1)];
                        allstatmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3}};
                        allstatmomentsizes(end+1)=numel(temp.(a3vec{a3}));
                    end
                else
                    targetmomentvec=[targetmomentvec; temp(:)];
                    allstatmomentnames(end+1,:)={a1vec{a1},a2vec{a2},''};
                    allstatmomentsizes(end+1)=numel(temp);
                end
            end
        end
        allstatcummomentsizes=cumsum(allstatmomentsizes);
        nrows=size(allstatmomentnames,1);
        rowfn=cell(nrows,1);
        rowstat=cell(nrows,1);
        rowpage=zeros(nrows,1);
        for cc=1:nrows
            a1=allstatmomentnames{cc,1}; a2=allstatmomentnames{cc,2}; a3=allstatmomentnames{cc,3};
            isfn=any(strcmp(FnNamesAll,a1));
            isrestr=any(strcmp(RestrNames,a1)) || (buildselectors==0 && ~isfn);
            if isfn && any(strcmp(RestrNames,a1))
                error(['TargetMoments.AllStats.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
            elseif isrestr
                if isempty(a3)
                    error(['TargetMoments.AllStats.',a1,'.',a2,': a target under a conditional restriction needs a function and then a statistic'])
                end
                if ~any(strcmp(FnNamesAll,a2))
                    error(['TargetMoments.AllStats.',a1,'.',a2,': ',a2,' is not one of the FnsToEvaluate'])
                end
                rowpage(cc)=1+find(strcmp(RestrNames,a1));
                rowfn{cc}=a2;
                rowstat{cc}=a3;
            elseif isfn
                if ~isempty(a3) && ~strcmp(a2,'MoreInequality')
                    error(['TargetMoments.AllStats.',a1,'.',a2,'.',a3,': a target is .(fn).(stat), .(fn).MoreInequality.(substat) or .(restriction).(fn).(stat)'])
                end
                rowpage(cc)=1;
                rowfn{cc}=a1;
                rowstat{cc}=a2;
            else
                error(['TargetMoments.AllStats.',a1,' is neither a FnsToEvaluate nor a conditional restriction in simoptions.conditionalrestrictions'])
            end
            if ~any(strcmp(rowstat{cc},{'Mean','Median','RatioMeanToMedian','Variance','StdDeviation','Gini','LorenzCurve','Minimum','Maximum','QuantileCutoffs','QuantileMeans','MoreInequality'}))
                error(['TargetMoments.AllStats: ',rowstat{cc},' is not a statistic of EvalFnOnAgentDist_AllStats_InfHorz (Mean, Median, RatioMeanToMedian, Variance, StdDeviation, Gini, LorenzCurve, Minimum, Maximum, QuantileCutoffs, QuantileMeans, MoreInequality.(substat))'])
            end
        end
        FnsToEvaluate_AllStats=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowfn,FnNamesAll{ff}))
                FnsToEvaluate_AllStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesA=fieldnames(FnsToEvaluate_AllStats);
        % The union whichstats (caliboptions.whichcombos=0) and, with simoptions, the per-combination selectors
        AllStats_whichstats=zeros(7,1);
        AllStats_whichcombos=zeros(length(FnNamesA),1+nRestr);
        AllStats_whichstatsArr=zeros(length(FnNamesA),1+nRestr,7);
        for cc=1:nrows
            if isempty(allstatmomentnames{cc,3})
                temp=TargetMoments.AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2});
            else
                temp=TargetMoments.AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2}).(allstatmomentnames{cc,3});
            end
            if all(isnan(temp(:)))
                continue % an omitted target selects nothing
            end
            stat=rowstat{cc};
            ff=find(strcmp(FnNamesA,rowfn{cc}));
            pp=rowpage(cc);
            AllStats_whichcombos(ff,pp)=1;
            if strcmp(stat,'Mean')
                sidx=1;
            elseif strcmp(stat,'Median')
                sidx=2;
            elseif strcmp(stat,'RatioMeanToMedian')
                sidx=[1,2];
            elseif strcmp(stat,'Variance') || strcmp(stat,'StdDeviation')
                sidx=3;
            elseif strcmp(stat,'Gini')
                sidx=[]; % Gini: code 3 (Gini without the Lorenz curve) unless a LorenzCurve target has already set a 1
                if AllStats_whichstats(4)==0
                    AllStats_whichstats(4)=3;
                end
                if AllStats_whichstatsArr(ff,pp,4)==0
                    AllStats_whichstatsArr(ff,pp,4)=3;
                end
            elseif strcmp(stat,'LorenzCurve')
                sidx=[];
                AllStats_whichstats(4)=1;
                AllStats_whichstatsArr(ff,pp,4)=1;
            elseif strcmp(stat,'Minimum') || strcmp(stat,'Maximum')
                sidx=5;
            elseif strcmp(stat,'QuantileCutoffs') || strcmp(stat,'QuantileMeans')
                sidx=6; % [was 5, the Minimum/Maximum entry, before 2026-10-08]
            elseif strcmp(stat,'MoreInequality')
                sidx=7;
            end
            AllStats_whichstats(sidx)=1;
            AllStats_whichstatsArr(ff,pp,sidx)=1;
        end
    else
        % Placeholders
        allstatmomentnames=cell(1,3);
        allstatcummomentsizes=0;
        AllStats_whichstats=zeros(7,1);
        FnsToEvaluate_AllStats=struct();
        AllStats_whichcombos=[];
        AllStats_whichstatsArr=[];
    end

    %% AutoCorrTransProbs (EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz): targets .(fn).(stat) and .(restriction).(fn).(stat); the stat
    % is Mean, StdDeviation, AutoCorrelation, AutoCovariance or a Pair byproduct, each with an optional _kK horizon suffix (K>=2). Scalars.
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
                error(['TargetMoments.AutoCorrTransProbs.',a1,' is neither a FnsToEvaluate nor a conditional restriction in simoptions.conditionalrestrictions (restricted targets need simoptions to be passed to SetupTargetMoments_InfHorz; CalibrateInfHorzAgentModel does)'])
            end
            if ~any(strcmp(FnNamesAll,rowfn{cc}))
                error(['TargetMoments.AutoCorrTransProbs: ',rowfn{cc},' is not one of the FnsToEvaluate'])
            end
        end
        FnsToEvaluate_AutoCorrStats=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowfn,FnNamesAll{ff}))
                FnsToEvaluate_AutoCorrStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesC=fieldnames(FnsToEvaluate_AutoCorrStats);
        autocorrtimehorizons=[];
        AutoCorrStats_whichstats=zeros(4,1); % (not read by the command; kept for the union form)
        AutoCorr_whichcombos=zeros(length(FnNamesC),1+nRestr);
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
            elseif ~any(strcmp(base,{'AutoCorrelation','AutoCovariance','PairMass','PairMean_t','PairMean_tplusk','PairStdDeviation_t','PairStdDeviation_tplusk'}))
                error(['TargetMoments.AutoCorrTransProbs: ',stat,' is not a targetable output (Mean, StdDeviation, AutoCorrelation, AutoCovariance and the Pair byproducts, each with an optional _kK horizon suffix; TransitionProbs cannot be targeted)'])
            end
            if K>=2
                autocorrtimehorizons=[autocorrtimehorizons,K];
            end
            if strcmp(base,'Mean')
                AutoCorrStats_whichstats(1)=1;
            elseif strcmp(base,'StdDeviation')
                AutoCorrStats_whichstats(2)=1;
            elseif strcmp(base,'AutoCovariance')
                AutoCorrStats_whichstats(3)=1;
            elseif strcmp(base,'AutoCorrelation')
                AutoCorrStats_whichstats(4)=1;
            end
            if isempty(autocorrmomentnames{cc,3})
                temp=TargetMoments.AutoCorrTransProbs.(autocorrmomentnames{cc,1}).(autocorrmomentnames{cc,2});
            else
                temp=TargetMoments.AutoCorrTransProbs.(autocorrmomentnames{cc,1}).(autocorrmomentnames{cc,2}).(autocorrmomentnames{cc,3});
            end
            if numel(temp)~=1
                error(['TargetMoments.AutoCorrTransProbs target ',rowfn{cc},' ',stat,' has ',num2str(numel(temp)),' entries, but the infinite-horizon statistic is a scalar'])
            end
            if ~isnan(temp)
                ff=find(strcmp(FnNamesC,rowfn{cc}));
                AutoCorr_whichcombos(ff,rowpage(cc))=1;
            end
        end
        autocorrtimehorizons=unique(autocorrtimehorizons);
    else
        % Placeholders
        autocorrmomentnames=cell(1,3);
        autocorrcummomentsizes=0;
        AutoCorrStats_whichstats=zeros(4,1);
        FnsToEvaluate_AutoCorrStats=struct();
        autocorrtimehorizons=[];
        AutoCorr_whichcombos=[];
    end

    %% CrossSectionCovarCorr (EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz): targets .(fn1).CovarianceWith.(fn2), .(fn1).CorrelationWith.(fn2),
    % .(fn).Mean, .(fn).StdDeviation, or the matrices .CovarianceMatrix / .CorrelationMatrix ([nFns,nFns] over the reduced FnsToEvaluate);
    % each also under a conditional restriction as .(restriction).(...). The self-correlation .(fn).(fn) is always one and cannot be
    % targeted. NaN omits an entry (a pair, a matrix entry). The selector is pair-shaped per page, [nFns,nFns,1+nRestr], symmetric, the
    % diagonal being the own stats. A matrix target is indexed over every FnsToEvaluate, so it makes the reduced set the full one.
    if usingcrosssec==1
        xnames=cell(0,4);
        xsizes=[];
        a1vec=fieldnames(TargetMoments.CrossSectionCovarCorr);
        for a1=1:length(a1vec)
            temp1=TargetMoments.CrossSectionCovarCorr.(a1vec{a1});
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
                                        error(['TargetMoments.CrossSectionCovarCorr.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},'.',a4vec{a4},' is a structure: a target has at most four levels (restriction.fn1.CovarianceWith.fn2)'])
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
        crosssecmomentnames=xnames;
        crossseccummomentsizes=cumsum(xsizes);
        nrows=size(xnames,1);
        rowfn1=cell(nrows,1);
        rowfn2=cell(nrows,1);
        rowpage=zeros(nrows,1);
        rowmatrix=zeros(nrows,1);
        CrossSecStats_whichstats=zeros(4,1); % (not read by the command; 1 Mean, 2 CovarianceWith, 3 StdDeviation, 4 CorrelationWith)
        for cc=1:nrows
            names=xnames(cc,:);
            names=names(~cellfun(@isempty,names));
            a1=names{1};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
            if isrestr && isfn
                error(['TargetMoments.CrossSectionCovarCorr.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
            elseif isrestr
                rowpage(cc)=1+find(strcmp(RestrNames,a1));
                rest=names(2:end);
                if isempty(rest)
                    error(['TargetMoments.CrossSectionCovarCorr.',a1,': a target under a conditional restriction needs a function (then Mean, StdDeviation, CovarianceWith.(fn2) or CorrelationWith.(fn2)) or CovarianceMatrix/CorrelationMatrix'])
                end
            else
                rowpage(cc)=1;
                rest=names;
            end
            b1=rest{1};
            if isscalar(rest) % a matrix
                if ~any(strcmp(b1,{'CovarianceMatrix','CorrelationMatrix'}))
                    error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': a target with no statistic must be CovarianceMatrix or CorrelationMatrix (or is this a restriction missing from simoptions.conditionalrestrictions? restricted targets need simoptions to be passed to SetupTargetMoments_InfHorz; CalibrateInfHorzAgentModel does)'])
                end
                rowmatrix(cc)=1;
            elseif ~any(strcmp(FnNamesAll,b1))
                error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': ',b1,' is not one of the FnsToEvaluate (nor a conditional restriction in simoptions.conditionalrestrictions, nor CovarianceMatrix/CorrelationMatrix)'])
            elseif any(strcmp(rest{2},{'CovarianceWith','CorrelationWith'}))
                if numel(rest)~=3 || ~any(strcmp(FnNamesAll,rest{3}))
                    error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': ',rest{2},' needs a second FnsToEvaluate name'])
                end
                if strcmp(rest{3},b1) && strcmp(rest{2},'CorrelationWith')
                    error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': the self-correlation is always one and cannot be targeted'])
                end
                rowfn1{cc}=b1;
                rowfn2{cc}=rest{3};
                if strcmp(rest{2},'CovarianceWith')
                    CrossSecStats_whichstats(2)=1;
                else
                    CrossSecStats_whichstats(4)=1;
                end
            elseif numel(rest)==2 && any(strcmp(rest{2},{'Mean','StdDeviation'}))
                rowfn1{cc}=b1;
                rowfn2{cc}=b1;
                if strcmp(rest{2},'Mean')
                    CrossSecStats_whichstats(1)=1;
                else
                    CrossSecStats_whichstats(3)=1;
                end
            else
                error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': the targets are Mean, StdDeviation, CovarianceWith.(fn2), CorrelationWith.(fn2), or CovarianceMatrix/CorrelationMatrix, each optionally under a conditional restriction'])
            end
        end
        FnsToEvaluate_CrossSecStats=struct();
        for ff=1:length(FnNamesAll)
            if any(rowmatrix==1) || any(strcmp(rowfn1,FnNamesAll{ff})) || any(strcmp(rowfn2,FnNamesAll{ff}))
                FnsToEvaluate_CrossSecStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesX=fieldnames(FnsToEvaluate_CrossSecStats);
        if any(rowmatrix==1)
            CrossSecStats_whichstats(:)=1;
        end
        CrossSec_whichcombos=zeros(length(FnNamesX),length(FnNamesX),1+nRestr);
        for cc=1:nrows
            temp=TargetMoments.CrossSectionCovarCorr;
            for kk=1:size(xnames,2)
                if ~isempty(xnames{cc,kk})
                    temp=temp.(xnames{cc,kk});
                end
            end
            pp=rowpage(cc);
            if rowmatrix(cc)==1 % a matrix: non-NaN entries select the pairs
                if ~isequal(size(temp),[length(FnNamesX),length(FnNamesX)])
                    error(['TargetMoments.CrossSectionCovarCorr target ',strjoin(xnames(cc,~cellfun(@isempty,xnames(cc,:))),'.'),' must be of size [',num2str(length(FnNamesX)),',',num2str(length(FnNamesX)),'] (number of FnsToEvaluate, twice)'])
                end
                on=double(~isnan(temp));
                CrossSec_whichcombos(:,:,pp)=max(CrossSec_whichcombos(:,:,pp),max(on,on')); % symmetric: the command reads the upper triangle, a lower-triangle entry alone must still select the pair
            else
                if numel(temp)~=1
                    error(['TargetMoments.CrossSectionCovarCorr target ',strjoin(xnames(cc,~cellfun(@isempty,xnames(cc,:))),'.'),' has ',num2str(numel(temp)),' entries, but the infinite-horizon statistic is a scalar'])
                end
                if ~isnan(temp)
                    f1=find(strcmp(FnNamesX,rowfn1{cc}));
                    f2=find(strcmp(FnNamesX,rowfn2{cc}));
                    CrossSec_whichcombos(f1,f2,pp)=1;
                    CrossSec_whichcombos(f2,f1,pp)=1;
                end
            end
        end
    else
        % Placeholders
        crosssecmomentnames=cell(1,4);
        crossseccummomentsizes=0;
        CrossSecStats_whichstats=zeros(4,1);
        FnsToEvaluate_CrossSecStats=struct();
        CrossSec_whichcombos=[];
    end

    %% A target that is NaN in full is dropped, with a warning (NaN is for omitting some entries of a target, matrix entries; a target
    % omitted in full should not be listed, and with caliboptions.whichcombos=1 nothing is computed for it; the objective skips it)
    nanchk_kinds={'AllStats','AutoCorrTransProbs','CrossSectionCovarCorr'};
    nanchk_using=[usingallstats,usingautocorr,usingcrosssec];
    nanchk_sofar=0;
    for nanchk_kk=1:3
        if nanchk_using(nanchk_kk)==1
            if nanchk_kk==1
                nanchk_names=allstatmomentnames; nanchk_cum=allstatcummomentsizes;
            elseif nanchk_kk==2
                nanchk_names=autocorrmomentnames; nanchk_cum=autocorrcummomentsizes;
            else
                nanchk_names=crosssecmomentnames; nanchk_cum=crossseccummomentsizes;
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
    %% The selectors for simoptions.whichcombos/whichstats (used by CalibrateInfHorzAgentModel when caliboptions.whichcombos=1)
    selectors=struct();
    if buildselectors==1
        selectors.AllStats.whichcombos=AllStats_whichcombos;
        selectors.AllStats.whichstats=AllStats_whichstatsArr;
        selectors.AutoCorr.whichcombos=AutoCorr_whichcombos;
        selectors.CrossSec.whichcombos=CrossSec_whichcombos;
    end

elseif useptype==1
    % With permanent types a target has two to five names: fn.stat (grouped), fn.type.stat, restriction.fn.stat (grouped),
    % restriction.fn.type.stat, fn.MoreInequality.substat (or with a type), and for the cross-section kind fn1.type.CovarianceWith.fn2
    % and restriction.fn1.type.CovarianceWith.fn2. The selectors carry a trailing type dimension of N_i+1: one slot per permanent type
    % in the order of Names_i, then the grouped stats (see the InfHorz PType stats commands). A grouped target selects the grouped
    % slot; the commands then compute every ptype's stats for that combination too, as the grouped Mean, StdDeviation, Minimum and
    % Maximum are built from them. Names_i (5th input) tells a type from a statistic; without it (the legacy call) a second-level
    % name that is not a known statistic is taken to be a type, and no selectors are built. [Rewritten 2026-10-09 (CalibInfHorzPTypeTests);
    % before, CalibrateInfHorzAgentModel_PType called this with useptype=0, so per-type names only worked by accident of the generic
    % extraction and four-level names errored.]
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
    buildselectors=double(exist('simoptions','var')==1 && N_i>0);
    targetmomentvec=[]; % Can't preallocate as have no idea how big this will be (ends up a column vector)

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
                                targetmomentvec=[targetmomentvec; temp4(:)];
                                allstatmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4}};
                                allstatmomentsizes(end+1)=numel(temp4);
                            end
                        else
                            targetmomentvec=[targetmomentvec; temp3(:)];
                            allstatmomentnames(end+1,:)={a1vec{a1},a2vec{a2},a3vec{a3},''};
                            allstatmomentsizes(end+1)=numel(temp3);
                        end
                    end
                else
                    targetmomentvec=[targetmomentvec; temp2(:)];
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
            elseif isrestr || (~isfn && buildselectors==0) % (without simoptions a first-level name that is not a function is taken to be a restriction, as before)
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
                error(['TargetMoments.AllStats: the target for ',rowfn{cc},' needs a statistic (e.g. .Mean)'])
            end
            istype=any(strcmp(Names_i,rest{1})) || (N_i==0 && ~any(strcmp(knownstats,rest{1})));
            if istype % a permanent type: its own output
                if N_i>0
                    rowslot(cc)=find(strcmp(Names_i,rest{1}));
                end
                rest=rest(2:end);
                if isempty(rest)
                    error(['TargetMoments.AllStats: the target for ',rowfn{cc},' of a permanent type needs a statistic after the type name (.(fn).(type).(stat))'])
                end
            else
                rowslot(cc)=N_i+1;
            end
            rowstat{cc}=rest{1};
            if ~any(strcmp(knownstats,rowstat{cc}))
                error(['TargetMoments.AllStats: ',rowstat{cc},' is neither a statistic of EvalFnOnAgentDist_AllStats_InfHorz_PType (Mean, Median, RatioMeanToMedian, Variance, StdDeviation, Gini, LorenzCurve, Minimum, Maximum, QuantileCutoffs, QuantileMeans, MoreInequality.(substat)) nor a permanent type in Names_i'])
            end
            if strcmp(rowstat{cc},'MoreInequality') && length(rest)~=2
                error(['TargetMoments.AllStats: the target for ',rowfn{cc},' MoreInequality needs one sub-statistic name (e.g. .MoreInequality.Top1share)'])
            elseif ~strcmp(rowstat{cc},'MoreInequality') && length(rest)~=1
                error(['TargetMoments.AllStats: the target for ',rowfn{cc},' ',rowstat{cc},' has too many names (a target is .(fn).(stat), .(fn).(type).(stat), .(restriction).(fn).(stat) or .(restriction).(fn).(type).(stat))'])
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
        AllStats_whichcombos=zeros(length(FnNamesA),1+nRestr,N_i+1);
        AllStats_whichstatsArr=zeros(length(FnNamesA),1+nRestr,N_i+1,7);
        for cc=1:nrows
            temp=TargetMoments.AllStats;
            for kk=1:size(allstatmomentnames,2)
                if ~isempty(allstatmomentnames{cc,kk})
                    temp=temp.(allstatmomentnames{cc,kk});
                end
            end
            rowallnan=all(isnan(temp(:))); % a target that is NaN in full selects nothing (dropped, with a warning below)
            stat=rowstat{cc};
            if strcmp(stat,'Mean')
                sidx=1;
            elseif strcmp(stat,'Median')
                sidx=2;
            elseif strcmp(stat,'RatioMeanToMedian')
                sidx=[1,2];
            elseif strcmp(stat,'Variance') || strcmp(stat,'StdDeviation')
                sidx=3;
            elseif strcmp(stat,'Gini') || strcmp(stat,'LorenzCurve')
                sidx=4;
            elseif strcmp(stat,'Minimum') || strcmp(stat,'Maximum')
                sidx=5;
            elseif strcmp(stat,'QuantileCutoffs') || strcmp(stat,'QuantileMeans')
                sidx=6;
            else % MoreInequality
                sidx=7;
            end
            % the union: Gini is code 3 unless a LorenzCurve target has set a 1
            for ss=sidx
                if ss==4
                    if strcmp(stat,'LorenzCurve')
                        AllStats_whichstats(4)=1;
                    elseif AllStats_whichstats(4)==0
                        AllStats_whichstats(4)=3;
                    end
                else
                    AllStats_whichstats(ss)=1;
                end
            end
            if buildselectors==1 && rowallnan==0
                ff=find(strcmp(FnNamesA,rowfn{cc}));
                pp=rowpage(cc);
                ss_slot=rowslot(cc);
                AllStats_whichcombos(ff,pp,ss_slot)=1;
                for ss=sidx
                    if ss==4
                        if strcmp(stat,'LorenzCurve')
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
    else
        % Placeholders
        allstatmomentnames=cell(1,4);
        allstatcummomentsizes=0;
        AllStats_whichstats=zeros(7,1);
        FnsToEvaluate_AllStats=struct();
        AllStats_whichcombos=[];
        AllStats_whichstatsArr=[];
    end

    %% AutoCorrTransProbs (EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz_PType): targets .(fn).(stat), .(fn).(type).(stat), .(restriction).(fn).(stat),
    % .(restriction).(fn).(type).(stat); the stat is Mean, StdDeviation, AutoCorrelation, AutoCovariance or a Pair byproduct, each with an
    % optional _kK horizon suffix (K>=2; the command reports horizon K under .tperiodsK, grouped or inside the type's struct). Scalars.
    if usingautocorr==1
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
        nrows=size(autocorrmomentnames,1);
        rowfn=cell(nrows,1);
        rowstat=cell(nrows,1);
        rowpage=zeros(nrows,1);
        rowslot=zeros(nrows,1);
        acstats={'Mean','StdDeviation','AutoCorrelation','AutoCovariance','PairMass','PairMean_t','PairMean_tplusk','PairStdDeviation_t','PairStdDeviation_tplusk'};
        for cc=1:nrows
            names=autocorrmomentnames(cc,:);
            names=names(~cellfun(@isempty,names));
            a1=names{1};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
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
                error(['TargetMoments.AutoCorrTransProbs.',a1,' is neither a FnsToEvaluate nor a conditional restriction in simoptions.conditionalrestrictions (restricted targets need simoptions to be passed to SetupTargetMoments_InfHorz; CalibrateInfHorzAgentModel_PType does)'])
            end
            if isempty(rest)
                error(['TargetMoments.AutoCorrTransProbs: the target for ',rowfn{cc},' needs a statistic (e.g. .AutoCorrelation)'])
            end
            tkbase=regexprep(rest{1},'_k\d+$','');
            istype=any(strcmp(Names_i,rest{1})) || (N_i==0 && ~any(strcmp(acstats,tkbase)));
            if istype
                if N_i>0
                    rowslot(cc)=find(strcmp(Names_i,rest{1}));
                end
                if length(rest)~=2
                    error(['TargetMoments.AutoCorrTransProbs: the target for ',rowfn{cc},' of type ',rest{1},' is .(fn).(type).(stat), one statistic name after the type'])
                end
                rowstat{cc}=rest{2};
            else
                rowslot(cc)=N_i+1;
                if length(rest)~=1
                    error(['TargetMoments.AutoCorrTransProbs: the target for ',rowfn{cc},' is .(fn).(stat) or .(fn).(type).(stat) (',rest{1},' is neither a permanent type nor the last name)'])
                end
                rowstat{cc}=rest{1};
            end
        end
        FnsToEvaluate_AutoCorrStats=struct();
        for ff=1:length(FnNamesAll)
            if any(strcmp(rowfn,FnNamesAll{ff}))
                FnsToEvaluate_AutoCorrStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesC=fieldnames(FnsToEvaluate_AutoCorrStats);
        autocorrtimehorizons=[];
        AutoCorrStats_whichstats=zeros(4,1); % (not read by the command; kept for the union form)
        AutoCorr_whichcombos=zeros(length(FnNamesC),1+nRestr,N_i+1);
        for cc=1:nrows
            stat=rowstat{cc};
            tk=regexp(stat,'_k(\d+)$','tokens','once');
            if ~isempty(tk)
                K=str2double(tk{1});
                base=stat(1:end-length(tk{1})-2);
            else
                K=1;
                base=stat;
            end
            if any(strcmp(stat,{'Mean','StdDeviation'}))
                K=0;
            elseif ~any(strcmp(base,{'AutoCorrelation','AutoCovariance','PairMass','PairMean_t','PairMean_tplusk','PairStdDeviation_t','PairStdDeviation_tplusk'}))
                error(['TargetMoments.AutoCorrTransProbs: ',stat,' is not a targetable output (Mean, StdDeviation, AutoCorrelation, AutoCovariance and the Pair byproducts, each with an optional _kK horizon suffix; TransitionProbs cannot be targeted)'])
            end
            if K>=2
                autocorrtimehorizons=[autocorrtimehorizons,K];
            end
            if strcmp(base,'Mean')
                AutoCorrStats_whichstats(1)=1;
            elseif strcmp(base,'StdDeviation')
                AutoCorrStats_whichstats(2)=1;
            elseif strcmp(base,'AutoCovariance')
                AutoCorrStats_whichstats(3)=1;
            elseif strcmp(base,'AutoCorrelation')
                AutoCorrStats_whichstats(4)=1;
            end
            temp=TargetMoments.AutoCorrTransProbs;
            for kk=1:size(autocorrmomentnames,2)
                if ~isempty(autocorrmomentnames{cc,kk})
                    temp=temp.(autocorrmomentnames{cc,kk});
                end
            end
            if numel(temp)~=1
                error(['TargetMoments.AutoCorrTransProbs target ',rowfn{cc},' ',stat,' has ',num2str(numel(temp)),' entries, but the infinite-horizon statistic is a scalar'])
            end
            if buildselectors==1 && ~isnan(temp)
                ff=find(strcmp(FnNamesC,rowfn{cc}));
                AutoCorr_whichcombos(ff,rowpage(cc),rowslot(cc))=1;
            end
        end
        autocorrtimehorizons=unique(autocorrtimehorizons);
    else
        % Placeholders
        autocorrmomentnames=cell(1,4);
        autocorrcummomentsizes=0;
        AutoCorrStats_whichstats=zeros(4,1);
        FnsToEvaluate_AutoCorrStats=struct();
        autocorrtimehorizons=[];
        AutoCorr_whichcombos=[];
    end

    %% CrossSectionCovarCorr (EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz_PType): grouped targets .(fn).Mean, .(fn).StdDeviation,
    % .(fn1).CovarianceWith.(fn2), .(fn1).CorrelationWith.(fn2), .CovarianceMatrix, .CorrelationMatrix; per-type targets .(fn).(type).Mean,
    % .(fn).(type).StdDeviation, .(fn1).(type).CovarianceWith.(fn2), .(fn1).(type).CorrelationWith.(fn2); each also under a conditional
    % restriction as .(restriction).(...) (up to five names). The self-correlation .(fn).(fn) is always one and cannot be targeted; nor can
    % the per-type matrices CovarianceMatrix_ptype/CorrelationMatrix_ptype (the command shrinks them to the functions evaluated for the
    % type): use the per-type pairs. Scalars, and [nFns,nFns] for a matrix (over the reduced FnsToEvaluate, which a matrix makes the full
    % set); NaN omits an entry. The selector is pair-shaped per page and slot, [nFns,nFns,1+nRestr,N_i+1], symmetric.
    if usingcrosssec==1
        xnames=cell(0,5);
        xsizes=[];
        a1vec=fieldnames(TargetMoments.CrossSectionCovarCorr);
        for a1=1:length(a1vec)
            temp1=TargetMoments.CrossSectionCovarCorr.(a1vec{a1});
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
                                                error(['TargetMoments.CrossSectionCovarCorr.',a1vec{a1},'.',a2vec{a2},'.',a3vec{a3},'.',a4vec{a4},'.',a5vec{a5},' is a structure: a target has at most five levels (restriction.fn1.type.CovarianceWith.fn2)'])
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
        crosssecmomentnames=xnames;
        crossseccummomentsizes=cumsum(xsizes);
        nrows=size(xnames,1);
        rowfn1=cell(nrows,1);
        rowfn2=cell(nrows,1);
        rowpage=zeros(nrows,1);
        rowslot=zeros(nrows,1);
        rowmatrix=zeros(nrows,1);
        CrossSecStats_whichstats=zeros(4,1); % (not read by the command; 1 Mean, 2 CovarianceWith, 3 StdDeviation, 4 CorrelationWith)
        for cc=1:nrows
            names=xnames(cc,:);
            names=names(~cellfun(@isempty,names));
            a1=names{1};
            isrestr=any(strcmp(RestrNames,a1));
            isfn=any(strcmp(FnNamesAll,a1));
            if isrestr && isfn
                error(['TargetMoments.CrossSectionCovarCorr.',a1,': this name is both a FnsToEvaluate and a conditional restriction, so the target is ambiguous'])
            elseif isrestr
                rowpage(cc)=1+find(strcmp(RestrNames,a1));
                rest=names(2:end);
                if isempty(rest)
                    error(['TargetMoments.CrossSectionCovarCorr.',a1,': a target under a conditional restriction needs a function (then a type and/or Mean, StdDeviation, CovarianceWith.(fn2) or CorrelationWith.(fn2)) or CovarianceMatrix/CorrelationMatrix'])
                end
            else
                rowpage(cc)=1;
                rest=names;
            end
            b1=rest{1};
            if isscalar(rest) % a grouped matrix
                if ~any(strcmp(b1,{'CovarianceMatrix','CorrelationMatrix'}))
                    error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': a target with no statistic must be CovarianceMatrix or CorrelationMatrix (or is this a restriction missing from simoptions.conditionalrestrictions?)'])
                end
                rowmatrix(cc)=1; rowslot(cc)=N_i+1;
                CrossSecStats_whichstats(:)=1;
            elseif any(strcmp(b1,{'CovarianceMatrix_ptype','CorrelationMatrix_ptype'}))
                error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': the per-type matrices cannot be targeted (the command shrinks them to the functions evaluated for the type); target the per-type pairs .(fn1).(type).CovarianceWith.(fn2)'])
            elseif ~any(strcmp(FnNamesAll,b1))
                error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': ',b1,' is not one of the FnsToEvaluate (nor a conditional restriction in simoptions.conditionalrestrictions, nor CovarianceMatrix/CorrelationMatrix)'])
            else
                if length(rest)<2
                    error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': the target needs Mean, StdDeviation, CovarianceWith.(fn2) or CorrelationWith.(fn2)'])
                end
                istype=any(strcmp(Names_i,rest{2})) || (N_i==0 && ~any(strcmp({'Mean','StdDeviation','CovarianceWith','CorrelationWith'},rest{2})) && ~any(strcmp(FnNamesAll,rest{2})));
                if istype % per type: .(fn).(type).(...)
                    if N_i>0
                        rowslot(cc)=find(strcmp(Names_i,rest{2}));
                    end
                    rest2=rest(3:end);
                else
                    rowslot(cc)=N_i+1;
                    rest2=rest(2:end);
                end
                if isempty(rest2)
                    error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': the target needs Mean, StdDeviation, CovarianceWith.(fn2) or CorrelationWith.(fn2) (after the type name for a per-type target)'])
                elseif any(strcmp(rest2{1},{'CovarianceWith','CorrelationWith'}))
                    if numel(rest2)~=2 || ~any(strcmp(FnNamesAll,rest2{2}))
                        error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': ',rest2{1},' needs a second FnsToEvaluate name'])
                    end
                    if strcmp(rest2{2},b1) && strcmp(rest2{1},'CorrelationWith')
                        error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': the self-correlation is always one and cannot be targeted'])
                    end
                    rowfn1{cc}=b1; rowfn2{cc}=rest2{2};
                    if strcmp(rest2{1},'CovarianceWith')
                        CrossSecStats_whichstats(2)=1;
                    else
                        CrossSecStats_whichstats(4)=1;
                    end
                elseif isscalar(rest2) && any(strcmp(rest2{1},{'Mean','StdDeviation'}))
                    rowfn1{cc}=b1; rowfn2{cc}=b1;
                    if strcmp(rest2{1},'Mean')
                        CrossSecStats_whichstats(1)=1;
                    else
                        CrossSecStats_whichstats(3)=1;
                    end
                elseif isscalar(rest2) && strcmp(rest2{1},b1)
                    error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': the self-correlation is always one and cannot be targeted'])
                else
                    error(['TargetMoments.CrossSectionCovarCorr.',strjoin(names,'.'),': the targets are Mean, StdDeviation, CovarianceWith.(fn2), CorrelationWith.(fn2) (grouped, or after a type name), or the grouped CovarianceMatrix/CorrelationMatrix, each optionally under a conditional restriction'])
                end
            end
        end
        FnsToEvaluate_CrossSecStats=struct();
        for ff=1:length(FnNamesAll)
            if any(rowmatrix==1) || any(strcmp(rowfn1,FnNamesAll{ff})) || any(strcmp(rowfn2,FnNamesAll{ff}))
                FnsToEvaluate_CrossSecStats.(FnNamesAll{ff})=FnsToEvaluate.(FnNamesAll{ff});
            end
        end
        FnNamesX=fieldnames(FnsToEvaluate_CrossSecStats);
        CrossSec_whichcombos=zeros(length(FnNamesX),length(FnNamesX),1+nRestr,N_i+1);
        for cc=1:nrows
            temp=TargetMoments.CrossSectionCovarCorr;
            for kk=1:size(xnames,2)
                if ~isempty(xnames{cc,kk})
                    temp=temp.(xnames{cc,kk});
                end
            end
            pp=rowpage(cc);
            ss_slot=rowslot(cc);
            if rowmatrix(cc)==1 % a matrix: non-NaN entries select the pairs (symmetrised: the command reads the upper triangle)
                if ~isequal(size(temp),[length(FnNamesX),length(FnNamesX)])
                    error(['TargetMoments.CrossSectionCovarCorr target ',strjoin(xnames(cc,~cellfun(@isempty,xnames(cc,:))),'.'),' must be of size [',num2str(length(FnNamesX)),',',num2str(length(FnNamesX)),'] (number of FnsToEvaluate, twice), NaN where an entry is not targeted'])
                end
                if buildselectors==1
                    on=double(~isnan(temp));
                    CrossSec_whichcombos(:,:,pp,ss_slot)=max(CrossSec_whichcombos(:,:,pp,ss_slot),max(on,on'));
                end
            else
                if numel(temp)~=1
                    error(['TargetMoments.CrossSectionCovarCorr target ',strjoin(xnames(cc,~cellfun(@isempty,xnames(cc,:))),'.'),' has ',num2str(numel(temp)),' entries, but the infinite-horizon statistic is a scalar'])
                end
                if buildselectors==1 && ~isnan(temp)
                    f1=find(strcmp(FnNamesX,rowfn1{cc}));
                    f2=find(strcmp(FnNamesX,rowfn2{cc}));
                    CrossSec_whichcombos(f1,f2,pp,ss_slot)=1;
                    CrossSec_whichcombos(f2,f1,pp,ss_slot)=1;
                end
            end
        end
    else
        % Placeholders
        crosssecmomentnames=cell(1,5);
        crossseccummomentsizes=0;
        CrossSecStats_whichstats=zeros(4,1);
        FnsToEvaluate_CrossSecStats=struct();
        CrossSec_whichcombos=[];
    end

    %% A target that is NaN in full is dropped, with a warning (as in the single-type case)
    nanchk_kinds={'AllStats','AutoCorrTransProbs','CrossSectionCovarCorr'};
    nanchk_using=[usingallstats,usingautocorr,usingcrosssec];
    nanchk_sofar=0;
    for nanchk_kk=1:3
        if nanchk_using(nanchk_kk)==1
            if nanchk_kk==1
                nanchk_names=allstatmomentnames; nanchk_cum=allstatcummomentsizes;
            elseif nanchk_kk==2
                nanchk_names=autocorrmomentnames; nanchk_cum=autocorrcummomentsizes;
            else
                nanchk_names=crosssecmomentnames; nanchk_cum=crossseccummomentsizes;
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

    %% The selectors (used by CalibrateInfHorzAgentModel_PType when caliboptions.whichcombos=1)
    selectors=struct();
    if buildselectors==1
        selectors.AllStats.whichcombos=AllStats_whichcombos;
        selectors.AllStats.whichstats=AllStats_whichstatsArr;
        selectors.AutoCorr.whichcombos=AutoCorr_whichcombos;
        selectors.CrossSec.whichcombos=CrossSec_whichcombos;
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
