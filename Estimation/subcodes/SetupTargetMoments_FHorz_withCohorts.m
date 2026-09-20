function [targetmomentvec,cohortmoments]=SetupTargetMoments_FHorz_withCohorts(TargetMoments,FnsToEvaluate,N_j,cohortagejshifter,useptype)
% Targets by cohort: TargetMoments.AgeConditionalStats.cohortC.(...) and TargetMoments.AllStats.cohortC.(...),
% where C=1,...,ncohorts indexes the entry ages in cohortagejshifter (an ordinal, not the age itself). Below the
% cohortC level the structure is exactly as without cohorts (fn.stat, or restriction.fn.stat). Age-conditional
% targets are N_j vectors and must be NaN before the cohort's entry age (the model has no mass there).
% Each cohort is parsed by SetupTargetMoments_FHorz() (useptype=0 or 1, as for the standard parser), and the moments
% are stacked cohort by cohort (within a cohort: AllStats, then AgeConditionalStats).
%
% cohortmoments holds, per cohort, everything CalibrateLifeCycleModel_withCohorts_objectivefn needs to place
% that cohort's moments: usingallstats(cc), usinglcp(cc), allstatmomentnames{cc}, allstatcummomentsizes{cc},
% AllStats_whichstats{cc}, FnsToEvaluate_AllStats{cc}, acsmomentnames{cc}, acscummomentsizes{cc},
% ACStats_whichstats{cc}, FnsToEvaluate_ACStats{cc}, and offset(cc) (start of the cohort's segment of targetmomentvec).

ncohorts=length(cohortagejshifter);

% Only cohortC names are allowed at the top level
for ss={'AllStats','AgeConditionalStats'}
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
    if isempty(fieldnames(TargetMoments_cc))
        error(['estimoptions.cohortagejshifter is being used, so targets must be by cohort, but there are no targets for ',cohortstr])
    end
    [targetmomentvec_cc,cohortmoments.usingallstats(cc),cohortmoments.usinglcp(cc),~, cohortmoments.allstatmomentnames{cc},cohortmoments.allstatcummomentsizes{cc},cohortmoments.AllStats_whichstats{cc},cohortmoments.FnsToEvaluate_AllStats{cc}, cohortmoments.acsmomentnames{cc}, cohortmoments.acscummomentsizes{cc}, cohortmoments.ACStats_whichstats{cc},cohortmoments.FnsToEvaluate_ACStats{cc}, ~,~]=SetupTargetMoments_FHorz(TargetMoments_cc,FnsToEvaluate,useptype);
    % Age-conditional targets before the cohort's entry age must be NaN
    if cohortmoments.usinglcp(cc)==1
        j0=cohortagejshifter(cc);
        temp=targetmomentvec_cc(cohortmoments.allstatcummomentsizes{cc}(end)+1:end); % the age-conditional part
        if mod(length(temp),N_j)~=0
            error(['TargetMoments.AgeConditionalStats.',cohortstr,': age-conditional targets must be vectors of length N_j (NaN for ages not targeted)'])
        end
        temp=reshape(temp,N_j,[]); % (age, moment)
        if any(~isnan(temp(1:j0-1,:)),'all')
            error(['TargetMoments.AgeConditionalStats.',cohortstr,' has non-NaN targets at ages before the cohort enters (age ',num2str(j0),'); these must be NaN'])
        end
    end
    cohortmoments.offset(cc)=length(targetmomentvec);
    targetmomentvec=[targetmomentvec; targetmomentvec_cc];
end

end
