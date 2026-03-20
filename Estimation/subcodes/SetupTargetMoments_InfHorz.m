function [targetmomentvec,usingallstats,usingautocorr,usingcrosssec, allstatmomentnames,allstatcummomentsizes,AllStats_whichstats, FnsToEvaluate_AllStats, autocorrmomentnames, autocorrcummomentsizes, AutoCorrStats_whichstats, FnsToEvaluate_AutoCorrStats, crosssecmomentnames, crossseccummomentsizes, CrossSecStats_whichstats, FnsToEvaluate_CrossSecStats]=SetupTargetMoments_InfHorz(TargetMoments,FnsToEvaluate,useptype)
% useptype is 0 or 1
% Also divides FnsToEvaluate up into separate versions for each command

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

if any(~strcmp(fieldnames(TargetMoments),'AllStats') .* ~strcmp(fieldnames(TargetMoments),'AutoCorrTransProbs')  .* ~strcmp(fieldnames(TargetMoments),'CrossSectionCovarCorr') )
    warning('TargetMoments seems to be set up incorrect: it has a field which is neither AllStats nor AutoCorrTransProbs nor CrossSectionCovarCorr')
end

if useptype==0
    % conditionalrestrictions means we need the third level a3vec

    % Get all of the moments out of TargetMoments and make them into a vector
    % Also, store all the names
    targetmomentvec=[]; % Can't preallocate as have no idea how big this will be
    % Ends up a colmumn vector
    
    % First, do those in AllStats
    if usingallstats==1
        allstatmomentnames=cell(1,3);
        allstatmomentcounter=0;
        allstatmomentsizes=0;
        a1vec=fieldnames(TargetMoments.AllStats); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}));% These will be Mean, etc
                    for a3=1:length(a3vec)
                        allstatmomentcounter=allstatmomentcounter+1;
                        if size(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}),2)==1 % already column vector
                            targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})]; % append to end
                        else
                            targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})']; % transpose, then append to end
                        end
                        allstatmomentnames(allstatmomentcounter,:)={a1vec{a1},a2vec{a2},a3vec{a3}};
                        allstatmomentsizes(allstatmomentcounter)=length(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));
                    end
                else
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
        for aa=2:3
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
                AllStats_whichstats(5)=1;
            end
            if any(strcmp(allstatmomentnames(:,aa),'MoreInequality'))
                AllStats_whichstats(7)=1;
            end
        end
        % To do AllStats faster, just evaluate the relevant functions
        FnsToEvaluate_AllStats=struct();
        % Put a1vec and a2vec together, then find just those which are in FnsToEvaluate
        a12vec=[a1vec;a2vec];
        a12vec=intersect(a12vec,fieldnames(FnsToEvaluate));
        for ff=1:length(a12vec)
            FnsToEvaluate_AllStats.(a12vec{ff})=FnsToEvaluate.(a12vec{ff});
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


    % Second, do those in AutoCorrTransProbs
    if usingautocorr==1
        autocorrmomentnames=cell(1,3);
        autocorrmomentcounter=0;
        autocorrmomentsizes=0;
        a1vec=fieldnames(TargetMoments.AutoCorrTransProbs); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AutoCorrTransProbs.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}));% These will be Mean, etc
                    for a3=1:length(a3vec)
                        autocorrmomentcounter=autocorrmomentcounter+1;
                        if size(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}),2)==1 % already column vector
                            targetmomentvec=[targetmomentvec; TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})]; % append to end
                        else
                            targetmomentvec=[targetmomentvec; TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})']; % transpose, then append to end
                        end
                        autocorrmomentnames(autocorrmomentcounter,:)={a1vec{a1},a2vec{a2},a3vec{a3}};
                        autocorrmomentsizes(autocorrmomentcounter)=length(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));
                    end
                else
                    autocorrmomentcounter=autocorrmomentcounter+1;
                    if size(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}),2)==1 % already column vector
                        targetmomentvec=[targetmomentvec; TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2})]; % append to end
                    else
                        targetmomentvec=[targetmomentvec; TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2})']; % transpose, then append to end
                    end
                    autocorrmomentnames(autocorrmomentcounter,1:2)={a1vec{a1},a2vec{a2}};
                    autocorrmomentsizes(autocorrmomentcounter)=length(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}));
                end
            end
        end
        autocorrcummomentsizes=cumsum(autocorrmomentsizes); % Note: this is zero is AllStats is unused
        % To do AutoCorrTransProbs faster, we use simoptions.whichstats so that we only compute the stats we want.
        AutoCorrStats_whichstats=zeros(4,1);
        for aa=2:3
            if any(strcmp(autocorrmomentnames(:,aa),'Mean'))
                AutoCorrStats_whichstats(1)=1;
            end
            if any(strcmp(autocorrmomentnames(:,aa),'StdDeviation'))
                AutoCorrStats_whichstats(2)=1;
            end
            if any(strcmp(autocorrmomentnames(:,aa),'AutoCovariance'))
                AutoCorrStats_whichstats(3)=1;
            end
            if any(strcmp(autocorrmomentnames(:,aa),'AutoCorrelation'))
                AutoCorrStats_whichstats(4)=1;
            end
        end
        % To do AllStats faster, just evaluate the relevant functions
        FnsToEvaluate_AutoCorrStats=struct();
        % Put a1vec and a2vec together, then find just those which are in FnsToEvaluate
        a12vec=[a1vec;a2vec];
        a12vec=intersect(a12vec,fieldnames(FnsToEvaluate));
        for ff=1:length(a12vec)
            FnsToEvaluate_AutoCorrStats.(a12vec{ff})=FnsToEvaluate.(a12vec{ff});
        end
    else
        % Placeholders
        autocorrmomentnames=cell(1,3);
        autocorrcummomentsizes=0;
        AutoCorrStats_whichstats=zeros(4,1);
        FnsToEvaluate_AutoCorrStats=struct();
    end

    % Third, do those in CrossSectionCovarCorr
    if usingcrosssec==1
        crosssecmomentnames=cell(1,4);
        crosssecmomentcounter=0;
        crosssecmomentsizes=0;
        a1vec=fieldnames(TargetMoments.CrossSectionCovarCorr); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}));% These will be Mean, etc
                    for a3=1:length(a3vec)
                        if isstruct(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}))
                            a4vec=fieldnames(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));% These will be Mean, etc
                            for a4=1:length(a4vec)
                                crosssecmomentcounter=crosssecmomentcounter+1;
                                if size(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}),2)==1 % already column vector
                                    targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4})]; % append to end
                                else
                                    targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4})']; % transpose, then append to end
                                end
                                crosssecmomentnames(crosssecmomentcounter,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4}};
                                crosssecmomentsizes(crosssecmomentcounter)=length(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}));
                            end
                        else
                            crosssecmomentcounter=crosssecmomentcounter+1;
                            if size(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}),2)==1 % already column vector
                                targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})]; % append to end
                            else
                                targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})']; % transpose, then append to end
                            end
                            crosssecmomentnames(crosssecmomentcounter,1:3)={a1vec{a1},a2vec{a2},a3vec{a3}};
                            crosssecmomentsizes(crosssecmomentcounter)=length(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));
                        end
                    end
                else
                    crosssecmomentcounter=crosssecmomentcounter+1;
                    if size(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}),2)==1 % already column vector
                        targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2})]; % append to end
                    else
                        targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2})']; % transpose, then append to end
                    end
                    crosssecmomentnames(crosssecmomentcounter,1:2)={a1vec{a1},a2vec{a2}};
                    crosssecmomentsizes(crosssecmomentcounter)=length(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}));
                end
            end
        end
        crossseccummomentsizes=cumsum(crosssecmomentsizes); % Note: this is zero is CrossSectionCovarCorr is unused
        % To do CrossSectionCovarCorr faster, we use simoptions.whichstats so that we only compute the stats we want.
        CrossSecStats_whichstats=zeros(4,1);
        for aa=2:3
            if any(strcmp(crosssecmomentnames(:,aa),'Mean'))
                CrossSecStats_whichstats(1)=1;
            end
            if any(strcmp(crosssecmomentnames(:,aa),'CorrelationWith'))
                CrossSecStats_whichstats(2)=1;
            end
            if any(strcmp(crosssecmomentnames(:,aa),'StdDeviation'))
                CrossSecStats_whichstats(3)=1;
            end
            if any(strcmp(crosssecmomentnames(:,aa),'CorrelationWith'))
                CrossSecStats_whichstats(4)=1;
            end
        end
        % To do AllStats faster, just evaluate the relevant functions
        FnsToEvaluate_CrossSecStats=struct();
        % Put a1vec and a2vec together, then find just those which are in FnsToEvaluate
        a1234vec=[a1vec;a2vec;a3vec];
        a1234vec=intersect(a1234vec,fieldnames(FnsToEvaluate));
        for ff=1:length(a1234vec)
            FnsToEvaluate_CrossSecStats.(a1234vec{ff})=FnsToEvaluate.(a1234vec{ff});
        end
        if any(strcmp(a1vec,{'CovarianceMatrix'}))
            error('TargetMoments.CrossSectionCovarCorr is not allowed to contain CovarianceMatrix as a target (you can target the individual covariances)')
        end
        if any(strcmp(a1vec,{'CorrelationMatrix'}))
            error('TargetMoments.CrossSectionCovarCorr is not allowed to contain CorrelationMatrix as a target (you can target the individual correlations)')
        end
        
    else
        % Placeholders
        crosssecmomentnames=cell(1,4);
        crossseccummomentsizes=0;
        CrossSecStats_whichstats=zeros(4,1);
        FnsToEvaluate_CrossSecStats=struct();
    end


elseif useptype==1
    % PType means we need the third level a3vec
    % conditionalrestrictions means we need the third level a3vec
    % To allow conditionalrestrictions and ptype at once we go to fourth level a4vec

    % Get all of the moments out of TargetMoments and make them into a vector
    % Also, store all the names
    targetmomentvec=[]; % Can't preallocate as have no idea how big this will be
    % Ends up a colmumn vector (create row vector, then transpose)

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
                AllStats_whichstats(5)=1;
            end
            if any(strcmp(allstatmomentnames(:,aa),'MoreInequality'))
                AllStats_whichstats(7)=1;
            end
        end
        % To do AllStats faster, just evaluate the relevant functions
        FnsToEvaluate_AllStats=struct();
        % Put a1vec through a3vec together, then find just those which are in FnsToEvaluate
        a123vec=[a1vec;a2vec;a3vec];
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
        allstatmomentnames=cell(1,4);
        allstatcummomentsizes=0;
        AllStats_whichstats=zeros(7,1);
        FnsToEvaluate_AllStats=struct();
    end



    % Second, do those in AutoCorrTransProbs
    if usingautocorr==1
        autocorrmomentnames=cell(1,3);
        autocorrmomentcounter=0;
        autocorrmomentsizes=0;
        a1vec=fieldnames(TargetMoments.AutoCorrTransProbs); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AutoCorrTransProbs.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}));% These will be Mean, etc
                    for a3=1:length(a3vec)
                        if isstruct(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}))
                            a4vec=fieldnames(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));% These will be Mean, etc
                            for a4=1:length(a4vec)
                                autocorrmomentcounter=autocorrmomentcounter+1;
                                if size(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}),2)==1 % already column vector
                                    targetmomentvec=[targetmomentvec; TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4})]; % append to end
                                else
                                    targetmomentvec=[targetmomentvec; TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4})']; % transpose, then append to end
                                end
                                autocorrmomentnames(autocorrmomentcounter,:)={a1vec{a1},a2vec{a2},a3vec{a3},(a4vec{a4})};
                                autocorrmomentsizes(autocorrmomentcounter)=length(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}));
                            end
                        else
                            autocorrmomentcounter=autocorrmomentcounter+1;
                            if size(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}),2)==1 % already column vector
                                targetmomentvec=[targetmomentvec; TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})]; % append to end
                            else
                                targetmomentvec=[targetmomentvec; TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})']; % transpose, then append to end
                            end
                            autocorrmomentnames(autocorrmomentcounter,1:3)={a1vec{a1},a2vec{a2},a3vec{a3}};
                            autocorrmomentsizes(autocorrmomentcounter)=length(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));
                        end
                    end
                else
                    autocorrmomentcounter=autocorrmomentcounter+1;
                    if size(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}),2)==1 % already column vector
                        targetmomentvec=[targetmomentvec; TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2})]; % append to end
                    else
                        targetmomentvec=[targetmomentvec; TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2})']; % transpose, then append to end
                    end
                    autocorrmomentnames(autocorrmomentcounter,1:2)={a1vec{a1},a2vec{a2}};
                    autocorrmomentsizes(autocorrmomentcounter)=length(TargetMoments.AutoCorrTransProbs.(a1vec{a1}).(a2vec{a2}));
                end
            end
        end
        autocorrcummomentsizes=cumsum(autocorrmomentsizes); % Note: this is zero is AllStats is unused
        % To do AutoCorrTransProbs faster, we use simoptions.whichstats so that we only compute the stats we want.
        AutoCorrStats_whichstats=zeros(4,1);
        for aa=2:4
            if any(strcmp(autocorrmomentnames(:,aa),'Mean'))
                AutoCorrStats_whichstats(1)=1;
            end
            if any(strcmp(autocorrmomentnames(:,aa),'StdDeviation'))
                AutoCorrStats_whichstats(2)=1;
            end
            if any(strcmp(autocorrmomentnames(:,aa),'AutoCovariance'))
                AutoCorrStats_whichstats(3)=1;
            end
            if any(strcmp(autocorrmomentnames(:,aa),'AutoCorrelation'))
                AutoCorrStats_whichstats(4)=1;
            end
        end
        % To do AllStats faster, just evaluate the relevant functions
        FnsToEvaluate_AutoCorrStats=struct();
        % Put a1vec through a3vec together, then find just those which are in FnsToEvaluate
        a123vec=[a1vec;a2vec;a3vec];
        a123vec=intersect(a123vec,fieldnames(FnsToEvaluate));
        for ff=1:length(a123vec)
            FnsToEvaluate_AutoCorrStats.(a123vec{ff})=FnsToEvaluate.(a123vec{ff});
        end
    else
        % Placeholders
        autocorrmomentnames=cell(1,4);
        autocorrcummomentsizes=0;
        AutoCorrStats_whichstats=zeros(4,1);
        FnsToEvaluate_AutoCorrStats=struct();
    end

    % Third, do those in CrossSectionCovarCorr
    if usingcrosssec==1
        crosssecmomentnames=cell(1,4);
        crosssecmomentcounter=0;
        crosssecmomentsizes=0;
        a1vec=fieldnames(TargetMoments.CrossSectionCovarCorr); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}));% These will be Mean, etc
                    for a3=1:length(a3vec)
                        if isstruct(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}))
                            a4vec=fieldnames(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));% These will be Mean, etc
                            for a4=1:length(a4vec)
                                if isstruct(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}))
                                    a4vec=fieldnames(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}));% These will be Mean, etc
                                    for a5=1:length(a5vec)
                                        crosssecmomentcounter=crosssecmomentcounter+1;
                                        if size(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}).(a5vec{a5}),2)==1 % already column vector
                                            targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}).(a5vec{a5})]; % append to end
                                        else
                                            targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}).(a5vec{a5})']; % transpose, then append to end
                                        end
                                        crosssecmomentnames(crosssecmomentcounter,:)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4},a5vec{a5}};
                                        crosssecmomentsizes(crosssecmomentcounter)=length(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}).(a5vec{a5}));
                                    end
                                else
                                    crosssecmomentcounter=crosssecmomentcounter+1;
                                    if size(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}),2)==1 % already column vector
                                        targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4})]; % append to end
                                    else
                                        targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4})']; % transpose, then append to end
                                    end
                                    crosssecmomentnames(crosssecmomentcounter,1:4)={a1vec{a1},a2vec{a2},a3vec{a3},a4vec{a4}};
                                    crosssecmomentsizes(crosssecmomentcounter)=length(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}).(a4vec{a4}));
                                end
                            end
                        else
                            crosssecmomentcounter=crosssecmomentcounter+1;
                            if size(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}),2)==1 % already column vector
                                targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})]; % append to end
                            else
                                targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})']; % transpose, then append to end
                            end
                            crosssecmomentnames(crosssecmomentcounter,1:3)={a1vec{a1},a2vec{a2},a3vec{a3}};
                            crosssecmomentsizes(crosssecmomentcounter)=length(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));
                        end
                    end
                else
                    crosssecmomentcounter=crosssecmomentcounter+1;
                    if size(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}),2)==1 % already column vector
                        targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2})]; % append to end
                    else
                        targetmomentvec=[targetmomentvec; TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2})']; % transpose, then append to end
                    end
                    crosssecmomentnames(crosssecmomentcounter,1:2)={a1vec{a1},a2vec{a2}};
                    crosssecmomentsizes(crosssecmomentcounter)=length(TargetMoments.CrossSectionCovarCorr.(a1vec{a1}).(a2vec{a2}));
                end
            end
        end
        crossseccummomentsizes=cumsum(crosssecmomentsizes); % Note: this is zero is CrossSectionCovarCorr is unused
        % To do CrossSectionCovarCorr faster, we use simoptions.whichstats so that we only compute the stats we want.
        CrossSecStats_whichstats=zeros(4,1);
        for aa=2:4
            if any(strcmp(crosssecmomentnames(:,aa),'Mean'))
                CrossSecStats_whichstats(1)=1;
            end
            if any(strcmp(crosssecmomentnames(:,aa),'CorrelationWith'))
                CrossSecStats_whichstats(2)=1;
            end
            if any(strcmp(crosssecmomentnames(:,aa),'StdDeviation'))
                CrossSecStats_whichstats(3)=1;
            end
            if any(strcmp(crosssecmomentnames(:,aa),'CorrelationWith'))
                CrossSecStats_whichstats(4)=1;
            end
        end
        % To do AllStats faster, just evaluate the relevant functions
        FnsToEvaluate_CrossSecStats=struct();
        % Put a1vec and a2vec together, then find just those which are in FnsToEvaluate
        a1234vec=[a1vec;a2vec;a3vec;a4vec];
        a1234vec=intersect(a1234vec,fieldnames(FnsToEvaluate));
        for ff=1:length(a1234vec)
            FnsToEvaluate_CrossSecStats.(a1234vec{ff})=FnsToEvaluate.(a1234vec{ff});
        end
        if any(strcmp(a1vec,{'CovarianceMatrix'}))
            error('TargetMoments.CrossSectionCovarCorr is not allowed to contain CovarianceMatrix as a target (you can target the individual covariances)')
        end
        if any(strcmp(a1vec,{'CorrelationMatrix'}))
            error('TargetMoments.CrossSectionCovarCorr is not allowed to contain CorrelationMatrix as a target (you can target the individual correlations)')
        end
        
    else
        % Placeholders
        crosssecmomentnames=cell(1,5);
        crossseccummomentsizes=0;
        CrossSecStats_whichstats=zeros(4,1);
        FnsToEvaluate_CrossSecStats=struct();
    end
end



end

