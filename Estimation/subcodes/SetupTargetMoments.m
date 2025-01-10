function [targetmomentvec,usingallstats,usinglcp, allstatmomentnames,allstatcummomentsizes,AllStats_whichstats, acsmomentnames, acscummomentsizes, ACStats_whichstats]=SetupTargetMoments(TargetMoments,useptype)
% useptype is 0 or 1

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

if any(~strcmp(fieldnames(TargetMoments),'AllStats') .* ~strcmp(fieldnames(TargetMoments),'AgeConditionalStats') )
    warning('TargetMoments seems to be set up incorrect: it has a field which is neither AllStats nor AgeConditionalStats')
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
    end


    % Second, do those in AgeConditionalStats
    if usinglcp==1
        acsmomentnames=cell(1,3);
        acsmomentcounter=0;
        acsmomentsizes=0;
        a1vec=fieldnames(TargetMoments.AgeConditionalStats); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1})); % These will be Mean, etc
            for a2=1:length(a2vec)
                if isstruct(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}))
                    a3vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2})); % These will be Mean, etc
                    for a3=1:length(a3vec)
                        acsmomentcounter=acsmomentcounter+1;
                        if size(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}),2)==1 % already column vector
                            targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})]; % append to end
                        else
                            targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3})']; % transpose, then append to end
                        end
                        acsmomentnames(acsmomentcounter,:)={a1vec{a1},a2vec{a2},a3vec{a3}};
                        acsmomentsizes(acsmomentcounter)=length(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}).(a3vec{a3}));
                    end
                else
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
        for aa=2:3
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
                if ACStats_whichstats(4)==0 % Avoid overwriting if it is 2 from LorenzCurve
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
                ACStats_whichstats(5)=1;
            end
            if any(strcmp(acsmomentnames(:,aa),'MoreInequality'))
                ACStats_whichstats(7)=1;
            end
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
        acsmomentnames=cell(1,3);
        acscummomentsizes=0;
        ACStats_whichstats=zeros(7,1);
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
                ACStats_whichstats(5)=1;
            end
            if any(strcmp(acsmomentnames(:,aa),'MoreInequality'))
                ACStats_whichstats(7)=1;
            end
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
    end

end



end

