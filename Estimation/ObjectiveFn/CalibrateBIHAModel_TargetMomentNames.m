function momentnames=CalibrateBIHAModel_TargetMomentNames(actualtarget,allstatmomentnames,autocorrmomentnames,crosssecmomentnames,cmsmomentnames,allstatcummomentsizes,autocorrcummomentsizes,crossseccummomentsizes,cmscummomentsizes)
% CalibrateBIHAModel_TargetMomentNames builds labels for targeted moments.
% Inputs are moment-name metadata from SetupTargetMoments_InfHorz.

momentnames=cell(length(actualtarget),1);

for ii=1:length(actualtarget)
    momentnames{ii}=sprintf('moment_%i',ii);
end

offset=0;
momentnames=fillnames(momentnames,offset,allstatmomentnames,allstatcummomentsizes,'AllStats');
if ~isempty(allstatcummomentsizes)
    offset=offset+allstatcummomentsizes(end);
end
momentnames=fillnames(momentnames,offset,autocorrmomentnames,autocorrcummomentsizes,'AutoCorrTransProbs');
if ~isempty(autocorrcummomentsizes)
    offset=offset+autocorrcummomentsizes(end);
end
momentnames=fillnames(momentnames,offset,crosssecmomentnames,crossseccummomentsizes,'CrossSectionCovarCorr');
if ~isempty(crossseccummomentsizes)
    offset=offset+crossseccummomentsizes(end);
end
momentnames=fillnames(momentnames,offset,cmsmomentnames,cmscummomentsizes,'CustomModelStats');

end %end function

function momentnames=fillnames(momentnames,offset,rawnames,cummomentsizes,prefix)

if isempty(rawnames) || isempty(cummomentsizes)
    return
end
for rr=1:size(rawnames,1)
    if rr==1
        firstindex=1;
    else
        firstindex=cummomentsizes(rr-1)+1;
    end
    lastindex=cummomentsizes(rr);
    label=prefix;
    for cc=1:size(rawnames,2)
        if ~isempty(rawnames{rr,cc})
            label=[label,'.',rawnames{rr,cc}]; %#ok<AGROW>
        end
    end
    for ii=firstindex:lastindex
        momentnames{offset+ii}=label;
    end
end

end %end function
