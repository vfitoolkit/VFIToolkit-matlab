function CalibrateBIHAModel_ProgressReport(caliboptions,currentObj,CalibParamNames,calibparamsvec,calibparamsvecindex,currentmomentvec,targetmomentvec,momentnames,GEPriceParamNames,Parameters,GEeqnNames,GeneralEqmConditionsVec)
% CalibrateBIHAModel_ProgressReport appends improvements to a text progress report.
% Inputs are already-computed calibration objects from the BIHA objective functions.

persistent bestObj evalCount

if isempty(bestObj)
    bestObj=Inf;
    evalCount=0;
end
evalCount=evalCount+1;

if currentObj>=bestObj
    return
end
bestObj=currentObj;

actualtarget=(~isnan(targetmomentvec));
fid=fopen(caliboptions.progressreportfilename,'a');
if fid==-1
    warning('Unable to open caliboptions.progressreportfilename for appending')
    return
end

fprintf(fid,'New best objective\n');
fprintf(fid,'Evaluation: %i\n',evalCount);
fprintf(fid,'Timestamp: %s\n',datestr(now,31));
fprintf(fid,'Objective: %.16g\n',currentObj);

fprintf(fid,'Parameter values:\n');
for pp=1:length(CalibParamNames)
    currparam=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
    fprintf(fid,'    %s:',CalibParamNames{pp});
    fprintf(fid,' %.16g',currparam);
    fprintf(fid,'\n');
end

fprintf(fid,'Model and target moments:\n');
for ii=1:length(actualtarget)
    if actualtarget(ii)
        if isempty(momentnames)
            momentname=sprintf('moment_%i',ii);
        else
            momentname=momentnames{ii};
        end
        fprintf(fid,'    %s: model %.16g, target %.16g\n',momentname,currentmomentvec(ii),targetmomentvec(ii));
    end
end

if ~isempty(GEPriceParamNames)
    fprintf(fid,'GE price values:\n');
    for ii=1:length(GEPriceParamNames)
        fprintf(fid,'    %s: %.16g\n',GEPriceParamNames{ii},Parameters.(GEPriceParamNames{ii}));
    end
end
if ~isempty(GEeqnNames)
    fprintf(fid,'General equilibrium residuals:\n');
    for ii=1:length(GEeqnNames)
        fprintf(fid,'    %s: %.16g\n',GEeqnNames{ii},GeneralEqmConditionsVec(ii));
    end
end
fprintf(fid,'\n');
fclose(fid);

end %end function
