function outputpanel=PSIDpanelmatrixfromVarNames(PSIDexplore,VarNamesAcrossYears)
% PSIDexplore is a PSID dataset imported/created using ImportPSIDdata() command.
%
% VarNamesAcrossYears should be a cell containing the names of the variables
% that correspond to a concept for a number of years, e.g., {'V1138','V1839','V2439'}

outputpanel=nan(length(PSIDexplore.(VarNamesAcrossYears{1}).value),length(VarNamesAcrossYears));

% Note: this is a slow loop, but you only import data once so who cares
for vv=1:length(VarNamesAcrossYears)
    currentdata=PSIDexplore.(VarNamesAcrossYears{vv}).value;
    for ii=1:length(currentdata)
        temp=str2double(currentdata(ii));
        if isempty(temp)
            outputpanel(ii,vv)=nan;
        else
            outputpanel(ii,vv)=temp;
        end
    end
end



end
