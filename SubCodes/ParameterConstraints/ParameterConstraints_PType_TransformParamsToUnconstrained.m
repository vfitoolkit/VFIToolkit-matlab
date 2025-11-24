function [calibparamsvec,caliboptions]=ParameterConstraints_PType_TransformParamsToUnconstrained(calibparamsvec,calibparamsvecindex,CalibParamNames,nCalibParamsFinder,caliboptions,constraintsbyname)
% Change from 'constrained' to 'unconstrained' parameters.
% Constrained parameters as the original model parameters.
%
% constraintsbyname=1, caliboptions currently contains the constraints by name, they will be converted to vector in output
%                  =0, caliboptions already has constraints as vector

% Works by type of constraint:
% Denote cparam as the constrained parameter uparam as the unconstrained parameter.
% So original problem will be in terms of cparam.
% ParameterConstraints_TransformParamsToUnconstrained: transforms cparam to uparam
% ParameterConstraints_TransformParamsToOriginal: transforms uparam to cparam
% 
% - Constrain parameter to be positive
%     uparam=exp(cparam)
%     cparam=log(uparam)
%
% - Constrain parameter to be zero-to-one
%     uparam=
%     cparam=1/(1+exp(-uparam))
% - Constrain parameter to be A-to-B [first converts to zero-to-one]
%     intermediateparam=(cparam-A)/(B-A)
%     uparam=
%     intermediateparam==1/(1+exp(-uparam))
%     cparam=A+(B-A)*intermediateparam
%
% The bits about '50' are to avoid numerical error when the number gets
% crazy big/small. It just cuts them off at 50, and sets penalty>0 so as
% you know that you shouldn't go to those regions.


%% Backup the parameter constraint names, so I can replace them with vectors
if constraintsbyname==1
    caliboptions.constrainpositivenames=caliboptions.constrainpositive;
    caliboptions.constrainpositive=zeros(length(CalibParamNames),1); % if equal 1, then that parameter is constrained to be positive
    caliboptions.constrain0to1names=caliboptions.constrain0to1;
    caliboptions.constrain0to1=zeros(length(CalibParamNames),1); % if equal 1, then that parameter is constrained to be 0 to 1
    caliboptions.constrainAtoBnames=caliboptions.constrainAtoB;
    caliboptions.constrainAtoB=zeros(length(CalibParamNames),1); % if equal 1, then that parameter is constrained to be 0 to 1

    if ~isempty(caliboptions.constrainAtoBnames)
        caliboptions.constrainAtoBlimitsnames=caliboptions.constrainAtoBlimits;
        caliboptions.constrainAtoBlimits=zeros(length(CalibParamNames),2); % rows are parameters, column is lower (A) and upper (B) bounds [row will be [0,0] is unconstrained]
    end

    for pp=1:length(CalibParamNames)
        % Check the name, and note the ones that need to be coverted
        if any(strcmp(caliboptions.constrainpositivenames,CalibParamNames{nCalibParamsFinder(pp,1)}))
            caliboptions.constrainpositive(pp)=1;
        end
        if any(strcmp(caliboptions.constrain0to1names,CalibParamNames{nCalibParamsFinder(pp,1)}))
            caliboptions.constrain0to1(pp)=1;
        end
        if any(strcmp(caliboptions.constrainAtoBnames,CalibParamNames{nCalibParamsFinder(pp,1)}))
            % For parameters A to B, I convert via 0 to 1
            caliboptions.constrain0to1(pp)=1;
            caliboptions.constrainAtoB(pp)=1;
            caliboptions.constrainAtoBlimits(pp,:)=caliboptions.constrainAtoBlimitsnames.(CalibParamNames{nCalibParamsFinder(pp,1)});
        end
    end
end


%% Convert the parameters
for pp=1:length(CalibParamNames)
    if caliboptions.constrainpositive(pp)==1
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=max(log(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))),-49.99);
        % Note, the max() is because otherwise p=0 returns -Inf. [Matlab evaluates exp(-50) as about 10^-22, I overrule and use exp(-50) as zero, so I set -49.99 here so solver can realise the boundary is there; not sure if this setting -49.99 instead of my -50 cutoff actually helps, but seems like it might so I have done it here].
    end
    if caliboptions.constrainAtoB(pp)==1
        % Constraint parameter to be A to B (by first converting to 0 to 1, and then treating it as contraint 0 to 1)
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))-caliboptions.constrainAtoBlimits(pp,1))/(caliboptions.constrainAtoBlimits(pp,2)-caliboptions.constrainAtoBlimits(pp,1));
        % x=(y-A)/(B-A), converts A-to-B y, into 0-to-1 x
        % And then the next if-statement converts this 0-to-1 into unconstrained
    end
    if caliboptions.constrain0to1(pp)==1
        % Constrain parameter to be 0 to 1 (be working with log(p/(1-p)), where p is parameter) then always take exp()/(1+exp()) before inputting to model
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=min(49.99,max(-49.99,  log(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))/(1-calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)))) ));
        % Note: the max() and min() are because otherwise p=0 or 1 returns -Inf or Inf [Matlab evaluates 1/(1+exp(-50)) as one, and 1/(1+exp(50)) as about 10^-22, so I overrule them as 1 and 0, so I set -49.99 here so solver can realise the boundary is there; not sure if this setting -49.99 instead of my -50 cutoff actually helps, but seems like it might so I have done it here].
    end
    if caliboptions.constrainpositive(pp)==1 && caliboptions.constrain0to1(pp)==1 % Double check of inputs
        fprinf(['Relating to following error message: Parameter ',num2str(pp),' of ',num2str(length(CalibParamNames))])
        error('You cannot constrain parameter twice (you are constraining one of the parameters using both caliboptions.constrainpositive and in one of caliboptions.constrain0to1 and caliboptions.constrainAtoB')
    end
end

%% Parameters are now in the 'unconstrainted' form.

end