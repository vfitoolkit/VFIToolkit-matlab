function [calibparamsvec,penalty]=ParameterConstraints_TransformParamsToOriginal(calibparamsvec,calibparamsvecindex,CalibParamNames,caliboptions)
% Change from 'unconstrained' to 'constrained' parameters.
% Constrained parameters as the original model parameters.

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


% Do any transformations of parameters before we say what they are
penalty=zeros(length(calibparamsvec),1); % Used to apply penalty to objective function when parameters try to leave restricted ranges
for pp=1:length(CalibParamNames)
    if caliboptions.constrainpositive(pp)==1 % Forcing this parameter to be positive
        temp=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        penalty((calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)))=abs(temp/50).*(temp<-51); % 1 if out of range [Note: 51, rather than 50, so penalty only hits once genuinely out of range]
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=exp(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)));
    elseif caliboptions.constrain0to1(pp)==1
        temp=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        penalty((calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)))=abs(temp/50).*((temp>51)+(temp<-51)); % 1 if out of range [Note: 51, rather than 50, so penalty only hits once genuinely out of range]
        % Constrain parameter to be 0 to 1 (be working with x=log(p/(1-p)), where p is parameter) then always take 1/(1+exp(-x)) before inputting to model
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=1/(1+exp(-calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))));
        % Note: This does not include the endpoints of 0 and 1 as 1/(1+exp(-x)) maps from the Real line into the open interval (0,1)
        %       R is not compact, and [0,1] is compact, so cannot have a continuous bijection (one-to-one and onto) function from R into [0,1].
        %       So I settle for a function from R to (0,1) and then trim ends of R to give 0 and 1, like I do for constrainpositive I use +-50 as the cutoffs
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)).*(temp>-50); % set values less than -50 to zero
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)).*(1-(temp>50))+(temp>50); % set values greater than 50 to one
    end
    % Note: sometimes, need to do both of constrainAtoB and constrain0to1, so cannot use elseif
    if caliboptions.constrainAtoB(pp)==1
        % Constrain parameter to be A to B
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=caliboptions.constrainAtoBlimits(pp,1)+(caliboptions.constrainAtoBlimits(pp,2)-caliboptions.constrainAtoBlimits(pp,1))*calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        % Note, this parameter will have first been converted to 0 to 1 already, so just need to further make it A to B
        % y=A+(B-A)*x, converts 0-to-1 x, into A-to-B y
    end
end
if sum(penalty)>0
    penalty=1/prod(1./penalty(penalty>0)); % Turn into a scalar penalty [I try to do opposite of geometric mean, and penalize more when one gets extreme]
else
    penalty=0;
end

%% Parameters are now in the 'constrainted' form.


end