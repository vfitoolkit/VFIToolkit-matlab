function [InputNames,OutputNames,parsefailed]=ParseParametrizeParamsFn(ParametrizeParamsFn,Parameters)
% Discovers which fields of Parameters a ParametrizeParamsFn reads
% (InputNames) and a conservative superset of the fields it writes
% (OutputNames). The estimation/calibration commands use OutputNames to
% decide whether objects set up from parameters (shock processes via
% ExogShockFn/EiidShockFn, etc.) depend on the parameters being
% estimated: with a ParametrizeParamsFn the dependence can run through
% DERIVED parameters, so name-matching the estimated parameter names
% alone misses it.
%
% Method: call ParametrizeParamsFn on an initially EMPTY structure inside
% try/catch. Each failure names a missing field; that field is added with
% its actual value from Parameters, and the call retried, until it runs
% clean. The fields that had to be added are the inputs. The fields of
% the (minimal-input) output are the writes plus the inputs; OutputNames
% is their union, conservatively treating every input as possibly
% transformed. A second call on the full Parameters catches new fields
% created only inside branches that the minimal structure did not reach
% (e.g. behind isfield() guards).
%
% KNOWN LIMITATION: a field that already exists in Parameters and is
% OVERWRITTEN only inside such a guarded branch is invisible to both
% passes (nothing errors, and the field is not new). If such a field
% feeds a shock process, set estimoptions.ParametrizeParamsFn_OutputNames
% (or estimoptions.calibrateshocks=1) yourself.
%
% parsefailed=1 whenever the probe cannot complete (an error other than a
% missing field, a missing field that is not in Parameters, or the
% function failing on the full Parameters). The caller should then treat
% the outputs as unknown (conservatively assume dependence). InputNames
% and OutputNames hold whatever was discovered before the failure.
%
% Note: ParametrizeParamsFn must be a pure parameter map (no side
% effects); the commands call it at every objective evaluation anyway, so
% this is already required. It is called here up to #inputs+2 times.

InputNames=cell(1,0);
OutputNames=cell(1,0);
parsefailed=0;

FullParamNames=fieldnames(Parameters);
maxiter=length(FullParamNames)+2;

ProbeParams=struct();
Out=[];
for probe_c=1:maxiter
    try
        Out=ParametrizeParamsFn(ProbeParams);
        break
    catch probeME
        if ~strcmp(probeME.identifier,'MATLAB:nonExistentField')
            parsefailed=1; % errored on something other than a missing field: cannot parse
            return
        end
        % The message names the missing field in quotes, e.g.
        %   Unrecognized field name "gamma".          (R2019b and later)
        %   Reference to non-existent field 'gamma'.  (earlier releases)
        nametoken=regexp(probeME.message,'["'']([^"'']+)["'']','tokens','once');
        if isempty(nametoken)
            parsefailed=1; % could not read the field name out of the error message
            return
        end
        missingname=nametoken{1};
        if ~isfield(Parameters,missingname)
            parsefailed=1; % it reads a field Parameters does not have (a subfield of a struct parameter, or a user bug)
            return
        elseif any(strcmp(InputNames,missingname))
            parsefailed=1; % already supplied this field yet it is reported missing again: cannot parse
            return
        end
        InputNames{1,end+1}=missingname;
        ProbeParams.(missingname)=Parameters.(missingname);
    end
end
if ~isstruct(Out)
    parsefailed=1; % never ran clean within the iteration cap (or did not return a structure)
    return
end

% Writes observed on the minimal input, plus (conservatively) the inputs
OutputNames=union(InputNames,fieldnames(Out)');
OutputNames=OutputNames(:)';

% Second pass on the full Parameters: catches fields only created inside
% branches the minimal structure did not reach
try
    Out2=ParametrizeParamsFn(Parameters);
catch
    parsefailed=1; % it fails on the actual Parameters: the estimation itself is going to fail, but that is for the caller to discover with a readable error
    return
end
OutputNames=union(OutputNames,setdiff(fieldnames(Out2),FullParamNames));
OutputNames=OutputNames(:)';

end
