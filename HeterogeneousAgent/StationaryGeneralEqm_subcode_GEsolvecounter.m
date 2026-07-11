function count=StationaryGeneralEqm_subcode_GEsolvecounter(cmd)
% Opt-in counter for the number of GE-condition evaluations (i.e. model
% solves: one full VFI + agent-dist + aggregates solve) performed inside the
% stationary general eqm solvers. Purely for benchmarking/reporting.
%
% It is inert during normal runs: the counting call inside the ..._subfn only
% fires when heteroagentoptions.countGEsolves==1. Usage from a caller:
%    StationaryGeneralEqm_subcode_GEsolvecounter('reset');
%    [p_eqm,...]=HeteroAgentStationaryEqm_InfHorz(...);   % heteroagentoptions.countGEsolves=1
%    nsolves=StationaryGeneralEqm_subcode_GEsolvecounter('get');
%
% cmd is one of 'reset' | 'incr' | 'get'; the current count is returned.
persistent count_internal
if isempty(count_internal)
    count_internal=0;
end
switch cmd
    case 'reset'
        count_internal=0;
    case 'incr'
        count_internal=count_internal+1;
    case 'get'
        % just return the current value
    otherwise
        error('StationaryGeneralEqm_subcode_GEsolvecounter: unrecognised command ''%s'' (use reset/incr/get)',cmd)
end
count=count_internal;

end
