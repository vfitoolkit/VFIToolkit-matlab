function StationaryDistKron=StationaryDist_InfHorz_IterationTan_noz_raw(StationaryDistKron,Policy_aprime,N_a,simoptions)
% Will treat the agents as being on a continuum of mass 1.
% Uses the improvement of: Tan (2020) - A fast and low computational memory algorithm for non-stochastic simulations in heterogeneous agent models

% Options needed
%  simoptions.tolerance
%  simoptions.maxit
%  simoptions.multiiter

% First, get Gamma
Policy_aprime=gather(reshape(Policy_aprime,[1,N_a]));

%% Use Tan improvement [not actually relevant without z variable]

StationaryDistKron=sparse(gather(StationaryDistKron));

% Gamma for first step of Tan improvement
Gammatranspose=sparse(Policy_aprime,1:1:N_a,ones(1,N_a),N_a,N_a);

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit
    
    % First step of Tan improvement
    StationaryDistKron=reshape(Gammatranspose*StationaryDistKron,[N_a]); %No point checking distance every single iteration. Do 100, then check.
    
    % Only check covergence every couple of iterations
    if rem(counter,simoptions.multiiter)==0
        StationaryDistKronOld=StationaryDistKron;
    elseif rem(counter,simoptions.multiiter)==10
        currdist=max(abs(StationaryDistKron-StationaryDistKronOld));
    end

    counter=counter+1;

    if simoptions.verbose==1
        if rem(counter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance ratio is %8.6f (currdist/tolerance, convergence when reaches 1) \n', counter, full(currdist)/simoptions.tolerance)            
        end
    end
end

%%
% Convert back to full matrix for output
StationaryDistKron=full(StationaryDistKron);

if ~(counter<simoptions.maxit)
    warning('SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end 

end
