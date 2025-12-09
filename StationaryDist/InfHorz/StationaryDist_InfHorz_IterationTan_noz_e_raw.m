function StationaryDist=StationaryDist_InfHorz_IterationTan_noz_e_raw(StationaryDist,Policy_aprime,N_a,N_e,pi_e,simoptions)
% Will treat the agents as being on a continuum of mass 1.
% Uses the improvement of: Tan (2020) - A fast and low computational memory algorithm for non-stochastic simulations in heterogeneous agent models

% Options needed
%  simoptions.tolerance
%  simoptions.maxit
%  simoptions.multiiter

% Policy_aprime is currently [N_a,N_e]
Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_e,1])); % sparse() requires inputs to be 2-D

%% Use Tan improvement
% Cannot reshape() with sparse gpuArrays. [And not obvious how to do Tan improvement without reshape()]
% Using full gpuArrays is marginally slower than just spare cpu arrays, so no point doing that.
% Hence, just force sparse cpu arrays.

StationaryDist=sparse(gather(StationaryDist));

% Gamma for first step of Tan improvement
Gammatranspose=sparse(Policy_aprime,1:1:N_a*N_e,ones(1,N_a*N_e),N_a,N_a*N_e);
% pi_e
pi_e=sparse(gather(pi_e));

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit
    
    % First step of Tan improvement
    StationaryDist=Gammatranspose*StationaryDist; %No point checking distance every single iteration. Do 100, then check.
    
     % Put e back into dist
    StationaryDist=kron(pi_e,StationaryDist);

    % Only check covergence every couple of iterations
    if rem(counter,simoptions.multiiter)==0
        StationaryDistKronOld=StationaryDist;
    elseif rem(counter,simoptions.multiiter)==10
        currdist=max(abs(StationaryDist-StationaryDistKronOld));
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
StationaryDist=full(StationaryDist);

if ~(counter<simoptions.maxit)
    warning('SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end 

end
