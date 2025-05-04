function StationaryDistKron=StationaryDist_Case1_IterationTan_raw(StationaryDistKron,PolicyIndexesKron,N_d,N_a,N_z,pi_z,simoptions)
% Will treat the agents as being on a continuum of mass 1.
% Uses the improvement of: Tan (2020) - A fast and low computational memory algorithm for non-stochastic simulations in heterogeneous agent models

% Options needed
%  simoptions.tolerance
%  simoptions.maxit
%  simoptions.multiiter

% First, get Gamma
if N_d==0
    Policy_aprimez=PolicyIndexesKron+N_a*(0:1:N_z-1);
else
    Policy_aprimez=shiftdim(PolicyIndexesKron(2,:,:),1)+N_a*(0:1:N_z-1);
end
Policy_aprimez=gather(reshape(Policy_aprimez,[1,N_a*N_z]));

%% Use Tan improvement
% Cannot reshape() with sparse gpuArrays. [And not obvious how to do Tan improvement without reshape()]
% Using full gpuArrays is marginally slower than just spare cpu arrays, so no point doing that.
% Hence, just force sparse cpu arrays.

StationaryDistKron=sparse(gather(StationaryDistKron));

% Gamma for first step of Tan improvement
Gammatranspose=sparse(Policy_aprimez,1:1:N_a*N_z,ones(1,N_a*N_z),N_a*N_z,N_a*N_z);
% pi_z for second step of Tan improvement
pi_z=sparse(gather(pi_z));

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit
    
    % First step of Tan improvement
    StationaryDistKron=reshape(Gammatranspose*StationaryDistKron,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
    % Second step of Tan improvement
    StationaryDistKron=reshape(StationaryDistKron*pi_z,[N_a*N_z,1]);
    
    % Only check covergence every couple of iterations
    if rem(counter,simoptions.multiiter)==0
        StationaryDistKronOld=StationaryDistKron;
    elseif rem(counter,simoptions.multiiter)==10
        currdist=full(max(abs(StationaryDistKron-StationaryDistKronOld)));
    end

    counter=counter+1;

    if simoptions.verbose==1
        if rem(counter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance ratio is %8.6f (currdist/tolerance, convergence when reaches 1) \n', counter, currdist/simoptions.tolerance)            
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
