function StationaryDist=StationaryDist_InfHorz_SemiExo_GI_raw(StationaryDist,Policy_dsemiz,Policy_aprime,PolicyProbUpper,N_dsemiz,N_a,N_semiz,N_z,pi_semiz,pi_z,simoptions)
% Will treat the agents as being on a continuum of mass 1.
% Uses the improvement of: Tan (2020) - A fast and low computational memory algorithm for non-stochastic simulations in heterogeneous agent models
% With semi-exogenous state semiz and markov z. The states are (a,semiz,z), with semiz varying first.
% Policy_dsemiz and Policy_aprime are [N_a*N_semiz*N_z,1], the (joint) index of the decision variable(s) that determine
% the semiz transitions, pi_semiz(semiz,semizprime,dsemiz), and the (joint) index of aprime.
%
% To do Tan improvement with semiz we treat the first step as (a,semiz,z) to (a',semiz',z) and then the second is the
% standard just updating z to z'. [As StationaryDist_FHorz_Iteration_SemiExo_raw, but without the age dimension.]
% With the grid interpolation layer: Policy_aprime is the lower grid index of aprime, and PolicyProbUpper the probability
% of the grid point above it (from the L2 index, with the L2flag override already applied), so in the first step each
% state goes to the two aprime points, each with every semizprime.

% Options needed
%  simoptions.tolerance
%  simoptions.maxit
%  simoptions.multiiter

%%
% It is likely that most of the elements in pi_semiz are zero, we can take advantage of this to speed things up.
% Ignore for a moment the dependence on d, and pretend it is just a N_semiz-by-N_semiz matrix. Then we can calculate
% N_semizshort=max(sum((pi_semiz>0),2)), the maximum number of non-zeros in any row of pi_semiz. And we then use this
% in place of N_semiz as the second dimension.
pi_semiz=gather(pi_semiz);
N_semizshort=max(max(sum((pi_semiz>0),2)));
% Create smaller version of pi_semiz that eliminates as many non-zeros as possible
[pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix
pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:);
idxshort=idx(:,end-N_semizshort+1:end,:);

N_bothz=N_semiz*N_z;
Policy_dsemiz=gather(reshape(Policy_dsemiz,[N_a*N_bothz,1]));
Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_bothz,1]));
PolicyProbUpper=gather(reshape(PolicyProbUpper,[N_a*N_bothz,1]));

semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1)+N_semiz*(0:1:N_semizshort-1)+(N_semiz*N_semizshort)*(Policy_dsemiz-1); % [N_a*N_bothz,N_semizshort], index for (semiz,semizprime(short),dsemiz) in pi_semiz_short and idxshort
zoffset=repelem(N_a*N_semiz*(0:1:N_z-1)',N_a*N_semiz,1); % [Tan improvement, z stays where it is in the first step]
Policy_aprimesemizz=repelem(Policy_aprime,1,N_semizshort)+N_a*(idxshort(semizindex_short)-1)+zoffset; % Note: add semiz' index, and z index for Tan improvement
semiztransitions=pi_semiz_short(semizindex_short);
% The two aprime points (lower and upper grid index), with their probabilities
Policy_aprimesemizz=[Policy_aprimesemizz,Policy_aprimesemizz+1];
semiztransitions=[(1-PolicyProbUpper).*semiztransitions,PolicyProbUpper.*semiztransitions];

II2=repelem((1:1:N_a*N_bothz)',1,2*N_semizshort);

%% Use Tan improvement
StationaryDist=sparse(gather(StationaryDist));

% Gamma for first step of Tan improvement
Gammatranspose=sparse(Policy_aprimesemizz,II2,semiztransitions,N_a*N_bothz,N_a*N_bothz); % From (a,semiz,z) to (a',semiz',z)
% pi_z for second step of Tan improvement
pi_z=sparse(gather(pi_z));

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit

    % First step of Tan improvement
    StationaryDist=reshape(Gammatranspose*StationaryDist,[N_a*N_semiz,N_z]); %No point checking distance every single iteration. Do 100, then check.
    % Second step of Tan improvement
    StationaryDist=reshape(StationaryDist*pi_z,[N_a*N_bothz,1]);

    % Only check convergence every couple of iterations
    if rem(counter,simoptions.multiiter)==0
        StationaryDistKronOld=StationaryDist;
    elseif rem(counter,simoptions.multiiter)==10
        currdist=max(abs(StationaryDist-StationaryDistKronOld));
    end

    counter=counter+1;

    if simoptions.verbose==1
        if rem(counter,50)==0
            fprintf('StationaryDist_InfHorz: after %i iterations the current distance ratio is %8.6f (currdist/tolerance, convergence when reaches 1) \n', counter, full(currdist)/simoptions.tolerance)
        end
    end
end

%%
% Convert back to full matrix for output
StationaryDist=full(StationaryDist);

if ~(counter<simoptions.maxit)
    warning('StationaryDist_InfHorz stopped due to reaching simoptions.maxit, this might be causing a problem')
end

end
