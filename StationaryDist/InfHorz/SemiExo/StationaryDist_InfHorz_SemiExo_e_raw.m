function StationaryDist=StationaryDist_InfHorz_SemiExo_e_raw(StationaryDist,Policy_dsemiz,Policy_aprime,N_dsemiz,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z,pi_e,simoptions)
% Will treat the agents as being on a continuum of mass 1.
% Uses the improvement of: Tan (2020) - A fast and low computational memory algorithm for non-stochastic simulations in heterogeneous agent models
% With semi-exogenous state semiz, markov z and iid e. The states are (a,semiz,z,e), with semiz varying first, then z, then e.
% Policy_dsemiz and Policy_aprime are [N_a*N_semiz*N_z*N_e,1], the (joint) index of the decision variable(s) that determine
% the semiz transitions, pi_semiz(semiz,semizprime,dsemiz), and the (joint) index of aprime.
%
% To do Tan improvement with semiz we treat the first step as (a,semiz,z,e) to (a',semiz',z), the second is the
% standard just updating z to z', and the third puts e' back in (kron(pi_e,.), as e is iid).
% [As StationaryDist_InfHorz_SemiExo_raw, with the e step of StationaryDist_InfHorz_IterationTan_e_raw.]

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
Policy_dsemiz=gather(reshape(Policy_dsemiz,[N_a*N_bothz*N_e,1]));
Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_bothz*N_e,1]));

semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_z*N_e,1)+N_semiz*(0:1:N_semizshort-1)+(N_semiz*N_semizshort)*(Policy_dsemiz-1); % [N_a*N_bothz*N_e,N_semizshort], index for (semiz,semizprime(short),dsemiz) in pi_semiz_short and idxshort
zoffset=repmat(repelem(N_a*N_semiz*(0:1:N_z-1)',N_a*N_semiz,1),N_e,1); % [Tan improvement, z stays where it is in the first step, e goes]
Policy_aprimesemizz=repelem(Policy_aprime,1,N_semizshort)+N_a*(idxshort(semizindex_short)-1)+zoffset; % Note: add semiz' index, and z index for Tan improvement
semiztransitions=pi_semiz_short(semizindex_short);

II2=repelem((1:1:N_a*N_bothz*N_e)',1,N_semizshort);

%% Use Tan improvement
StationaryDist=sparse(gather(StationaryDist));

% Gamma for first step of Tan improvement
Gammatranspose=sparse(Policy_aprimesemizz,II2,semiztransitions,N_a*N_bothz,N_a*N_bothz*N_e); % From (a,semiz,z,e) to (a',semiz',z)
% pi_z for second step of Tan improvement
pi_z=sparse(gather(pi_z));
% pi_e for third step
pi_e=sparse(gather(pi_e));

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit

    % First step of Tan improvement
    StationaryDist=reshape(Gammatranspose*StationaryDist,[N_a*N_semiz,N_z]); %No point checking distance every single iteration. Do 100, then check.
    % Second step of Tan improvement
    StationaryDist=reshape(StationaryDist*pi_z,[N_a*N_bothz,1]);
    % Put e back into dist
    StationaryDist=kron(pi_e,StationaryDist);

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
