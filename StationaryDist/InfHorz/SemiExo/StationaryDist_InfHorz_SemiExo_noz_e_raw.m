function StationaryDist=StationaryDist_InfHorz_SemiExo_noz_e_raw(StationaryDist,Policy_dsemiz,Policy_aprime,N_dsemiz,N_a,N_semiz,N_e,pi_semiz,pi_e,simoptions)
% Will treat the agents as being on a continuum of mass 1.
% Uses the improvement of: Tan (2020) - A fast and low computational memory algorithm for non-stochastic simulations in heterogeneous agent models
% With semi-exogenous state semiz and iid e, no z. The states are (a,semiz,e), with semiz varying first.
% Policy_dsemiz and Policy_aprime are [N_a*N_semiz*N_e,1], the (joint) index of the decision variable(s) that determine
% the semiz transitions, pi_semiz(semiz,semizprime,dsemiz), and the (joint) index of aprime.
% Without z there are two steps, from (a,semiz,e) to (a',semiz'), and then put e' back in (kron(pi_e,.), as e is iid).

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

Policy_dsemiz=gather(reshape(Policy_dsemiz,[N_a*N_semiz*N_e,1]));
Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_semiz*N_e,1]));

semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1)+N_semiz*(0:1:N_semizshort-1)+(N_semiz*N_semizshort)*(Policy_dsemiz-1); % [N_a*N_semiz*N_e,N_semizshort], index for (semiz,semizprime(short),dsemiz) in pi_semiz_short and idxshort
Policy_aprimesemiz=repelem(Policy_aprime,1,N_semizshort)+N_a*(idxshort(semizindex_short)-1); % Note: add semiz' index
semiztransitions=pi_semiz_short(semizindex_short);

II2=repelem((1:1:N_a*N_semiz*N_e)',1,N_semizshort);

%% Iterate
StationaryDist=sparse(gather(StationaryDist));

Gammatranspose=sparse(Policy_aprimesemiz,II2,semiztransitions,N_a*N_semiz,N_a*N_semiz*N_e); % From (a,semiz,e) to (a',semiz')
% pi_e for second step
pi_e=sparse(gather(pi_e));

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit

    StationaryDist=Gammatranspose*StationaryDist;
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
