function [VKron, Policy]=ValueFnIter_InfHorz_SemiExo_HowardGreedy_nod1_e_raw(VKron, N_d2, N_a, N_semiz, N_z, N_e, pi_z, pi_semiz, pi_e, DiscountFactorParamsVec, ReturnMatrix, MaxHowards, Tolerance, maxiter)
% Greedy Howards (a.k.a. Policy Function Iteration): Howards is done by solving the linear system for V given the
% policy, using the sparse transition matrix from (a,semiz,z,e) to (aprime,semizprime,zprime,eprime), which has probabilities
% pi_semiz(semiz,semizprime,d2)*pi_z(z,zprime)*pi_e(eprime) at the optimal d2.
% Value fn iteration, with Howards improvement iterations (a.k.a. modified Policy function iteration)
% With semi-exogenous state semiz, markov z and iid e. The only decision variable is d2, which determines the transition
% probabilities of semiz: pi_semiz(semiz,semizprime,d2). (With a d1, the d1 is refined out before getting here.)
% VKron is a-by-(semiz,z)-by-e, ReturnMatrix is (d2,aprime)-by-a-by-(semiz,z)-by-e [semiz varies first, then z, then e]
% pi_e is a column vector of probabilities
% Policy is returned as [d2; aprime] (indexes)
%
% The expectation of next period value, conditional on (semiz,z) and d2, is
%   EV(aprime,semiz,z,d2) = sum_{semizprime} pi_semiz(semiz,semizprime,d2) * sum_{zprime} pi_z(z,zprime) * sum_{eprime} pi_e(eprime) * V(aprime,semizprime,zprime,eprime)
% so the expectation over eprime is taken first, then over zprime, and then the one over semizprime (for every d2 at once).
% EV does not depend on e (it is iid), so it is the same for every e.

N_bothz=N_semiz*N_z;

ReturnMatrix=reshape(ReturnMatrix,[N_d2,N_a,N_a,N_semiz,N_z,N_e]);

pi_e_alt=shiftdim(pi_e,-3); % 1-by-1-by-1-by-eprime
pi_z_alt=shiftdim(pi_z',-2); % 1-by-1-by-zprime-by-z
pi_semiz_alt=permute(pi_semiz,[4,2,1,5,3]); % 1-by-semizprime-by-semiz-by-1-by-d2

% For Howards we need
addindexforstate=gpuArray(N_d2*N_a*(0:1:N_a*N_bothz*N_e-1)'); % index for (a,semiz,z,e), for indexing ReturnMatrix at (d2,aprime)
semizind=gpuArray(repmat(repelem((1:1:N_semiz)',N_a,1),N_z*N_e,1)); % semiz of each state (a,semiz,z,e)
zind=gpuArray(repmat(repelem((1:1:N_z)',N_a*N_semiz,1),N_e,1)); % z of each state (a,semiz,z,e)
Pindex=semizind+N_semiz*gpuArray(0:1:N_semiz-1); % [N_a*N_bothz*N_e,N_semiz], index for (semiz,semizprime) in pi_semiz
spI=gpuArray.speye(N_a*N_bothz*N_e);
stateind=gpuArray((1:1:N_a*N_bothz*N_e)'); % rows of the greedy Howards matrix, (a,semiz,z,e)
greedycolindex=gpuArray(reshape(N_a*(0:1:N_semiz-1)'+N_a*N_semiz*(0:1:N_z-1)+N_a*N_semiz*N_z*shiftdim(0:1:N_e-1,-1),[1,N_semiz*N_z*N_e])); % (semizprime,zprime,eprime), semizprime fastest
greedyprobze=repelem(reshape(pi_e,[1,N_e]),1,N_semiz*N_z); % [1,N_semiz*N_z*N_e], pi_e(eprime) for each column of greedycolindex

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on (semiz,z) and d2
    % First the expectation over eprime, which is iid
    EVe=reshape(VKronold,[N_a,N_semiz,N_z,N_e]).*pi_e_alt;
    EVe(isnan(EVe))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
    EVe=sum(EVe,4); % sum over eprime, aprime-by-semizprime-by-zprime
    % Then the expectation over zprime
    EVz=EVe.*pi_z_alt;
    EVz(isnan(EVz))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EVz=sum(EVz,3); % sum over zprime, aprime-by-semizprime-by-1-by-z
    % Then the expectation over semizprime, conditional on semiz and d2
    EV=reshape(EVz,[N_a,N_semiz,1,N_z]).*pi_semiz_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over semizprime, aprime-by-1-by-semiz-by-z-by-d2
    EV=permute(EV,[5,1,2,3,4]); % d2-by-aprime-by-1-by-semiz-by-z

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % d2 by aprime by a by semiz by z by e (EV autoexpands over a and e)

    % Calc the max and it's index
    [VKron,Policy]=max(reshape(entireRHS,[N_d2*N_a,N_a*N_bothz*N_e]),[],1);
    VKron=reshape(VKron,[N_a,N_bothz,N_e]);

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
        Policy=Policy(:); % (d2,aprime) index for each state (a,semiz,z,e)
        Ftemp=ReturnMatrix(Policy+addindexforstate); % keep return function of optimal policy for using in Howards
        Policy_d2=rem(Policy-1,N_d2)+1;
        Policy_aprime=ceil(Policy/N_d2);
        Pdz=pi_semiz(Pindex+N_semiz*N_semiz*(Policy_d2-1)); % [N_a*N_bothz*N_e,N_semiz], pi_semiz(semiz,semizprime,d2) at the optimal d2

        T_E=sparse(repmat(stateind,1,N_semiz*N_z*N_e),Policy_aprime+greedycolindex,repmat(Pdz,1,N_z*N_e).*repmat(repelem(pi_z(zind,:),1,N_semiz),1,N_e).*greedyprobze,N_a*N_bothz*N_e,N_a*N_bothz*N_e); % row (a,semiz,z,e) to column (aprime,semizprime,zprime,eprime)

        % Greedy Howards with -Inf in V: a direct solve is global, so one -Inf in Ftemp gives NaN everywhere, and a finite Ftemp at a
        % state whose policy puts positive weight on a dead state gives a finite V where the truth is -Inf. So solve on the live states
        % only: a state is dead if V is -Inf there or if its policy puts positive weight on a dead state (the closure under T_E);
        % (I-beta*T_E) is block triangular in live-then-dead order, so the live block solves exactly on its own and V is -Inf on the
        % dead set. (CoreInfHorzVFIAlgoTests doPart(11); InfHorzGI_InfNaN_proposal.md, section 8)
        dead=(VKron(:)==-Inf);
        if any(dead)
            newdead=dead;
            while any(newdead)
                newdead=(T_E*double(dead)>0) & ~dead;
                dead=dead | newdead;
            end
            live=~dead;
            N_live=gather(sum(live));
            livemap=cumsum(live);
            [T_Erow,T_Ecol,T_Eval]=find(T_E);
            keep=live(T_Erow) & live(T_Ecol); % a live row has no positive weight on a dead column, so this is just the live rows
            T_E=sparse(livemap(T_Erow(keep)),livemap(T_Ecol(keep)),T_Eval(keep),N_live,N_live);
            VKrontemp=-Inf(numel(Ftemp),1,'gpuArray');
            VKrontemp(live)=(gpuArray.speye(N_live)-DiscountFactorParamsVec*T_E)\Ftemp(live);
            VKron=VKrontemp;
        else
            VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
        end
        VKron=reshape(VKron,[N_a,N_bothz,N_e]);
    end

    tempcounter=tempcounter+1;

end

Policy=reshape(Policy,[1,N_a,N_bothz,N_e]);
Policy=[rem(Policy-1,N_d2)+1; ceil(Policy/N_d2)]; % [d2; aprime]

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end


end
