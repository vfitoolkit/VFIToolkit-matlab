function [VKron, Policy]=ValueFnIter_InfHorz_SemiExo_noz_raw(VKron, N_d1, N_d2, N_a, N_semiz, pi_semiz, DiscountFactorParamsVec, ReturnMatrix, Howards, MaxHowards, Tolerance, maxiter)
% Value fn iteration, with Howards improvement iterations (a.k.a. modified Policy function iteration)
% With semi-exogenous state semiz, no z, and decision variables d1 and d2, where d2 determines the transition
% probabilities of semiz: pi_semiz(semiz,semizprime,d2). This is purediscretization (without refinement of d1).
% VKron is a-by-semiz, ReturnMatrix is (d1,d2,aprime)-by-a-by-semiz
% Policy is returned as [d1; d2; aprime] (indexes)
%
% The expectation of next period value, conditional on semiz and d2, is
%   EV(aprime,semiz,d2) = sum_{semizprime} pi_semiz(semiz,semizprime,d2) * V(aprime,semizprime)
% (for every d2 at once).

ReturnMatrix=reshape(ReturnMatrix,[N_d1,N_d2,N_a,N_a,N_semiz]);

pi_semiz_alt=permute(pi_semiz,[4,2,1,5,3]); % 1-by-semizprime-by-semiz-by-1-by-d2

% For Howards we need
addindexforstate=gpuArray(N_d1*N_d2*N_a*(0:1:N_a*N_semiz-1)'); % index for (a,semiz), for indexing ReturnMatrix at (d1,d2,aprime)
semizind=gpuArray(repelem((1:1:N_semiz)',N_a,1)); % semiz of each state (a,semiz)
EVcolindex=N_a*gpuArray(0:1:N_semiz-1); % [1,N_semiz], index for semizprime in V (which is aprime-by-semizprime)
Pindex=semizind+N_semiz*gpuArray(0:1:N_semiz-1); % [N_a*N_semiz,N_semiz], index for (semiz,semizprime) in pi_semiz

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on semiz and d2
    EV=reshape(VKronold,[N_a,N_semiz]).*pi_semiz_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over semizprime, aprime-by-1-by-semiz-by-1-by-d2
    EV=permute(EV,[6,5,1,2,3,4]); % 1-by-d2-by-aprime-by-1-by-semiz

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % d1 by d2 by aprime by a by semiz

    % Calc the max and it's index
    [VKron,Policy]=max(reshape(entireRHS,[N_d1*N_d2*N_a,N_a*N_semiz]),[],1);
    VKron=reshape(VKron,[N_a,N_semiz]);

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
        Policy=Policy(:); % (d1,d2,aprime) index for each state (a,semiz)
        Ftemp=ReturnMatrix(Policy+addindexforstate); % keep return function of optimal policy for using in Howards
        Policy_d2=rem(ceil(Policy/N_d1)-1,N_d2)+1;
        Policy_aprime=ceil(Policy/(N_d1*N_d2));
        EVindex=Policy_aprime+EVcolindex; % [N_a*N_semiz,N_semiz], index for (aprime,semizprime) in V
        Pdz=pi_semiz(Pindex+N_semiz*N_semiz*(Policy_d2-1)); % [N_a*N_semiz,N_semiz], pi_semiz(semiz,semizprime,d2) at the optimal d2

        for Howards_counter=1:Howards
            EVKrontemp=VKron(EVindex).*Pdz;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=sum(EVKrontemp,2); % sum over semizprime
            VKron=reshape(Ftemp+DiscountFactorParamsVec*EVKrontemp,[N_a,N_semiz]);
        end
    end

    tempcounter=tempcounter+1;

end

Policy=reshape(Policy,[1,N_a,N_semiz]);
Policy=[rem(Policy-1,N_d1)+1; rem(ceil(Policy/N_d1)-1,N_d2)+1; ceil(Policy/(N_d1*N_d2))]; % [d1; d2; aprime]

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end


end
