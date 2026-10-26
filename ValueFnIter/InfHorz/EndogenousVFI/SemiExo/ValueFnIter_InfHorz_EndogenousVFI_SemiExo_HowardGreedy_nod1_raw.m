function [VKron, Policy]=ValueFnIter_InfHorz_EndogenousVFI_SemiExo_HowardGreedy_nod1_raw(VKron, N_d2, N_a, N_semiz, N_z, pi_z, pi_semiz, DiscountFactorParamsVec, ReturnMatrix, MaxHowards, Tolerance, maxiter)
% Endogenous VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_SemiExo_HowardGreedy_nod1_raw, but every loop stops on the
% span over (a,semiz) of V-Vold within each (z,e) (tolerances apply to that span), and V is recovered in levels after
% each fine-grid loop.
% Greedy Howards (a.k.a. Policy Function Iteration): Howards is done by solving the linear system for V given the
% policy, using the sparse transition matrix from (a,semiz,z) to (aprime,semizprime,zprime), which has probabilities
% pi_semiz(semiz,semizprime,d2)*pi_z(z,zprime) at the optimal d2.
% Value fn iteration, with Howards improvement iterations (a.k.a. modified Policy function iteration)
% With semi-exogenous state semiz and markov z. The only decision variable is d2, which determines the transition
% probabilities of semiz: pi_semiz(semiz,semizprime,d2). (With a d1, the d1 is refined out before getting here.)
% VKron is a-by-(semiz,z), ReturnMatrix is (d2,aprime)-by-a-by-(semiz,z) [semiz varies first, then z]
% Policy is returned as [d2; aprime] (indexes)
%
% The expectation of next period value, conditional on (semiz,z) and d2, is
%   EV(aprime,semiz,z,d2) = sum_{semizprime} pi_semiz(semiz,semizprime,d2) * sum_{zprime} pi_z(z,zprime) * V(aprime,semizprime,zprime)
% so the expectation over zprime is taken first, and then the one over semizprime (for every d2 at once).

N_bothz=N_semiz*N_z;

ReturnMatrix=reshape(ReturnMatrix,[N_d2,N_a,N_a,N_semiz,N_z]);

pi_z_alt=shiftdim(pi_z',-2); % 1-by-1-by-zprime-by-z
pi_semiz_alt=permute(pi_semiz,[4,2,1,5,3]); % 1-by-semizprime-by-semiz-by-1-by-d2

% For Howards we need
addindexforstate=gpuArray(N_d2*N_a*(0:1:N_a*N_bothz-1)'); % index for (a,semiz,z), for indexing ReturnMatrix at (d2,aprime)
semizind=gpuArray(repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1)); % semiz of each state (a,semiz,z)
zind=gpuArray(repelem((1:1:N_z)',N_a*N_semiz,1)); % z of each state (a,semiz,z)
EVcolindex=N_a*gpuArray(0:1:N_semiz-1)+N_a*N_semiz*(zind-1); % [N_a*N_bothz,N_semiz], index for (semizprime,z) in EVz (which is aprime-by-semizprime-by-z)
Pindex=semizind+N_semiz*gpuArray(0:1:N_semiz-1); % [N_a*N_bothz,N_semiz], index for (semiz,semizprime) in pi_semiz
spI=gpuArray.speye(N_a*N_bothz);
stateind=gpuArray((1:1:N_a*N_bothz)'); % rows of the greedy Howards matrix, (a,semiz,z)
greedycolindex=gpuArray(reshape(N_a*(0:1:N_semiz-1)'+N_a*N_semiz*(0:1:N_z-1),[1,N_semiz*N_z])); % (semizprime,zprime), semizprime fastest

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on (semiz,z) and d2
    % First the expectation over zprime
    EVz=reshape(VKronold,[N_a,N_semiz,N_z]).*pi_z_alt;
    EVz(isnan(EVz))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EVz=sum(EVz,3); % sum over zprime, aprime-by-semizprime-by-1-by-z
    % Then the expectation over semizprime, conditional on semiz and d2
    EV=reshape(EVz,[N_a,N_semiz,1,N_z]).*pi_semiz_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over semizprime, aprime-by-1-by-semiz-by-z-by-d2
    EV=permute(EV,[5,1,2,3,4]); % d2-by-aprime-by-1-by-semiz-by-z

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % d2 by aprime by a by semiz by z

    % Calc the max and it's index
    [VKron,Policy]=max(reshape(entireRHS,[N_d2*N_a,N_a*N_bothz]),[],1);
    VKron=reshape(VKron,[N_a,N_bothz]);

    VKrondist=reshape(VKron,[N_a*N_semiz,numel(VKron)/(N_a*N_semiz)])-reshape(VKronold,[N_a*N_semiz,numel(VKronold)/(N_a*N_semiz)]); % (a,semiz)-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over (a,semiz), within each exogenous state (semiz transitions depend on d, so semiz is endogenous here) (max and min omit the NaN from -Inf minus -Inf)

    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
        Policy=Policy(:); % (d2,aprime) index for each state (a,semiz,z)
        Ftemp=ReturnMatrix(Policy+addindexforstate); % keep return function of optimal policy for using in Howards
        Policy_d2=rem(Policy-1,N_d2)+1;
        Policy_aprime=ceil(Policy/N_d2);
        EVindex=Policy_aprime+EVcolindex; % [N_a*N_bothz,N_semiz], index for (aprime,semizprime,z) in EVz
        Pdz=pi_semiz(Pindex+N_semiz*N_semiz*(Policy_d2-1)); % [N_a*N_bothz,N_semiz], pi_semiz(semiz,semizprime,d2) at the optimal d2

        T_E=sparse(repmat(stateind,1,N_semiz*N_z),Policy_aprime+greedycolindex,repmat(Pdz,1,N_z).*repelem(pi_z(zind,:),1,N_semiz),N_a*N_bothz,N_a*N_bothz); % row (a,semiz,z) to column (aprime,semizprime,zprime)

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
        VKron=reshape(VKron,[N_a,N_bothz]);
    end

    tempcounter=tempcounter+1;

end
% Endogenous VFI: recover V in levels. Once converged, VKron=T(VKronold) (the Howards step is skipped in the last
% iteration) and VKron-VKronold is a function of the exogenous states only, up to the tolerance; call it dexo (the
% mean over (a,semiz) of the finite entries). VKronold+c is the fixed point when c=dexo+beta*E[c'], and V=T(VKronold+c)=VKron+beta*E[c'].
% (semiz is endogenous here: its transitions depend on d, so only z and e are exogenous.)
szVKron=size(VKron);
VKrondist=reshape(VKron,[N_a*N_semiz,numel(VKron)/(N_a*N_semiz)])-reshape(VKronold,[N_a*N_semiz,numel(VKronold)/(N_a*N_semiz)]);
finitedist=isfinite(VKrondist);
VKrondist(~finitedist)=0;
dexo=sum(VKrondist,1)./max(sum(finitedist,1),1); % 1-by-(z,e), zero where there is no finite entry
% c solves c=dexo+beta*pi_z*c
cexo=(eye(N_z,'gpuArray')-DiscountFactorParamsVec*pi_z)\dexo'; % z-by-1
VKron=reshape(reshape(VKron,[N_a*N_semiz,N_z])+DiscountFactorParamsVec*(pi_z*cexo)',szVKron);

Policy=reshape(Policy,[1,N_a,N_bothz]);
Policy=[rem(Policy-1,N_d2)+1; ceil(Policy/N_d2)]; % [d2; aprime]

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end


end
