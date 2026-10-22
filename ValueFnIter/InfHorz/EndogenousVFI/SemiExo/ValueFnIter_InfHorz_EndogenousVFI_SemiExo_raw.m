function [VKron, Policy]=ValueFnIter_InfHorz_EndogenousVFI_SemiExo_raw(VKron, N_d1, N_d2, N_a, N_semiz, N_z, pi_z, pi_semiz, DiscountFactorParamsVec, ReturnMatrix, Howards, MaxHowards, Tolerance, maxiter)
% Endogenous VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_SemiExo_raw, but every loop stops on the
% span over (a,semiz) of V-Vold within each (z,e) (tolerances apply to that span), and V is recovered in levels after
% each fine-grid loop.
% Value fn iteration, with Howards improvement iterations (a.k.a. modified Policy function iteration)
% With semi-exogenous state semiz and markov z, and decision variables d1 and d2, where d2 determines the transition
% probabilities of semiz: pi_semiz(semiz,semizprime,d2). This is purediscretization (without refinement of d1).
% VKron is a-by-(semiz,z), ReturnMatrix is (d1,d2,aprime)-by-a-by-(semiz,z) [semiz varies first, then z]
% Policy is returned as [d1; d2; aprime] (indexes)
%
% The expectation of next period value, conditional on (semiz,z) and d2, is
%   EV(aprime,semiz,z,d2) = sum_{semizprime} pi_semiz(semiz,semizprime,d2) * sum_{zprime} pi_z(z,zprime) * V(aprime,semizprime,zprime)
% so the expectation over zprime is taken first, and then the one over semizprime (for every d2 at once).

N_bothz=N_semiz*N_z;

ReturnMatrix=reshape(ReturnMatrix,[N_d1,N_d2,N_a,N_a,N_semiz,N_z]);

pi_z_alt=shiftdim(pi_z',-2); % 1-by-1-by-zprime-by-z
pi_semiz_alt=permute(pi_semiz,[4,2,1,5,3]); % 1-by-semizprime-by-semiz-by-1-by-d2

% For Howards we need
addindexforstate=gpuArray(N_d1*N_d2*N_a*(0:1:N_a*N_bothz-1)'); % index for (a,semiz,z), for indexing ReturnMatrix at (d1,d2,aprime)
semizind=gpuArray(repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1)); % semiz of each state (a,semiz,z)
zind=gpuArray(repelem((1:1:N_z)',N_a*N_semiz,1)); % z of each state (a,semiz,z)
EVcolindex=N_a*gpuArray(0:1:N_semiz-1)+N_a*N_semiz*(zind-1); % [N_a*N_bothz,N_semiz], index for (semizprime,z) in EVz (which is aprime-by-semizprime-by-z)
Pindex=semizind+N_semiz*gpuArray(0:1:N_semiz-1); % [N_a*N_bothz,N_semiz], index for (semiz,semizprime) in pi_semiz

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
    EV=permute(EV,[6,5,1,2,3,4]); % 1-by-d2-by-aprime-by-1-by-semiz-by-z

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % d1 by d2 by aprime by a by semiz by z

    % Calc the max and it's index
    [VKron,Policy]=max(reshape(entireRHS,[N_d1*N_d2*N_a,N_a*N_bothz]),[],1);
    VKron=reshape(VKron,[N_a,N_bothz]);

    VKrondist=reshape(VKron,[N_a*N_semiz,numel(VKron)/(N_a*N_semiz)])-reshape(VKronold,[N_a*N_semiz,numel(VKronold)/(N_a*N_semiz)]); % (a,semiz)-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over (a,semiz), within each exogenous state (semiz transitions depend on d, so semiz is endogenous here) (max and min omit the NaN from -Inf minus -Inf)

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
        Policy=Policy(:); % (d1,d2,aprime) index for each state (a,semiz,z)
        Ftemp=ReturnMatrix(Policy+addindexforstate); % keep return function of optimal policy for using in Howards
        Policy_d2=rem(ceil(Policy/N_d1)-1,N_d2)+1;
        Policy_aprime=ceil(Policy/(N_d1*N_d2));
        EVindex=Policy_aprime+EVcolindex; % [N_a*N_bothz,N_semiz], index for (aprime,semizprime,z) in EVz
        Pdz=pi_semiz(Pindex+N_semiz*N_semiz*(Policy_d2-1)); % [N_a*N_bothz,N_semiz], pi_semiz(semiz,semizprime,d2) at the optimal d2

        for Howards_counter=1:Howards
            EVz=reshape(VKron,[N_a,N_semiz,N_z]).*pi_z_alt;
            EVz(isnan(EVz))=0;
            EVz=reshape(sum(EVz,3),[N_a,N_semiz,N_z]); % aprime-by-semizprime-by-z
            EVKrontemp=EVz(EVindex).*Pdz;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=sum(EVKrontemp,2); % sum over semizprime
            VKron=reshape(Ftemp+DiscountFactorParamsVec*EVKrontemp,[N_a,N_bothz]);
        end
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
Policy=[rem(Policy-1,N_d1)+1; rem(ceil(Policy/N_d1)-1,N_d2)+1; ceil(Policy/(N_d1*N_d2))]; % [d1; d2; aprime]

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end


end
