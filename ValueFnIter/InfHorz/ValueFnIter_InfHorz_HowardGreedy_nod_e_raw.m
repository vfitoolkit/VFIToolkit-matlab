function [VKron, Policy]=ValueFnIter_InfHorz_HowardGreedy_nod_e_raw(VKron, N_a, N_z, N_e, pi_z, pi_e, DiscountFactorParamsVec, ReturnMatrix, MaxHowards, Tolerance, maxiter)
% Value fn iteration using Howards-greedy, instead of Howards-iteration (a.k.a. Policy Function Iteration, instead of modified-Policy Function Iteration)
% No decision variable, with z and iid e
% VKron is a-by-z-by-e, ReturnMatrix is aprime-by-a-by-z-by-e, pi_e is a column vector of probabilities

% Setup specific to greedy Howards
% Each row (a,z,e) goes to the columns (aprime,zprime,eprime) with probability pi_z(z,zprime)*pi_e(eprime)
spI = gpuArray.speye(N_a*N_z*N_e);
greedyHind1=gpuArray(repelem((1:1:N_a*N_z*N_e)',1,N_z*N_e));
greedyHind2=gpuArray(reshape(N_a*(0:1:N_z-1)'+N_a*N_z*(0:1:N_e-1),[1,N_z*N_e])); % zprime fastest, then eprime
greedyHpi=gpuArray(repmat(repelem(kron(pi_e',pi_z),N_a,1),N_e,1)); % kron(pi_e',pi_z) is z-by-(zprime,eprime)

pi_z_alt=shiftdim(pi_z',-1); % 1-by-zprime-by-z
pi_e_alt=shiftdim(pi_e,-2); % 1-by-1-by-e

addindexforaze=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1)+N_a*N_a*N_z*shiftdim((0:1:N_e-1),-1));

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    % First take the expectation over e', which is iid
    EV=VKronold.*pi_e_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
    EV=sum(EV,3); % sum over e', aprime-by-zprime
    % Then the expectation over z' conditional on z
    EV=EV.*pi_z_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension, aprime-by-1-by-z

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % aprime by a by z by e (EV autoexpands over a and e)

    %Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by z by e

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
        tempmaxindex=shiftdim(Policy,1)+addindexforaze; % aprime index, add the index for a, z and e
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z*N_e,1]); % keep return function of optimal policy for using in Howards
        if any(~isfinite(Ftemp))
            error("Howards-greedy doesn't work for non-finite return values; rerun with `vfoptions.howardsgreedy=0;`")
        end
        T_E=sparse(greedyHind1,Policy(:)+greedyHind2,greedyHpi,N_a*N_z*N_e,N_a*N_z*N_e);

        VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
        VKron=reshape(VKron,[N_a,N_z,N_e]);
    end

    tempcounter=tempcounter+1;

end

Policy=reshape(Policy,[1,N_a,N_z,N_e]);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end


end
