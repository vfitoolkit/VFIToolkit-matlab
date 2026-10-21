function [VKron, Policy]=ValueFnIter_InfHorz_EndogenousVFI_HowardGreedy_nod_e_raw(VKron, N_a, N_z, N_e, pi_z, pi_e, DiscountFactorParamsVec, ReturnMatrix, MaxHowards, Tolerance, maxiter)
% Endogenous VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_HowardGreedy_nod_e_raw, but stops on the span over a of
% V-Vold within each exogenous state (Tolerance applies to that span), and then recovers V in levels.
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

    VKrondist=VKron-VKronold; % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

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

%% Endogenous VFI: recover V in levels
% Once converged, VKron=T(VKronold) (the Howards step is skipped in the last iteration), and VKron-VKronold is a function of
% the exogenous states only, up to the tolerance. Call that function dexo (the mean over a of the finite entries). Then
% VKronold+c is the fixed point when c=dexo+beta*E[c'] (the expectation of next period exogenous states, given this period's),
% and V=T(VKronold+c)=VKron+beta*E[c'].
VKrondist=VKron-VKronold;
finitedist=isfinite(VKrondist);
VKrondist(~finitedist)=0;
dexo=sum(VKrondist,1)./max(sum(finitedist,1),1); % 1-by-(exogenous states), zero where there is no finite entry
% e is iid, so E[c'|z]=pi_z*cbar where cbar(z)=sum_e pi_e(e)*c(z,e), and cbar solves cbar=dexo*pi_e+beta*pi_z*cbar
cbar=(eye(N_z,'gpuArray')-DiscountFactorParamsVec*pi_z)\(reshape(dexo,[N_z,N_e])*pi_e); % z-by-1
VKron=VKron+DiscountFactorParamsVec*(pi_z*cbar)'; % the 1-by-z autoexpands over a and e

Policy=reshape(Policy,[1,N_a,N_z,N_e]);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end


end
