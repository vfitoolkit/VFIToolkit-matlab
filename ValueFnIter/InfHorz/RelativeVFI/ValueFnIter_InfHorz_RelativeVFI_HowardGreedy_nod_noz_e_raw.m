function [VKron, Policy]=ValueFnIter_InfHorz_RelativeVFI_HowardGreedy_nod_noz_e_raw(VKron, N_a, N_e, pi_e, DiscountFactorParamsVec, ReturnMatrix, MaxHowards, Tolerance, maxiter)
% Relative VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_HowardGreedy_nod_noz_e_raw, but stops on the span over a of
% V-Vold within each exogenous state (Tolerance applies to that span), and then recovers V in levels.
% Value fn iteration using Howards-greedy, instead of Howards-iteration (a.k.a. Policy Function Iteration, instead of modified-Policy Function Iteration)
% No decision variable, no z, with iid e
% VKron is a-by-e, ReturnMatrix is aprime-by-a-by-e, pi_e is a column vector of probabilities

% Setup specific to greedy Howards
spI = gpuArray.speye(N_a*N_e);
greedyHind1=gpuArray(repelem((1:1:N_a*N_e)',1,N_e));
greedyHind2=gpuArray(N_a*(0:1:N_e-1));
greedyHpi=gpuArray(repmat(pi_e',N_a*N_e,1));

pi_e_alt=shiftdim(pi_e,-1); % 1-by-e

addindexforae=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_e-1));

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    % Calc the expectation term (except beta), e is iid so this does not depend on any current state
    EV=VKronold.*pi_e_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
    EV=sum(EV,2); % sum over e', aprime-by-1

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % aprime by a by e (EV autoexpands over a and e)

    %Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);
    VKron=shiftdim(VKron,1); % a by e

    VKrondist=VKron-VKronold; % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
        tempmaxindex=shiftdim(Policy,1)+addindexforae; % aprime index, add the index for a and e
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_e,1]); % keep return function of optimal policy for using in Howards
        if any(~isfinite(Ftemp))
            error("Howards-greedy doesn't work for non-finite return values; rerun with `vfoptions.howardsgreedy=0;`")
        end
        T_E=sparse(greedyHind1,Policy(:)+greedyHind2,greedyHpi,N_a*N_e,N_a*N_e); % row (a,e) to column (aprime,eprime), with probability pi_e(eprime)

        VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
        VKron=reshape(VKron,[N_a,N_e]);
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
% e is iid, so E[c']=pi_e'*c=:cbar is the same for every e, and cbar=pi_e'*dexo/(1-beta)
VKron=VKron+DiscountFactorParamsVec*(dexo*pi_e)/(1-DiscountFactorParamsVec);

Policy=reshape(Policy,[1,N_a,N_e]);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end


end
