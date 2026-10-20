function [VKron, Policy]=ValueFnIter_InfHorz_nod_noz_e_raw(VKron, N_a, N_e, pi_e, DiscountFactorParamsVec, ReturnMatrix, Howards, MaxHowards, Tolerance, maxiter)
% Value fn iteration, with Howards improvement iterations (a.k.a. modified Policy function iteration)
% No decision variable, no z, with iid e
% VKron is a-by-e, ReturnMatrix is aprime-by-a-by-e, pi_e is a column vector of probabilities

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

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
        tempmaxindex=shiftdim(Policy,1)+addindexforae; % aprime index, add the index for a and e
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_e]); % keep return function of optimal policy for using in Howards
        Policy=Policy(:); % a by e (this shape is just convenient for Howards)

        for Howards_counter=1:Howards
            EVKrontemp=VKron.*pi_e_alt;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=sum(EVKrontemp,2); % sum over e', aprime-by-1
            VKron=Ftemp+DiscountFactorParamsVec*reshape(EVKrontemp(Policy),[N_a,N_e]);
        end
    end

    tempcounter=tempcounter+1;

end

Policy=reshape(Policy,[1,N_a,N_e]);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end


end
