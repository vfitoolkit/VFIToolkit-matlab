function [VKron, Policy]=ValueFnIter_InfHorz_noz_e_raw(VKron, n_d,n_a,n_e, pi_e, beta, ReturnMatrix, Howards,Howards2, Tolerance, maxiter)
% With decision variable, no z, with iid e
% VKron is a-by-e, ReturnMatrix is (d,aprime)-by-a-by-e, pi_e is a column vector of probabilities

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

pi_e_alt=shiftdim(pi_e,-1); % 1-by-e

addindexforae=gpuArray((0:1:N_a-1)'*N_d*N_a+(0:1:N_e-1)*N_d*N_a*N_a);

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    % Calc the expectation term (except beta), e is iid so this does not depend on any current state
    EV=VKronold.*pi_e_alt;
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
    EV=sum(EV,2); % sum over e', aprime-by-1

    entireRHS=ReturnMatrix+beta*repelem(EV,N_d,1); % (d,aprime) by a by e (EV autoexpands over a and e)

    % Calc the max and it's index
    [VKron,PolicyIndexes]=max(entireRHS,[],1);

    PolicyIndexes=shiftdim(PolicyIndexes,1); % a by e

    tempmaxindex=PolicyIndexes+addindexforae; % (d,aprime) index, add the index for a and e
    Ftemp=ReturnMatrix(tempmaxindex); % keep return function of optimal policy for using in Howards

    VKron=shiftdim(VKron,1); % a by e

    % Update currdist
    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        Policy_aprime=ceil(PolicyIndexes/N_d); % a by e
        for Howards_counter=1:Howards
            EVKrontemp=VKron.*pi_e_alt;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=sum(EVKrontemp,2); % sum over e', aprime-by-1
            VKron=Ftemp+beta*EVKrontemp(Policy_aprime);
        end
    end

    tempcounter=tempcounter+1;

end


Policy=zeros(2,N_a,N_e,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end

end
