function [VKron, Policy]=ValueFnIter_InfHorz_e_raw(VKron, n_d,n_a,n_z,n_e, pi_z, pi_e, beta, ReturnMatrix, Howards,Howards2, Tolerance, maxiter)
% With decision variable, with z and iid e
% VKron is a-by-z-by-e, ReturnMatrix is (d,aprime)-by-a-by-z-by-e, pi_e is a column vector of probabilities

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

pi_z_alt=shiftdim(pi_z',-1); % 1-by-zprime-by-z
pi_e_alt=shiftdim(pi_e,-2); % 1-by-1-by-e

addindexforaze=gpuArray((0:1:N_a-1)'*N_d*N_a+(0:1:N_z-1)*N_d*N_a*N_a+shiftdim((0:1:N_e-1),-1)*N_d*N_a*N_a*N_z);
addindexforz=gpuArray(repmat(N_a*(0:1:N_z-1),N_a,1,N_e)); % a-by-z-by-e, for indexing EV (which is aprime-by-z)

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

    entireRHS=ReturnMatrix+beta*repelem(EV,N_d,1); % (d,aprime) by a by z by e (EV autoexpands over a and e)

    % Calc the max and it's index
    [VKron,PolicyIndexes]=max(entireRHS,[],1);

    PolicyIndexes=shiftdim(PolicyIndexes,1); % a by z by e

    tempmaxindex=PolicyIndexes+addindexforaze; % (d,aprime) index, add the index for a, z and e
    Ftemp=ReturnMatrix(tempmaxindex); % keep return function of optimal policy for using in Howards

    VKron=shiftdim(VKron,1); % a by z by e

    % Update currdist
    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        Policy_aprimez=ceil(PolicyIndexes/N_d)+addindexforz; % aprime index, add the index for z
        for Howards_counter=1:Howards
            EVKrontemp=VKron.*pi_e_alt;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=sum(EVKrontemp,3); % sum over e', aprime-by-zprime
            EVKrontemp=EVKrontemp.*pi_z_alt;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]); % sum over z', aprime-by-z
            VKron=Ftemp+beta*EVKrontemp(Policy_aprimez);
        end
    end

    tempcounter=tempcounter+1;

end


Policy=zeros(2,N_a,N_z,N_e,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end

end
