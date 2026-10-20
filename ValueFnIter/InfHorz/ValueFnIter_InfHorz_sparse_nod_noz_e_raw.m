function [VKron, Policy]=ValueFnIter_InfHorz_sparse_nod_noz_e_raw(VKron,N_a,N_e,pi_e,DiscountFactorParamsVec,ReturnMatrix,Howards,MaxHowards,Tolerance,maxiter)
% Value fn iteration, with Howards improvement iterations done using a sparse matrix
% No decision variable, no z, with iid e
% VKron is a-by-e, ReturnMatrix is aprime-by-a-by-e, pi_e is a column vector of probabilities

pi_e_alt=shiftdim(pi_e,-1); % 1-by-e

% Precompute variables for Howard improvement step
NaVec = gpuArray.colon(1,N_a)';
NeVec = gpuArray.colon(1,N_e)';
a_ind = repmat(NaVec,N_e,1);
e_ind = repelem(NeVec,N_a,1);
ind   = a_ind+(e_ind-1)*N_a;

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

    % Use Howards Policy Fn Iteration Improvement
    % (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards

        % Reshape Policy as column vector with size [N_a*N_e,1]
        Policy_vec=reshape(Policy,[N_a*N_e,1]);
        % Linear indices of the elements you want in ReturnMatrix
        linIdx = Policy_vec + (a_ind-1)*N_a + (e_ind-1)*N_a*N_a;
        Ftemp = ReturnMatrix(linIdx);
        % Build large sparse matrix, (a,e) to aprime (e is iid, so next period e does not depend on this period)
        Qmat  = sparse(ind,Policy_vec,1,N_a*N_e,N_a);
        % Howard iterations, with Ftemp and Qmat precomputed
        for Howards_counter=1:Howards
            EV_howard = VKron*pi_e; % aprime-by-1
            VKron = Ftemp+DiscountFactorParamsVec*Qmat*EV_howard;
            VKron = reshape(VKron,[N_a,N_e]);
        end
    end
    tempcounter=tempcounter+1;
end %end while loop

Policy=reshape(Policy,[1,N_a,N_e]);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end

end
