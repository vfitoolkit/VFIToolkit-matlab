function [VKron, Policy]=ValueFnIter_InfHorz_sparse_nod_noz_raw(VKron,N_a,DiscountFactorParamsVec,ReturnMatrix,Howards,MaxHowards,Tolerance,maxiter)
% Value fn iteration, with Howards improvement iterations done using a sparse matrix
% No decision variable, no z
% VKron is a-by-1, ReturnMatrix is aprime-by-a

% Precompute variables for Howard improvement step
a_ind = gpuArray.colon(1,N_a)';

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*VKronold; % aprime by a

    %Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);

    VKron=shiftdim(VKron,1); % a by 1

    VKrondist=VKron-VKronold;
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement
    % (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards

        % Reshape Policy as column vector with size [N_a,1]
        Policy_vec=reshape(Policy,[N_a,1]);
        % Linear indices of the elements you want in ReturnMatrix
        linIdx = Policy_vec + (a_ind-1)*N_a;
        Ftemp = ReturnMatrix(linIdx);
        % Build large sparse matrix, a to aprime (no shocks, so with probability one)
        Qmat  = sparse(a_ind,Policy_vec,1,N_a,N_a);
        % Howard iterations, with Ftemp and Qmat precomputed
        for Howards_counter=1:Howards
            VKron = Ftemp+DiscountFactorParamsVec*Qmat*VKron;
        end
    end
    tempcounter=tempcounter+1;
end %end while loop

Policy=reshape(Policy,[1,N_a,1]);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end

end
