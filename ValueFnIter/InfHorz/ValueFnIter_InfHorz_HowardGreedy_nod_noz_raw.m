function [VKron, Policy]=ValueFnIter_InfHorz_HowardGreedy_nod_noz_raw(VKron, N_a, DiscountFactorParamsVec, ReturnMatrix, MaxHowards, Tolerance, maxiter)
% Value fn iteration using Howards-greedy, instead of Howards-iteration (a.k.a. Policy Function Iteration, instead of modified-Policy Function Iteration)
% No decision variable, no z
% VKron is a-by-1, ReturnMatrix is aprime-by-a

% Setup specific to greedy Howards
spI = gpuArray.speye(N_a);
greedyHind1=gpuArray((1:1:N_a)');

addindexfora=gpuArray(N_a*(0:1:N_a-1)');

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

    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
        tempmaxindex=shiftdim(Policy,1)+addindexfora; % aprime index, add the index for a
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,1]); % keep return function of optimal policy for using in Howards
        if any(~isfinite(Ftemp))
            error("Howards-greedy doesn't work for non-finite return values; rerun with `vfoptions.howardsgreedy=0;`")
        end
        T_E=sparse(greedyHind1,Policy(:),1,N_a,N_a); % row a to column aprime (no shocks, so with probability one)

        VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
    end

    tempcounter=tempcounter+1;

end

Policy=reshape(Policy,[1,N_a,1]);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end


end
