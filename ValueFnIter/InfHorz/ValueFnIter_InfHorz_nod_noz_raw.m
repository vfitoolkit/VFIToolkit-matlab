function [VKron, Policy]=ValueFnIter_InfHorz_nod_noz_raw(VKron, N_a, DiscountFactorParamsVec, ReturnMatrix, Howards, MaxHowards, Tolerance, maxiter)
% Value fn iteration, with Howards improvement iterations (a.k.a. modified Policy function iteration)

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

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
        tempmaxindex=shiftdim(Policy,1)+addindexfora; % aprime index, add the index for a
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,1]); % keep return function of optimal policy for using in Howards
        Policy=Policy(:); % a by 1 (this shape is just convenient for Howards)

        for Howards_counter=1:Howards
            EVKrontemp=VKron(Policy,:);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
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
