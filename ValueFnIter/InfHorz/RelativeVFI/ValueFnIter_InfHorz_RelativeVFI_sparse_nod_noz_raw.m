function [VKron, Policy]=ValueFnIter_InfHorz_RelativeVFI_sparse_nod_noz_raw(VKron,N_a,DiscountFactorParamsVec,ReturnMatrix,Howards,MaxHowards,Tolerance,maxiter)
% Relative VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_sparse_nod_noz_raw, but stops on the span over a of
% V-Vold within each exogenous state (Tolerance applies to that span), and then recovers V in levels.
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

    VKrondist=VKron-VKronold; % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

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
            % GPU sparse*vector gives NaN when the vector holds -Inf, even where the row has no entry for that column (CPU gives the finite
            % product), so the -Inf entries are zeroed for the product and put back after: a row that puts positive weight on a -Inf entry is
            % -Inf itself. (CoreInfHorzVFIAlgoTests doPart(11); InfHorzGI_InfNaN_proposal.md, class B)
            VKronisdead=(VKron==-Inf);
            VKron(VKronisdead)=0;
            VKron = Ftemp+DiscountFactorParamsVec*Qmat*VKron;
            VKron(Qmat*double(VKronisdead)>0)=-Inf;
        end
    end
    tempcounter=tempcounter+1;
end %end while loop

%% Endogenous VFI: recover V in levels
% Once converged, VKron=T(VKronold) (the Howards step is skipped in the last iteration), and VKron-VKronold is a function of
% the exogenous states only, up to the tolerance. Call that function dexo (the mean over a of the finite entries). Then
% VKronold+c is the fixed point when c=dexo+beta*E[c'] (the expectation of next period exogenous states, given this period's),
% and V=T(VKronold+c)=VKron+beta*E[c'].
VKrondist=VKron-VKronold;
finitedist=isfinite(VKrondist);
VKrondist(~finitedist)=0;
dexo=sum(VKrondist,1)./max(sum(finitedist,1),1); % 1-by-(exogenous states), zero where there is no finite entry
% No exogenous states, so c=dexo/(1-beta) and E[c']=c (this is relative VFI)
VKron=VKron+DiscountFactorParamsVec*dexo/(1-DiscountFactorParamsVec);

Policy=reshape(Policy,[1,N_a,1]);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end

end
