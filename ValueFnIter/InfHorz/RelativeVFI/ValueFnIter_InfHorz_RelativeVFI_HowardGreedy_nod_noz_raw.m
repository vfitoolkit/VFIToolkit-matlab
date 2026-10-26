function [VKron, Policy]=ValueFnIter_InfHorz_RelativeVFI_HowardGreedy_nod_noz_raw(VKron, N_a, DiscountFactorParamsVec, ReturnMatrix, MaxHowards, Tolerance, maxiter)
% Relative VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_HowardGreedy_nod_noz_raw, but stops on the span over a of
% V-Vold within each exogenous state (Tolerance applies to that span), and then recovers V in levels.
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

    VKrondist=VKron-VKronold; % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

    % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
        tempmaxindex=shiftdim(Policy,1)+addindexfora; % aprime index, add the index for a
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,1]); % keep return function of optimal policy for using in Howards
        T_E=sparse(greedyHind1,Policy(:),1,N_a,N_a); % row a to column aprime (no shocks, so with probability one)

        % Greedy Howards with -Inf in V: a direct solve is global, so one -Inf in Ftemp gives NaN everywhere, and a finite Ftemp at a
        % state whose policy puts positive weight on a dead state gives a finite V where the truth is -Inf. So solve on the live states
        % only: a state is dead if V is -Inf there or if its policy puts positive weight on a dead state (the closure under T_E);
        % (I-beta*T_E) is block triangular in live-then-dead order, so the live block solves exactly on its own and V is -Inf on the
        % dead set. (CoreInfHorzVFIAlgoTests doPart(11); InfHorzGI_InfNaN_proposal.md, section 8)
        dead=(VKron(:)==-Inf);
        if any(dead)
            newdead=dead;
            while any(newdead)
                newdead=(T_E*double(dead)>0) & ~dead;
                dead=dead | newdead;
            end
            live=~dead;
            N_live=gather(sum(live));
            livemap=cumsum(live);
            [T_Erow,T_Ecol,T_Eval]=find(T_E);
            keep=live(T_Erow) & live(T_Ecol); % a live row has no positive weight on a dead column, so this is just the live rows
            T_E=sparse(livemap(T_Erow(keep)),livemap(T_Ecol(keep)),T_Eval(keep),N_live,N_live);
            VKrontemp=-Inf(numel(Ftemp),1,'gpuArray');
            VKrontemp(live)=(gpuArray.speye(N_live)-DiscountFactorParamsVec*T_E)\Ftemp(live);
            VKron=VKrontemp;
        else
            VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
        end
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
