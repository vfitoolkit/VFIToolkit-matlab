function [VKron, Policy]=ValueFnIter_InfHorz_RelativeVFI_noz_raw(VKron, n_d,n_a, beta, ReturnMatrix, Howards,Howards2, Tolerance, maxiter)
% Relative VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_noz_raw, but stops on the span over a of
% V-Vold within each exogenous state (Tolerance applies to that span), and then recovers V in levels.
% With decision variable, no z
% VKron is a-by-1, ReturnMatrix is (d,aprime)-by-a

N_d=prod(n_d);
N_a=prod(n_a);

addindexfora=gpuArray((0:1:N_a-1)'*N_d*N_a);

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    entireRHS=ReturnMatrix+beta*repelem(VKronold,N_d,1); % (d,aprime) by a (VKronold autoexpands over a)

    % Calc the max and it's index
    [VKron,PolicyIndexes]=max(entireRHS,[],1);

    PolicyIndexes=shiftdim(PolicyIndexes,1); % a by 1

    tempmaxindex=PolicyIndexes+addindexfora; % (d,aprime) index, add the index for a
    Ftemp=ReturnMatrix(tempmaxindex); % keep return function of optimal policy for using in Howards

    VKron=shiftdim(VKron,1); % a by 1

    % Update currdist
    VKrondist=VKron-VKronold; % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        Policy_aprime=ceil(PolicyIndexes/N_d); % a by 1
        for Howards_counter=1:Howards
            VKron=Ftemp+beta*VKron(Policy_aprime);
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
VKron=VKron+beta*dexo/(1-beta);

Policy=zeros(2,N_a,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:)=(rem(PolicyIndexes-1,N_d)+1)';
Policy(2,:)=(ceil(PolicyIndexes/N_d))';

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end

end
