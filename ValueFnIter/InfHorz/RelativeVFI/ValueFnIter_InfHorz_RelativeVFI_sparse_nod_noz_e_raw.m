function [VKron, Policy]=ValueFnIter_InfHorz_RelativeVFI_sparse_nod_noz_e_raw(VKron,N_a,N_e,pi_e,DiscountFactorParamsVec,ReturnMatrix,Howards,MaxHowards,Tolerance,maxiter)
% Relative VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_sparse_nod_noz_e_raw, but stops on the span over a of
% V-Vold within each exogenous state (Tolerance applies to that span), and then recovers V in levels.
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

    VKrondist=VKron-VKronold; % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

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
            % The dense expectation has the same -Inf*0=NaN problem as the sparse product below wherever the probability is zero against a
            % -Inf next-period state, so the -Inf entries are zeroed for the matmul and put back where they carry positive probability
            VKronisdead=(VKron==-Inf);
            VKronfinite=VKron;
            VKronfinite(VKronisdead)=0;
            EV_howard = VKronfinite*pi_e; % aprime-by-1
            EV_howard(double(VKronisdead)*pi_e>0)=-Inf;
            % GPU sparse*vector gives NaN when the vector holds -Inf, even where the row has no entry for that column (CPU gives the finite
            % product), so the -Inf entries are zeroed for the product and put back after: a row that puts positive weight on a -Inf entry is
            % -Inf itself. (CoreInfHorzVFIAlgoTests doPart(11); InfHorzGI_InfNaN_proposal.md, class B)
            EV_howardisdead=(EV_howard==-Inf);
            EV_howard(EV_howardisdead)=0;
            VKron = Ftemp+DiscountFactorParamsVec*Qmat*EV_howard;
            VKron(Qmat*double(EV_howardisdead)>0)=-Inf;
            VKron = reshape(VKron,[N_a,N_e]);
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
% e is iid, so E[c']=pi_e'*c=:cbar is the same for every e, and cbar=pi_e'*dexo/(1-beta)
VKron=VKron+DiscountFactorParamsVec*(dexo*pi_e)/(1-DiscountFactorParamsVec);

Policy=reshape(Policy,[1,N_a,N_e]);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end

end
