function [VKron, Policy]=ValueFnIter_InfHorz_EndogenousVFI_sparse_nod_e_raw(VKron,N_a,N_z,N_e,pi_z,pi_e,DiscountFactorParamsVec,ReturnMatrix,Howards,MaxHowards,Tolerance,maxiter)
% Endogenous VFI (Bray, 2019): identical iterations to ValueFnIter_InfHorz_sparse_nod_e_raw, but stops on the span over a of
% V-Vold within each exogenous state (Tolerance applies to that span), and then recovers V in levels.
% Value fn iteration, with Howards improvement iterations done using a sparse matrix
% No decision variable, with z and iid e
% VKron is a-by-z-by-e, ReturnMatrix is aprime-by-a-by-z-by-e, pi_e is a column vector of probabilities

pi_z_transpose = transpose(pi_z);
pi_z_alt       = shiftdim(pi_z_transpose,-1); % 1-by-zprime-by-z
pi_e_alt       = shiftdim(pi_e,-2); % 1-by-1-by-e

% Precompute variables for Howard improvement step
NaVec = gpuArray.colon(1,N_a)';
NzVec = gpuArray.colon(1,N_z)';
NeVec = gpuArray.colon(1,N_e)';
a_ind = repmat(NaVec,N_z*N_e,1);
z_ind = repmat(repelem(NzVec,N_a,1),N_e,1);
e_ind = repelem(NeVec,N_a*N_z,1);
ind   = a_ind+(z_ind-1)*N_a+(e_ind-1)*N_a*N_z;

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

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % aprime by a by z by e (EV autoexpands over a and e)

    %Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);

    VKron=shiftdim(VKron,1); % a by z by e

    VKrondist=VKron-VKronold; % a-by-(exogenous states)
    currdist=max(max(VKrondist,[],1)-min(VKrondist,[],1),[],'all'); % endogenous VFI: the span over a, within each exogenous state (max and min omit the NaN from -Inf minus -Inf)

    % Use Howards Policy Fn Iteration Improvement
    % (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards

        % Reshape Policy as column vector with size [N_a*N_z*N_e,1]
        Policy_vec=reshape(Policy,[N_a*N_z*N_e,1]);
        % Linear indices of the elements you want in ReturnMatrix
        linIdx = Policy_vec + (a_ind-1)*N_a + (z_ind-1)*N_a*N_a + (e_ind-1)*N_a*N_a*N_z;
        Ftemp = ReturnMatrix(linIdx);
        % Build large sparse matrix, (a,z,e) to (aprime,z) (the expectation over (zprime,eprime) given z is done before multiplying by it)
        indp  = Policy_vec+(z_ind-1)*N_a;
        Qmat  = sparse(ind,indp,1,N_a*N_z*N_e,N_a*N_z);
        % Howard iterations, with Ftemp and Qmat precomputed
        for Howards_counter=1:Howards
            % The dense expectation has the same -Inf*0=NaN problem as the sparse product below wherever the probability is zero against a
            % -Inf next-period state, so the -Inf entries are zeroed for the matmul and put back where they carry positive probability
            VKronisdead=(reshape(VKron,[N_a*N_z,N_e])==-Inf);
            VKronfinite=reshape(VKron,[N_a*N_z,N_e]);
            VKronfinite(VKronisdead)=0;
            EV_howard = VKronfinite*pi_e; % expectation over e', (aprime,zprime)
            EV_howard(double(VKronisdead)*pi_e>0)=-Inf;
            EV_howardisdeadz=(reshape(EV_howard,[N_a,N_z])==-Inf);
            EV_howardfinite=reshape(EV_howard,[N_a,N_z]);
            EV_howardfinite(EV_howardisdeadz)=0;
            EV_howard = EV_howardfinite*pi_z_transpose; % expectation over z' given z, (aprime,z)
            EV_howard(double(EV_howardisdeadz)*pi_z_transpose>0)=-Inf;
            EV_howard = reshape(EV_howard,[N_a*N_z,1]);
            % GPU sparse*vector gives NaN when the vector holds -Inf, even where the row has no entry for that column (CPU gives the finite
            % product), so the -Inf entries are zeroed for the product and put back after: a row that puts positive weight on a -Inf entry is
            % -Inf itself. (CoreInfHorzVFIAlgoTests doPart(11); InfHorzGI_InfNaN_proposal.md, class B)
            EV_howardisdead=(EV_howard==-Inf);
            EV_howard(EV_howardisdead)=0;
            VKron = Ftemp+DiscountFactorParamsVec*Qmat*EV_howard;
            VKron(Qmat*double(EV_howardisdead)>0)=-Inf;
            VKron = reshape(VKron,[N_a,N_z,N_e]);
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
% e is iid, so E[c'|z]=pi_z*cbar where cbar(z)=sum_e pi_e(e)*c(z,e), and cbar solves cbar=dexo*pi_e+beta*pi_z*cbar
cbar=(eye(N_z,'gpuArray')-DiscountFactorParamsVec*pi_z)\(reshape(dexo,[N_z,N_e])*pi_e); % z-by-1
VKron=VKron+DiscountFactorParamsVec*(pi_z*cbar)'; % the 1-by-z autoexpands over a and e

Policy=reshape(Policy,[1,N_a,N_z,N_e]);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end

end
