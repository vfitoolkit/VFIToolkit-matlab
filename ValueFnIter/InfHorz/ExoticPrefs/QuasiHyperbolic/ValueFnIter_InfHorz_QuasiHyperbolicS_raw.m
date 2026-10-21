function [Vhat, Policy, Vunderbar]=ValueFnIter_InfHorz_QuasiHyperbolicS_raw(Vunderbar, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, beta0, ReturnMatrix, Howards,Howards2, Tolerance, maxiter) %Verbose,
% (last two entries of) DiscountFactorParamNames contains the names for the two parameters relating to
% Quasi-hyperbolic preferences.
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the
% time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j: Vhat = u_t+beta0beta*E[Vunderbar_{j+1}]
% See documentation for a fuller explanation of this.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

Policyhat=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

% Weights for the Howards expectation: row (a+N_a*(z-1)) is pi_z(z,:), matching the column-major
% linearisation that Vunderbar(ceil(Policyhat/N_d),:) produces. Same object
% ValueFnIter_InfHorz_nod_raw builds under this name.
% (This replaces the old bbb/ccc/aaa construction, which produced exactly this. The TODO there
% suggested kron(ones(N_a,1),pi_z) as the simplification -- that is NOT the same thing: it stacks
% pi_z N_a times, giving row (r-1)*N_z+i, the transposed ordering. repelem is the right one.)
pi_z_howards=repelem(pi_z,N_a,1);

% Count of states whose policy changed each iteration, for the non-convergence diagnostic below.
npolicychanged=zeros(maxiter,1,'gpuArray');

beta=prod(DiscountFactorParamsVec); % Discount rate between two future periods
beta0beta=beta0*beta; % Discount rate between present period and next period

Vhat=zeros(N_a,N_z,'gpuArray');

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<maxiter
    VKronold=Vunderbar;

    PolicyhatOld=Policyhat;

    for z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_z=sum(EV_z,2);

        entireEV_z=kron(EV_z,ones(N_d,1));
        entireRHS=ReturnMatrix_z+beta0beta*entireEV_z*ones(1,N_a,1); % Use beta0 to calculate Vhat

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vhat(:,z_c)=Vtemp;
        Policyhat(:,z_c)=maxindex;

        tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex);

        % Now for Vunderbar
        entireRHS=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1); % Use beta to calculate Vunderbar
        Vunderbar(:,z_c)=entireRHS(tempmaxindex);
    end

    % Convergence is tracked on Vunderbar. Vhat needs no separate test: at a fixed policy it is
    % one step off Vunderbar (see below), so once Vunderbar has converged Vhat follows.
    VKrondist=reshape(Vunderbar-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?

    npolicychanged(tempcounter)=sum(Policyhat(:)~=PolicyhatOld(:));

    % Howards on Vunderbar ONLY. At a fixed Policyhat, Vunderbar solves an ordinary discounted
    % policy evaluation, Vunderbar = Ftemp + beta*E[Vunderbar], in which beta0 does not appear;
    % so this inner loop is a plain beta-contraction and the usual Howards guarantee applies to it.
    % Vhat is NOT iterated here: it is not self-referential (it is one beta0*beta step off
    % Vunderbar and feeds nothing), so iterating it is at best wasted work. The earlier version of
    % this block iterated both and sourced Vunderbar from Vhat, which composes to
    % Vunderbar = F + beta*E[F] + beta*beta0beta*E^2[Vunderbar] -- a different fixed point.
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            EVKrontemp=Vunderbar(ceil(Policyhat/N_d),:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            Vunderbar=Ftemp+beta*EVKrontemp;
        end
    end

%     if Verbose==1
%         if rem(tempcounter,100)==0
%             disp(tempcounter)
%             disp(currdist)
%             fprintf('times: %2.8f, %2.8f, %2.8f \n',time1,time2,time3)
%         end
%
%         tempcounter=tempcounter+1;
%     end

    tempcounter=tempcounter+1;

end

%% Vhat, once, from the Vunderbar actually being returned
% Vhat is a report, not a state of the iteration: Vhat = u(Policyhat) + beta0*beta*E[Vunderbar].
% Recomputing it here rather than leaving whatever the last in-loop pass produced matters because
% Howards advances Vunderbar after that pass has set Vhat. It also makes the returned pair
% internally exact: away from the fixed point the in-loop Vhat is consistent with a superseded
% Vunderbar, which is not a useful thing to return.
EVKrontemp=Vunderbar(ceil(Policyhat/N_d),:);
EVKrontemp=EVKrontemp.*pi_z_howards;
EVKrontemp(isnan(EVKrontemp))=0;
EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
Vhat=Ftemp+beta0beta*EVKrontemp;

%% Non-convergence diagnostic
if currdist>Tolerance
    npolicychanged=gather(npolicychanged(1:tempcounter-1));
    nlook=min(50,numel(npolicychanged));
    recent=npolicychanged(end-nlook+1:end);
    warning(['ValueFnIter_InfHorz_QuasiHyperbolicS_raw stopped on reaching the maximum number ', ...
             'of iterations, not on convergence (set by vfoptions.maxiter). ', ...
             'Last currdist = %.16g; tolerance = %.16g. ', ...
             'Over the last %d iterations the policy changed at %.1f states on average ', ...
             '(max %d, and it was unchanged on %d of them). ', ...
             'A policy that keeps changing while currdist stalls means the Sophisticated iteration ', ...
             'is cycling rather than converging, which Howards cannot fix; a policy that has settled ', ...
             'while currdist stalls means something else is wrong.'], ...
             gather(currdist), Tolerance, nlook, mean(recent), max(recent), sum(recent==0))
end

Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(Policyhat-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(Policyhat/N_d),-1);

end
