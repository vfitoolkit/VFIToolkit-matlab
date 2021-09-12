function [Vhat, Policy]=ValueFnIter_Case1_QuasiHyperbolic_Par2_raw(Vunderbar, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, Howards,Howards2, Tolerance, maxiter) %Verbose,
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

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);
% I suspect but have not yet double-checked that could instead just use
% aaa=kron(ones(N_a,1,'gpuArray'),pi_z);

beta=prod(DiscountFactorParamsVec(1:end-1)); % Discount rate between two future periods
beta0beta=prod(DiscountFactorParamsVec); % Discount rate between present period and next period

Vhat=zeros(N_a,N_z,'gpuArray');

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<maxiter
    VKronold=Vunderbar;
    
    for z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);     
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
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
    
    % I assume that once Vunderbar converges so has Vhat?
    VKrondist=reshape(Vunderbar-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            % Iterate Vhat
            EVKrontemp=Vunderbar(ceil(Policyhat/N_d),:);
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            Vhat=Ftemp+beta0beta*EVKrontemp;
            
            % Iterate Vunderbar
            EVKrontemp=Vhat(ceil(Policyhat/N_d),:);
            EVKrontemp=EVKrontemp.*aaa;
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

Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(Policyhat-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(Policyhat/N_d),-1);

end