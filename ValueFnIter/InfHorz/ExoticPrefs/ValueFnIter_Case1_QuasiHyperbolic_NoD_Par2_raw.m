function [Vhat, Policyhat]=ValueFnIter_Case1_QuasiHyperbolic_NoD_Par2_raw(Vunderbar, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, Howards,Howards2,Tolerance, maxiter) % Verbose, a_grid, z_grid, 
% (last two entries of) DiscountFactorParamNames contains the names for the two parameters relating to
% Quasi-hyperbolic preferences.
%
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j: Vhat = u_t+beta_0*E[Vunderbar_{j+1}]
% See documentation for a fuller explanation of this.

N_a=prod(n_a);
N_z=prod(n_z);

Policyhat=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

%
Vhat=zeros(N_a,N_z,'gpuArray');

beta=prod(DiscountFactorParamsVec(1:end-1)); % Discount rate between two future periods
beta0beta=prod(DiscountFactorParamsVec); % Discount rate between present period and next period

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
        EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
                
        entireRHS=ReturnMatrix_z+beta0beta*EV_z*ones(1,N_a,1,'gpuArray'); %aprime by a
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vhat(:,z_c)=Vtemp;
        Policyhat(:,z_c)=maxindex;
        
        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
        
        % Now calculate Vunderbar (use beta instead of beta0)
        % AM I TREATING EV_z CORRECTLY
        entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1,'gpuArray'); %aprime by a
        Vunderbar(:,z_c)=entireRHS(tempmaxindex);
    end
    
    % I assume that once Vunderbar converges so has Vhat?
    VKrondist=reshape(Vunderbar-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

   

    
%     % I am guessing that can use Howards to iterate on both of Vhat and Vunderbar
%     if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
%         for Howards_counter=1:Howards
%             % Iterate Vhat
%             EVKrontemp=Vunderbar(Policyhat,:);
%             EVKrontemp=EVKrontemp.*aaa;
%             EVKrontemp(isnan(EVKrontemp))=0;
%             EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
%             Vhat=Ftemp+beta0beta*EVKrontemp;
%             
%             % Iterate Vunderber
%             EVKrontemp=Vhat(Policyhat,:);
%             EVKrontemp=EVKrontemp.*aaa;
%             EVKrontemp(isnan(EVKrontemp))=0;
%             EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
%             Vunderbar=Ftemp+beta*EVKrontemp;
%         end
%     end

%     if Verbose==1
        if rem(tempcounter,50)==0
            figure(1)
            if tempcounter==50 % First graph
                PolicyhatLastGraph=Policyhat;
                PolicyhatFirst=Policyhat;
            end
            plot(Policyhat(:,1))
            hold on
            plot(PolicyhatLastGraph(:,1))
            plot(PolicyhatFirst(:,1))
            hold off
            PolicyhatLastGraph=Policyhat;
            
%             Vunderbar(1:10)
%             Vhat(1:10)
            [Policyhat(1:10,1),Policyhat(1:10,151),Policyhat(1:10,351),zeros(10,1),Policyhat(end-9:end,1),Policyhat(end-9:end,151),Policyhat(end-9:end,351)]
            
            
            max(max(abs(Policyhat-PolicyhatOld)))

            disp(tempcounter)
            disp(currdist)
        end
%     end
    tempcounter=tempcounter+1;

end

% Policy=Policyhat;



end