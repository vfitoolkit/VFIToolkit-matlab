function [VKron, Policy, ExitPolicy]=ValueFnIter_Case1_EndogExit_NoD_Par1_raw(VKron, N_a, N_z, pi_z, beta, ReturnMatrix,ReturnToExitMatrix, Howards,Howards2, Tolerance, keeppolicyonexit) %Verbose
%Does pretty much exactly the same as ValueFnIter_Case1, only without any
%decision variable (n_d=0)
% N_a=prod(n_a);
% N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z);
ExitPolicy=zeros(N_a,N_z);
Ftemp=zeros(N_a,N_z);

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance

    VKronold=VKron;
    
    parfor z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        ReturnToExitMatrix_z=ReturnToExitMatrix(:,z_c);
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=VKronold.*(ones(N_a,1)*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
                
        entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1); %aprime by a
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        % Exit decision
        ExitPolicy(:,z_c)=((ReturnToExitMatrix_z-Vtemp')>0); % Assumes that when indifferent you do not exit.
        VKron(:,z_c)=ExitPolicy(:,z_c).*ReturnToExitMatrix_z+(1-ExitPolicy(:,z_c)).*Vtemp';
        PolicyIndexes(:,z_c)=maxindex; % Note that this includes the policy that would be chosen if you did 
                % not exit, even when choose exit. This is because it makes it much easier to then implement 
                % Howards, and can just impose the =0 on exit on the final PolicyIndexes at the end of this 
                % function just prior to output.
        
        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
    end
        
    VKrondist=reshape(VKron-VKronold,[numel(VKron),1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        Ftemp=ExitPolicy.*ReturnToExitMatrix+(1-ExitPolicy).*Ftemp;
        for Howards_counter=1:Howards
            %VKrontemp=VKron;
            %EVKrontemp=VKrontemp(PolicyIndexes,:);
            EVKrontemp=VKron(PolicyIndexes,:);
            
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*(1-ExitPolicy).*EVKrontemp;
        end
    end
    
%     if Verbose==1
%         if rem(tempcounter,100)==0
%             disp(tempcounter)
%             disp(currdist)
%         end
%         tempcounter=tempcounter+1;
%     end
    tempcounter=tempcounter+1;

end
  
if keeppolicyonexit==0 % This is default
    Policy=(1-ExitPolicy).*PolicyIndexes; % Deliberate add zeros when ExitPolicy==1 so that cannot accidently make mistakes elsewhere in codes without throwing errors.
elseif keeppolicyonexit==1
    Policy=PolicyIndexes;
end


end