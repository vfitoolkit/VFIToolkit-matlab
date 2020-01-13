function [VKron, Policy, ExitPolicy]=ValueFnIter_Case1_EndogExit_NoD_Par1_raw(VKron, N_a, N_z, pi_z, beta, ReturnMatrix,ReturnToExitMatrix, Howards,Howards2, Tolerance, keeppolicyonexit) %Verbose
%Does pretty much exactly the same as ValueFnIter_Case1, only without any
%decision variable (n_d=0)
% N_a=prod(n_a);
% N_z=prod(n_z);
% 
% if Verbose==1
%     disp('Starting Value Fn Iteration')
%     tempcounter=1;
% end

PolicyIndexes=zeros(N_a,N_z);
ExitPolicy=zeros(N_a,N_z);

Ftemp=zeros(N_a,N_z);

tempcounter=1;
currdist=Inf;

while currdist>Tolerance

    VKronold=VKron;

    parfor z_c=1:N_z
        pi_z_z=pi_z(z_c,:);
        VKron_z=zeros(N_a,1);
        PolicyIndexes_z=zeros(N_a,1);
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        ReturnToExitMatrix_z=ReturnToExitMatrix(:,z_c)
        ExitPolicy_z=zeros(N_a,1);

        EV_z=VKronold.*kron(ones(N_a,1),pi_z_z(1,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        for a_c=1:N_a
            entireRHS=ReturnMatrix_z(:,a_c)+beta*EV_z; %aprime by 1
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS);
            % Exit decision
            ExitPolicy_z(a_c)=((ReturnToExitMatrix_z(a_c)-Vtemp)>0); % Assumes that when indifferent you do not exit.
            VKron_z(a_c)=ExitPolicy_z(a_c)*ReturnToExitMatrix_z(a_c)+(1-ExitPolicy_z(a_c))*Vtemp;
            PolicyIndexes_z(a_c)=maxindex;
                % Note that this includes the policy that would be chosen if you did 
                % not exit, even when choose exit. This is because it makes it much easier to then implement 
                % Howards, and can just impose the =0 on exit on the final PolicyIndexes at the end of this 
                % function just prior to output.
            Ftemp_z(a_c)=ReturnMatrix_z(maxindex,a_c);
        end
        
        VKron(:,z_c)=VKron_z;
        PolicyIndexes(:,z_c)=PolicyIndexes_z;
        ExitPolicy(:,z_c)=ExitPolicy_z;
        Ftemp(:,z_c)=Ftemp_z;
    end
        
    VKrondist=reshape(VKron-VKronold,[numel(VKron),1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
%         Ftemp=zeros(N_a,N_z);
%         for z_c=1:N_z
%             for a_c=1:N_a
%                 Ftemp(a_c,z_c)=ReturnMatrix(PolicyIndexes(a_c,z_c),a_c,z_c);%FmatrixKron(PolicyIndexes1(a_c,z_c),PolicyIndexes2(a_c,z_c),a_c,z_c);
%             end
%         end
        Ftemp=ExitPolicy.*ReturnToExitMatrix+(1-ExitPolicy).*Ftemp;
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            for z_c=1:N_z
                EVKrontemp_z=VKrontemp(PolicyIndexes(:,z_c),:).*kron(pi_z(z_c,:),ones(N_a,1)); %kron(pi_z(z_c,:),ones(nquad,1))
                EVKrontemp_z(isnan(EVKrontemp_z))=0; %Multiplying zero (transition prob) by -Inf (value fn) gives NaN
                VKron(:,z_c)=Ftemp(:,z_c)+beta*(1-ExitPolicy(:,z_c)).*sum(EVKrontemp_z,2);
            end
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