function [VKron, Policy, ExitPolicy]=ValueFnIter_Case1_EndogExit_SemiEndog_NoD_raw(VKron, N_a, N_z, pi_z_semiendog, beta, ReturnMatrix,ReturnToExitMatrix, Howards,Howards2,Tolerance, keeppolicyonexit) 
%Does pretty much exactly the same as ValueFnIter_Case1, only without any decision variable (n_d=0)

PolicyIndexes=zeros(N_a,N_z);
ExitPolicy=zeros(N_a,N_z);
Ftemp=zeros(N_a,N_z);

tempcounter=1;
currdist=Inf;
while currdist>Tolerance

    VKronold=VKron;

    for z_c=1:N_z
        %Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=VKronold.*squeeze(pi_z_semiendog(:,z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        for a_c=1:N_a
            entireRHS=ReturnMatrix(:,a_c,z_c)+beta*EV_z; %aprime by 1
            
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS);
            % Exit decision
            ExitPolicy(a_c,z_c)=((ReturnToExitMatrix(a_c,z_c)-Vtemp)>0); % Assumes that when indifferent you do not exit.
            VKron(a_c,z_c)=ExitPolicy(a_c,z_c)*ReturnToExitMatrix(a_c,z_c)+(1-ExitPolicy(a_c,z_c))*Vtemp;
            PolicyIndexes(a_c,z_c)=maxindex;
                % Note that this includes the policy that would be chosen if you did 
                % not exit, even when choose exit. This is because it makes it much easier to then implement 
                % Howards, and can just impose the =0 on exit on the final PolicyIndexes at the end of this 
                % function just prior to output.
            Ftemp(a_c,z_c)=ReturnMatrix(PolicyIndexes(a_c,z_c),a_c,z_c);
        end
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
                EVKrontemp_z=VKrontemp(PolicyIndexes(:,z_c),:).*squeeze(pi_z_semiendog(:,z_c,:));
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

if keeppolicyonexit==0
    Policy=(1-ExitPolicy).*PolicyIndexes; % Deliberate add zeros when ExitPolicy==1 so that cannot accidently make mistakes elsewhere in codes without throwing errors.
elseif keeppolicyonexit==1
    Policy=PolicyIndexes;
end

end