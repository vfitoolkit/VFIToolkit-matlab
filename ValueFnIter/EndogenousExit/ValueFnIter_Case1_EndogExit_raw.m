function [VKron, Policy, ExitPolicy]=ValueFnIter_Case1_EndogExit_raw(VKron, N_d,N_a,N_z,pi_z, beta, ReturnMatrix,ReturnToExitMatrix,Howards,Howards2,Tolerance,keeppolicyonexit)

PolicyIndexes1=zeros(N_a,N_z);
PolicyIndexes2=zeros(N_a,N_z);
ExitPolicy=zeros(N_a,N_z);
Ftemp=zeros(N_a,N_z);

tempcounter=1;
currdist=Inf;

while currdist>Tolerance
    
    VKronold=VKron;
    
    for z_c=1:N_z

        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=VKronold.*kron(pi_z(z_c,:),ones(N_a,1));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireEV_z=kron(EV_z,ones(N_d,1));
        
        for a_c=1:N_a
            %Calc the RHS
            entireRHS=ReturnMatrix(:,a_c,z_c)+beta*entireEV_z; %d by aprime by 1
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS);
            % Exit decision
            ExitPolicy(a_c,z_c)=((ReturnToExitMatrix(a_c,z_c)-Vtemp)>0); % Assumes that when indifferent you do not exit.
            VKron(a_c,z_c)=ExitPolicy(a_c,z_c)*ReturnToExitMatrix(a_c,z_c)+(1-ExitPolicy(a_c,z_c))*Vtemp;
            PolInd_temp=ind2sub_homemade([N_d,N_a],maxindex); %[d;aprime]
            PolicyIndexes1(a_c,z_c)=PolInd_temp(1);
            PolicyIndexes2(a_c,z_c)=PolInd_temp(2);
            % Note that this includes the policy that would be chosen if you did 
                % not exit, even when choose exit. This is because it makes it much easier to then implement 
                % Howards, and can just impose the =0 on exit on the final PolicyIndexes at the end of this 
                % function just prior to output.
            Ftemp(a_c,z_c)=ReturnMatrix(maxindex,a_c,z_c);
        end
    end
    
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
%         Ftemp=zeros(N_a,N_z);
%         for z_c=1:N_z
%             for a_c=1:N_a
%                 Ftemp(a_c,z_c)=ReturnMatrix(PolicyIndexes1(a_c,z_c)+(PolicyIndexes2(a_c,z_c)-1)*N_d,a_c,z_c);%FmatrixKron(PolicyIndexes1(a_c,z_c),PolicyIndexes2(a_c,z_c),a_c,z_c);
%             end
%         end
        Ftemp=ExitPolicy.*ReturnToExitMatrix+(1-ExitPolicy).*Ftemp;
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            for z_c=1:N_z
                EVKrontemp_z=VKrontemp(PolicyIndexes2(:,z_c),:).*kron(pi_z(z_c,:),ones(N_a,1)); %kron(pi_z(z_c,:),ones(nquad,1))
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

Policy=zeros(2,N_a,N_z);
if keeppolicyonexit==0 % This is default
    % Deliberate add zeros when ExitPolicy==1 so that cannot accidently make mistakes elsewhere in codes without throwing errors.
    Policy(1,:,:)=(1-ExitPolicy).*permute(PolicyIndexes1,[3,1,2]);
    Policy(2,:,:)=(1-ExitPolicy).*permute(PolicyIndexes2,[3,1,2]);
elseif keeppolicyonexit==1
    Policy(1,:,:)=permute(PolicyIndexes1,[3,1,2]);
    Policy(2,:,:)=permute(PolicyIndexes2,[3,1,2]);
end


end