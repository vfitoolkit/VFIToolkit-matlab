function [VKron, Policy, ExitPolicy]=ValueFnIter_Case1_EndogExit_Par1_raw(VKron, N_d,N_a,N_z, pi_z, beta, ReturnMatrix,ReturnToExitMatrix, Howards, Howards2, Tolerance,keeppolicyonexit) %,Verbose

PolicyIndexes1=zeros(N_a,N_z);
PolicyIndexes2=zeros(N_a,N_z);
ExitPolicy=zeros(N_a,N_z);
Ftemp=zeros(N_a,N_z);

tempcounter=1;
currdist=Inf;

while currdist>Tolerance
    
    VKronold=VKron;
    parfor z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        ReturnToExitMatrix_z=ReturnToExitMatrix(:,z_c);
        pi_z_z=pi_z(z_c,:);

        EV_z=VKronold.*kron(pi_z_z(1,:),ones(N_a,1));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        entireEV_z=kron(EV_z,ones(N_d,1));

        %Calc the RHS
        entireRHS=ReturnMatrix_z(:,:,1)+beta*entireEV_z*ones(1,N_a); %d by aprime by 1
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        % Exit decision
        ExitPolicy_z=((ReturnToExitMatrix_z-Vtemp)>0); % Assumes that when indifferent you do not exit.
            
        VKron_z=ExitPolicy_z.*ReturnToExitMatrix_z+(1-ExitPolicy_z).*Vtemp;
        % Next few lines are ind2sub_homemade([N_d,N_a],maxindex), just
        % explicitly rewritten to do a whole vector at once.
        PolicyIndexes1_z=rem(maxindex-1,N_d)+1;
        PolicyIndexes2_z=ceil(maxindex./N_d);
        
        VKron(:,z_c)=VKron_z;
        PolicyIndexes1(:,z_c)=PolicyIndexes1_z;
        PolicyIndexes2(:,z_c)=PolicyIndexes2_z;
        Ftemp(:,z_c)=ReturnMatrix_z(maxindex,:);
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