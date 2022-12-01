function [VKron, Policy, ExitPolicy]=ValueFnIter_Case1_EndogExit_Par1_raw(VKron, N_d,N_a,N_z, pi_z, beta, ReturnMatrix,ReturnToExitMatrix, Howards, Howards2, Tolerance,keeppolicyonexit) %,Verbose

PolicyIndexes=zeros(N_a,N_z);
ExitPolicy=zeros(N_a,N_z);
Ftemp=zeros(N_a,N_z);

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);
% I suspect but have not yet double-checked that could instead just use
% aaa=kron(ones(N_a,1,'gpuArray'),pi_z);


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
        EV_z=VKronold.*(ones(N_a,1)*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireEV_z=kron(EV_z,ones(N_d,1));
        entireRHS=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        % Exit decision
        ExitPolicy(:,z_c)=((ReturnToExitMatrix_z-Vtemp)>0); % Assumes that when indifferent you do not exit.
        
        VKron(:,z_c)=ExitPolicy(:,z_c).*ReturnToExitMatrix_z+(1-ExitPolicy(:,z_c)).*Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
             
        tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
    end
    
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        Ftemp=ExitPolicy.*ReturnToExitMatrix+(1-ExitPolicy).*Ftemp;
        for Howards_counter=1:Howards
            EVKrontemp=VKron(ceil(PolicyIndexes/N_d),:);
            
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

Policy=zeros(2,N_a,N_z); %NOTE: this is not actually in Kron form
if keeppolicyonexit==0 % This is default
    % Deliberate add zeros when ExitPolicy==1 so that cannot accidently make mistakes elsewhere in codes without throwing errors.
    Policy(1,:,:)=(1-ExitPolicy).*shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
    Policy(2,:,:)=(1-ExitPolicy).*shiftdim(ceil(PolicyIndexes/N_d),-1);
elseif keeppolicyonexit==1
    Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
    Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);
end

end