function [VKron, Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_Case1_EndogExit2_Par2_raw(VKron, n_d,n_a,n_z, pi_z, beta, ReturnMatrix,ReturnToExitMatrix, Howards,Howards2, Tolerance,keeppolicyonexit, exitprobabilities, continuationcost) %Verbose,

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray');
PolicyWhenExitIndexes=zeros(N_a,N_z,'gpuArray');
ExitPolicy=zeros(N_a,N_z,'gpuArray');
Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);
% I suspect but have not yet double-checked that could instead just use
% aaa=kron(ones(N_a,1,'gpuArray'),pi_z);


%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
%     tic;
    for z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);     
        ReturnToExitMatrix_z=ReturnToExitMatrix(:,:,z_c);     
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireEV_z=kron(EV_z,ones(N_d,1));
        entireRHS=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1);

        %Calc the max and it's index (when not exiting)
        [Vtemp,maxindex]=max(entireRHS,[],1);
        % Calc the max and it's index when exiting
        [FtempWhenExit,maxindexWhenExit]=max(ReturnToExitMatrix_z,[],1); % MOVE THIS OUTSIDE OF THE while loop
        % Endogenous Exit decision
        ExitPolicy_z=((FtempWhenExit-(Vtemp-continuationcost))>0); % Assumes that when indifferent you do not exit.
        
        % % The following line is implementing in a single line what is commented out here.
        % V_z_noexit=Vtemp;
        % V_z_endogexit=ExitPolicy(:,z_c).*FtempWhenExit+(1-ExitPolicy(:,z_c)).*(Vtemp-continuationcost);
        % V_z_exoexit=ReturnToExitMatrix_z;
        % VKron(:,z_c)=exitprobabilities(1)*V_z_noexit+exitprobabilities(2)*V_z_endoexit+exitprobabilities(3)*V_z_exoexit
        
        VKron(:,z_c)=exitprobabilities(1)*Vtemp+exitprobabilities(2)*(ExitPolicy_z.*FtempWhenExit+(1-ExitPolicy_z).*(Vtemp-continuationcost))+exitprobabilities(3)*FtempWhenExit;
        PolicyIndexes(:,z_c)=maxindex;
        PolicyWhenExitIndexes(:,z_c)=maxindexWhenExit;  % MOVE THIS OUTSIDE OF THE while loop
        ExitPolicy(:,z_c)=ExitPolicy_z;
             
        tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex);
%         tempmaxindexWhenExit=maxindexWhenExit+(0:1:N_a-1)*(N_d*N_a);
        FWhenExit(:,z_c)=FtempWhenExit; %ReturnToExitMatrix_z(tempmaxindexWhenExit);
    end
%     time1=toc;
% 
%     tic;
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?
%     time2=toc;
%     tic;
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        % ReturnToExitMatrix % When no exit
        Ftemp2=ExitPolicy.*FWhenExit+(1-ExitPolicy).*(Ftemp-continuationcost); % When endogenous exit
        % FWhenExit % When (exog) exit.
        for Howards_counter=1:Howards
%             VKrontemp=VKron;
%             EVKrontemp=VKrontemp(ceil(PolicyIndexes/N_d),:);
            EVKrontemp=VKron(ceil(PolicyIndexes/N_d),:);
            
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=exitprobabilities(1)*(Ftemp+beta*EVKrontemp)+exitprobabilities(2)*(Ftemp2+beta*(1-ExitPolicy).*EVKrontemp)+exitprobabilities(3)*FWhenExit;
        end
    end
%     time3=toc;
    
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
% if keeppolicyonexit==0 % This is default
%     % Deliberate add zeros when ExitPolicy==1 so that cannot accidently make mistakes elsewhere in codes without throwing errors.
%     Policy(1,:,:)=(1-ExitPolicy).*shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
%     Policy(2,:,:)=(1-ExitPolicy).*shiftdim(ceil(PolicyIndexes/N_d),-1);
% elseif keeppolicyonexit==1
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1); % With 'end-of-period' timing of exit this is the only relevant one.
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);
% end

PolicyWhenExit=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
PolicyWhenExit(1,:,:)=shiftdim(rem(PolicyWhenExitIndexes-1,N_d)+1,-1); % With 'end-of-period' timing of exit this is the only relevant one.
PolicyWhenExit(2,:,:)=shiftdim(ceil(PolicyWhenExitIndexes/N_d),-1);

% ExitPolicy

end