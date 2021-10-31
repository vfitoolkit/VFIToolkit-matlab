function [VKron, Policy]=ValueFnIter_Case1_SemiEndog_Par1_raw(VKron, n_d,n_a,n_z, pi_z_semiendog, beta, ReturnMatrix, Howards,Howards2, Tolerance) %Verbose,

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z);

Ftemp=zeros(N_a,N_z);

pi_z_semiendog=reshape(pi_z_semiendog,[N_a*N_z,N_z]);
% Set up a verions that works nicely with parfor to reduce overhead
pi_z_semiendog_parfor=zeros(N_a,N_z,N_z);
for z_c=1:N_z
     a_z_c=(1:1:N_a)+(z_c-1)*N_z;
     pi_z_semiendog_parfor(:,:,z_c)=pi_z_semiendog(a_z_c,:);
end

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
    parfor z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);   
        pi_z_semiendog_parfor_z=pi_z_semiendog_parfor(:,:,z_c);
        %Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=VKronold.*pi_z_semiendog_parfor_z;
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireEV_z=kron(EV_z,ones(N_d,1));
        entireRHS=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
             
        tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
    end

    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            EVKrontemp=VKron(ceil(PolicyIndexes/N_d),:);
            
            EVKrontemp=EVKrontemp.*pi_z_semiendog;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*EVKrontemp;
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
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

end