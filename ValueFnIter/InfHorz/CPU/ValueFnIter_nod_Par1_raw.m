function [VKron, Policy]=ValueFnIter_nod_Par1_raw(VKron, N_a, N_z, pi_z, beta, ReturnMatrix, Howards,Howards2, Tolerance) %Verbose

Policy=zeros(N_a,N_z);

Ftemp=zeros(N_a,N_z);

pi_z_howards=repelem(pi_z,N_a,1);

tempcounter=1;
currdist=Inf;

while currdist>Tolerance

    VKronold=VKron;

    parfor z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        pi_z_z=pi_z(z_c,:);

        EV_z=VKronold.*pi_z_z(1,:);
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);

        entireRHS=ReturnMatrix_z+beta*EV_z; %aprime by a

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS);
        VKron(:,z_c)=Vtemp;
        Policy(:,z_c)=maxindex;

        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
    end
        
    VKrondist=reshape(VKron-VKronold,[numel(VKron),1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            EVKrontemp=VKron(Policy,:);
            
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*EVKrontemp;
        end
    end
    
    tempcounter=tempcounter+1;

end
  
Policy=reshape(Policy,[1,N_a,N_z]);

end
