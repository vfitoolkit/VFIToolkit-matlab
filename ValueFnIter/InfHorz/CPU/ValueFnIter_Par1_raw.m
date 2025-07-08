function [VKron, Policy]=ValueFnIter_Par1_raw(VKron, N_d,N_a,N_z, pi_z, beta, ReturnMatrix, Howards, Howards2, Tolerance) %,Verbose

tempcounter=1;
currdist=Inf;

PolicyIndexes=zeros(N_a,N_z);
Ftemp=zeros(N_a,N_z);

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);
% I suspect but have not yet double-checked that could instead just use
% aaa=kron(ones(N_a,1,'gpuArray'),pi_z);

while currdist>Tolerance
    
    VKronold=VKron;

    parfor z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        pi_z_z=pi_z(z_c,:);

        EV_z=VKronold.*kron(pi_z_z(1,:),ones(N_a,1));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        entireEV_z=kron(EV_z,ones(N_d,1));

        %Calc the RHS
        entireRHS=ReturnMatrix_z(:,:,1)+beta*entireEV_z*ones(1,N_a); %d by aprime by 1
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
             
        tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); % Store Ftemp to use with Howards
    end

    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            EVKrontemp=VKron(ceil(PolicyIndexes/N_d),:);
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*EVKrontemp;
        end
    end
    
    tempcounter=tempcounter+1;
    
end

Policy=zeros(2,N_a,N_z);
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

end
