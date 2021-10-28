function [VKron, Policy]=ValueFnIter_Case1_NoD_Par2_raw(VKron, n_a, n_z, pi_z, beta, ReturnMatrix, Howards,Howards2,Tolerance) % Verbose, a_grid, z_grid, 
%Does pretty much exactly the same as ValueFnIter_Case1, only without any decision variable (n_d=0)

N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a*N_z,1,'gpuArray');

aaa=kron(pi_z,ones(N_a,1,'gpuArray'));

% aaa2=kron(ones(1,N_a,'gpuArray'),pi_z);
aaa3=kron(pi_z,ones(1,N_a,'gpuArray'));


aaa4=ones(N_a*N_z,N_a);


% I will be able to drop these
VKron=reshape(VKron,[N_a*N_z,1]);
ReturnMatrix=reshape(ReturnMatrix,[N_a,N_a*N_z]);

% I should make the ReturnMatrix (a,z)-by-aprime
ReturnMatrix=ReturnMatrix';

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;

    EV=aaa3.*VKronold';
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=reshape(EV,[N_z,N_a,N_z]);
    EV=sum(EV,3);
    EV=reshape(EV,[N_z,N_a]); % Depends on current z and on aprime

    entireRHS=ReturnMatrix+beta*repmat(EV,N_a,1); %aprime by a
    
    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],2);
    VKron=Vtemp;
    PolicyIndexes=maxindex;

    tempmaxindex=(1:1:N_a*N_z)'+(maxindex-1).*N_a*N_z';
    Ftemp=ReturnMatrix(tempmaxindex);
        
    VKrondist=VKron-VKronold; VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        tempPolicyIndexes=PolicyIndexes+kron((0:1:N_z-1)'*N_a,ones(N_a,1));
        for Howards_counter=1:Howards
            EVKrontemp=aaa3.*VKronold';
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(EVKrontemp,[N_z,N_a,N_z]);
            EVKrontemp=sum(EVKrontemp,3);
            EVKrontemp=reshape(EVKrontemp,[N_z,N_a]); % z by aprime
            VKron=Ftemp+beta*EVKrontemp(tempPolicyIndexes);
        end
    end
    
%     if Verbose==1
%         if rem(tempcounter,100)==0
%             disp(tempcounter)
%             disp(currdist)
%         end
%         tempcounter=tempcounter+1;
%     end
    tempcounter=tempcounter+1

end

Policy=PolicyIndexes;

VKron=reshape(VKron,[N_a,N_z]);
Policy=reshape(Policy,[N_a,N_z]);

end