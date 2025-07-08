function [VKron, Policy]=ValueFnIter_Case1_raw(VKron, n_d,n_a,n_z, pi_z, beta, ReturnMatrix, Howards,Howards2, Tolerance) %Verbose,

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% Policy=zeros(N_a,N_z,'gpuArray');
% Ftemp=zeros(N_a,N_z,'gpuArray');

pi_z_alt=shiftdim(pi_z',-1);
pi_z_howards=repelem(pi_z,N_a,1);

addindexforaz=gpuArray(shiftdim((0:1:N_a-1)*N_d*N_a+shiftdim((0:1:N_z-1)*N_d*N_a*N_a,-1),1));

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireRHS=ReturnMatrix+beta*repelem(EV,N_d,1); % z dimension of EV will autoexpand

    % Calc the max and it's index
    [VKron,PolicyIndexes]=max(entireRHS,[],1);

    PolicyIndexes=shiftdim(PolicyIndexes,1);

    tempmaxindex=PolicyIndexes+addindexforaz; % aprime index, add the index for a and z
    Ftemp=ReturnMatrix(tempmaxindex); % keep return function of optimal policy for using in Howards

    VKron=shiftdim(VKron,1); % a by z

    % Update currdist
    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        Policy_aprime=ceil(PolicyIndexes(:)/N_d);
        for Howards_counter=1:Howards
            EVKrontemp=VKron(Policy_aprime,:);
            
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;
    
end


Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

end
