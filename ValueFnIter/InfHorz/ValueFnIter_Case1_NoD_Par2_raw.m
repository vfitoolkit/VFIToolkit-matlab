function [VKron, Policy]=ValueFnIter_Case1_NoD_Par2_raw(VKron, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, Howards,Howards2,Tolerance,maxiter) % Verbose, a_grid, z_grid, 
%Does pretty much exactly the same as ValueFnIter_Case1, only without any
%decision variable (n_d=0)

N_a=prod(n_a);
N_z=prod(n_z);

% PolicyIndexes=zeros(N_a,N_z,'gpuArray');
% Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;
    
    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*shiftdim(pi_z',-1); %kron(ones(N_a,1),pi_z(z_c,:));
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %aprime by a by z

    %Calc the max and it's index
    [VKron,PolicyIndexes]=max(entireRHS,[],1);

    tempmaxindex=shiftdim(PolicyIndexes,1)+addindexforaz; % aprime index, add the index for a and z

    Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards

    PolicyIndexes=PolicyIndexes(:); % a by z (this shape is just convenient for Howards)
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 
        for Howards_counter=1:Howards
            EVKrontemp=VKron(PolicyIndexes,:);
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;

end

Policy=reshape(PolicyIndexes,[N_a,N_z]);



end