function [VKron, Policy]=ValueFnIterEndo_Case1_NoD_Par2_raw(VKron, n_a, n_z, pi_z, beta, ReturnMatrix, Howards,Howards2,Tolerance) % Verbose, a_grid, z_grid, 
% Does pretty much exactly the same as ValueFnIterEndo_Case1, only without any decision variable (n_d=0)
%
% Uses Endogenous VFI instead of VFI: see Bray (2019) - Markov Decision Processes with Exogenous Variables
% From that paper, point 15 on pg 4599: Vendogenous=Lambda * V is the
% endogenous value fn, the length-X*Z vector with ith element
% v(xi,yi)-sum_y v(xi,y)/Y.

N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

VKron=VKron-sum(VKron,2)/N_z; % Change value function into relative value function

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
    for z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
                
        entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1); %aprime by a
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
        
        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
    end
    
    % Switch these into relative value function
    VKron=VKron-sum(VKron,2)/N_z;
    
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            EVKrontemp=VKron(PolicyIndexes,:);
            
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*EVKrontemp;
        end
        % Switch into relative value function
        VKron=VKron-sum(VKron,2)/N_z;
    end
    

%     if Verbose==1
%         if rem(tempcounter,100)==0
            disp(tempcounter)
            disp(currdist)
%         end
%         tempcounter=tempcounter+1;
%     end
    tempcounter=tempcounter+1;

end

Policy=PolicyIndexes;

%% Create V in levels from the endogenous value function
% Involves doing one more iteration of the value function
% (I could probably just do it from the existing VKronold and VKron, but to
% ensure the expected level of accuracy I have done one more iteration as
% that is what the formal math does (see Bray, 2019).
VKronold=VKron;
for z_c=1:N_z
    ReturnMatrix_z=ReturnMatrix(:,:,z_c);
    %Calc the condl expectation term (except beta), which depends on z but
    %not on control variables
    EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV_z=sum(EV_z,2);
    
    entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1); %aprime by a
    
    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    VKron(:,z_c)=Vtemp;
    PolicyIndexes(:,z_c)=maxindex;
    
    tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
    Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex);
end
% Now V*=VE*+(I_z-beta*pi_z kron Xi)^(-1))*TVE*, where Xi is an N_a-by-N_a matrix of ones divided by N_a
VKron=VKronold+kron((eye(N_z)-beta*pi_z)^(-1),ones(N_a,N_a)/N_a)*VKron;
% NOTE TO SELF: DO I HAVE THE KRON THE RIGHT WAY AROUND

end