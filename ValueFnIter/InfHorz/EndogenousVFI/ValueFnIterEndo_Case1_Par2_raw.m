function [VKron, Policy]=ValueFnIterEndo_Case1_Par2_raw(VKron, n_d,n_a,n_z, pi_z, beta, ReturnMatrix, Howards,Howards2, Tolerance) %Verbose,
% Uses Endogenous VFI instead of VFI: see Bray (2019) - Markov Decision Processes with Exogenous Variables
% From that paper, point 15 on pg 4599: Vendogenous=Lambda * V is the
% endogenous value fn, the length-X*Z vector with ith element
% v(xi,yi)-sum_y v(xi,y)/Y.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);
% I suspect but have not yet double-checked that could instead just use
% aaa=kron(ones(N_a,1,'gpuArray'),pi_z);

VKron=VKron-sum(VKron,2)/N_z; % Change value function into endogenous value function

% beta=DiscountFactorParamsVec
% VKron=V0;
% Howards=vfoptions.howards
% Howards2=vfoptions.maxhowards
% Tolerance=vfoptions.tolerance

%%
tempcounter=1;
currdist=Inf;
maxiter=2000
while currdist>Tolerance && tempcounter<maxiter
    VKronold=VKron;
    PolicyIndexesold=PolicyIndexes;
    
    for z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);     
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
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
        
    % Switch these into relative value function
    VKron=VKron-sum(VKron,2)/N_z;
    
%     VKrondist=reshape(abs(VKron-VKronold)./abs(VKron),[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    VKrondist=reshape(abs(VKron-VKronold),[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    [currdist,extra]=max(abs(VKrondist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?

%     if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
%         for Howards_counter=1:Howards
%             EVKrontemp=VKron(ceil(PolicyIndexes/N_d),:);
%             
%             EVKrontemp=EVKrontemp.*aaa;
%             EVKrontemp(isnan(EVKrontemp))=0;
%             EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
%             VKron=Ftemp+beta*EVKrontemp;
%         end
%         % Switch these into relative value function
%         VKron=VKron-sum(VKron,2)/N_z;
%     end
    
%     if Verbose==1
%         if rem(tempcounter,100)==0
            disp(tempcounter)
            disp(currdist)
            fprintf('max diff is at index %i \n',extra)
            [VKron(extra),VKronold(extra)]
            fprintf('raw value of the max diff is %8.12f \n',VKron(extra)-VKronold(extra))
            fprintf('number of iterations %i \n', tempcounter)
            figure(1)
            surf(VKron)
%             fprintf('times: %2.8f, %2.8f, %2.8f \n',time1,time2,time3)
%         end
%         
%         tempcounter=tempcounter+1;
%     end

    PolicyDist=reshape(PolicyIndexes-PolicyIndexesold,[N_a*N_z,1]);
    currPolicyDist=max(abs(PolicyDist));
    fprintf('currPolicyDist is %i \n',currPolicyDist)

    tempcounter=tempcounter+1;
    
end

Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

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
    EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
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
% Now V*=VE*+(I_z-beta*pi_z kron Xi)^(-1))*TVE*, where Xi is an N_a-by-N_a matrix of ones divided by N_a
size(VKronold)
temp1=(eye(N_z)-beta*pi_z)^(-1);
size(temp1)
size(ones(N_a,N_a)/N_a)
temp=kron(temp1,ones(N_a,1)/N_a);
size(temp)
size(VKron)
temp2=kron(ones(N_z,1),VKron);
size(temp2)
temp3=temp.*temp2;
fprintf('Size of temp3 \n')
size(temp3)
temp4=reshape(sum(temp3,2),[N_a,N_z]);
size(temp4)
% VKron=VKronold+sum(kron((eye(N_z)-beta*pi_z)^(-1),ones(N_a,1)/N_a).*kron(ones(N_z,1),VKron),2);
VKron=VKronold+temp4;
% NOTE TO SELF: DO I HAVE THE KRON THE RIGHT WAY AROUND

end