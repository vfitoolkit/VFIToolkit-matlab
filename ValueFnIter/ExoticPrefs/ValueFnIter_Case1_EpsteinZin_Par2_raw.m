function [VKron, Policy]=ValueFnIter_Case1_EpsteinZin_Par2_raw(VKron, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, Howards,Howards2, Tolerance) %Verbose,
% DiscountFactorParamsVec contains the three parameters relating to
% Epstein-Zin preferences. Calling them beta, gamma, and psi,
% respectively the Epstein-Zin preferences are given by
% U_t= [(1-beta) u_t^(1-1/psi) + beta (E[(U_{t+1}^(1-gamma)])^((1-1/psi)/(1-gamma))]^(1/(1-1/psi))
% where
%  u_t is per-period utility function
% See eg., Caldara, Fernandez-Villaverde, Rubio-Ramirez, and Yao (2012)

%%
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

% Modify the Return Function appropriately for Epstein-Zin Preferences
% temp2=ReturnMatrix;
% temp2(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(3));
% temp2=(1-DiscountFactorParamsVec(1))*temp2;
for z_c=1:N_z % This slows code down, but removes what was otherwise a bottleneck for memory usage.
    ReturnMatrix_z=ReturnMatrix(:,:,z_c);
    ReturnMatrix_z(isfinite(ReturnMatrix_z))=ReturnMatrix_z(isfinite(ReturnMatrix_z)).^(1-1/DiscountFactorParamsVec(3));
    ReturnMatrix_z=(1-DiscountFactorParamsVec(1))*ReturnMatrix_z;
    ReturnMatrix(:,:,z_c)=ReturnMatrix_z;
end
% ReturnMatrix(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(3));
% ReturnMatrix=(1-DiscountFactorParamsVec(1))*ReturnMatrix;

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
%     tic;
    for z_c=1:N_z
%         ReturnMatrix_z=ReturnMatrix(:,:,z_c);    
        temp2_z=ReturnMatrix(:,:,z_c);
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        temp=VKronold;
        temp(isfinite(VKronold))=VKronold(isfinite(VKronold)).^(1-DiscountFactorParamsVec(2));
        temp(VKronold==0)=0;
         % When using GPU matlab objects to switching between real and
         % complex numbers when evaluating powers. Using temp avoids this
         % issue.
        EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
%        EV_z=(VKronold.^(1-DiscountFactorParamsVec(2))).*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireEV_z=kron(EV_z,ones(N_d,1));
%         temp2=ReturnMatrix_z;
%         temp2(isfinite(ReturnMatrix_z))=ReturnMatrix_z(isfinite(ReturnMatrix_z)).^(1-1/DiscountFactorParamsVec(3));
        temp3=entireEV_z*ones(1,N_a,1); 
        temp4=temp3;
        temp4(isfinite(temp4))=temp4(isfinite(temp4)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
        temp4(temp3==0)=0;
        entireRHS=temp2_z+DiscountFactorParamsVec(1)*temp4;
%        entireRHS=( (1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec(1)*temp4 );
        % No need to compute the .^(1/(1-1/DiscountFactorParamsVec(3))) of
        % the whole entireRHS. This will be a monotone function, so just find the max, and
        % then compute .^(1/(1-1/DiscountFactorParamsVec(3))) of the max.
%         entireRHS(isfinite(entireRHS))=entireRHS(isfinite(entireRHS)).^(1/(1-1/DiscountFactorParamsVec(3)));
%        entireRHS=( (1-DiscountFactorParamsVec(1))*ReturnMatrix_z.^(1-1/DiscountFactorParamsVec(3))+DiscountFactorParamsVec(1)*(entireEV_z*ones(1,N_a,1).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)))) ).^(1/(1-1/DiscountFactorParamsVec(3)));

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        VKron(:,z_c)=Vtemp.^(1/(1-1/DiscountFactorParamsVec(3)));
%         VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
             
        tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
%         Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
        Ftemp(:,z_c)=temp2_z(tempmaxindex); 
    end
%     time1=toc;
% 
%     tic;
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?
%     time2=toc;
%     tic;


    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
%             VKrontemp=VKron;
%             EVKrontemp=VKrontemp(ceil(PolicyIndexes/N_d),:);
            EVKrontemp=VKron(ceil(PolicyIndexes/N_d),:);
            
            EVKrontemp=(EVKrontemp.^(1-DiscountFactorParamsVec(2))).*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            
            temp3=EVKrontemp;
            temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
            temp3(EVKrontemp==0)=0;
            
            % Note that Ftemp already includes all the relevant Epstein-Zin modifications
            VKron=(Ftemp+DiscountFactorParamsVec(1)*temp3).^(1/(1-1/DiscountFactorParamsVec(3)));
        end
    end
    
%     time3=toc;
    
%     if vfoptions.verbose==1
%         if rem(tempcounter,10)==0
%             disp(tempcounter)
%             disp(currdist)
%             fprintf('times: %2.8f, %2.8f, %2.8f \n',time1,time2,time3)
%         end
%     end

    tempcounter=tempcounter+1;
    
end

Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

end