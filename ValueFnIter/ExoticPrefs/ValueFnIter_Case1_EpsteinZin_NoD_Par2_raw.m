function [VKron, Policy]=ValueFnIter_Case1_EpsteinZin_NoD_Par2_raw(VKron, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, Howards,Howards2,Tolerance) % Verbose, a_grid, z_grid, 
% Does pretty much exactly the same as ValueFnIter_Case1_EpsteinZin, only without any
% decision variable (n_d=0)
%
% DiscountFactorParamsVec contains the three parameters relating to
% Epstein-Zin preferences. Calling them beta, gamma, and psi,
% respectively the Epstein-Zin preferences are given by
% U_t= [ (1-beta)*u_t^(1-1/psi) + beta (E[(U_{t+1}^(1-gamma)])^((1-1/psi)/(1-gamma))]^(1/(1-1/psi))
% where
%  u_t is per-period utility function
% See eg., Caldara, Fernandez-Villaverde, Rubio-Ramirez, and Yao (2012)

N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

% Modify the Return Function appropriately for Epstein-Zin Preferences
temp2=ReturnMatrix;
temp2(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(3));
temp2=(1-DiscountFactorParamsVec(1))*temp2;

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
    for z_c=1:N_z
%         ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        temp2_z=temp2(:,:,z_c);
        
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
%         EV_z=(VKronold.^(1-DiscountFactorParamsVec(2))).*(ones(N_a,1,'gpuArray')*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
        temp=VKronold;
        temp(isfinite(VKronold))=VKronold(isfinite(VKronold)).^(1-DiscountFactorParamsVec(2));
        temp(VKronold==0)=0;
         % When using GPU matlab objects to switching between real and
         % complex numbers when evaluating powers. Using temp avoids this
         % issue.
        EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
                        
%         entireRHS=( (1-DiscountFactorParamsVec(1))*ReturnMatrix_z.^(1-1/DiscountFactorParamsVec(3))+DiscountFactorParamsVec(1)*(EV_z*ones(1,N_a,1).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)))) ).^(1/(1-1/DiscountFactorParamsVec(3))); %aprime by 1

        temp3=EV_z;
        temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
        temp3(EV_z==0)=0;
        entireRHS=temp2_z+DiscountFactorParamsVec(1)*temp3*ones(1,N_a,1);        

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS.^(1/(1-1/DiscountFactorParamsVec(3))),[],1);
        VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
        
        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=temp2_z(tempmaxindex); % note that temp2_z is the EZ ReturnMatrix_z
    end
    
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            %VKrontemp=VKron;
            %EVKrontemp=VKrontemp(PolicyIndexes,:);
            EVKrontemp=VKron(PolicyIndexes,:);
            
            EVKrontemp=(EVKrontemp.^(1-DiscountFactorParamsVec(2))).*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            
            temp3=EVKrontemp;
            temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
            temp3(EVKrontemp==0)=0;
            
            % Note that Ftemp already includes all the relevant Epstein-Zin modifications
            VKron=(Ftemp+DiscountFactorParamsVec(1)*temp3).^(1/(1-1/DiscountFactorParamsVec(3)));            
%             VKron=( (1-DiscountFactorParamsVec(1))*Ftemp.^(1-1/DiscountFactorParamsVec(3))+DiscountFactorParamsVec(1)*(EVKrontemp.^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)))) ).^(1/(1-1/DiscountFactorParamsVec(3)));
        end
    end

%     if Verbose==1
%         if rem(tempcounter,100)==0
%             disp(tempcounter)
%             disp(currdist)
%         end
%         tempcounter=tempcounter+1;
%     end
    tempcounter=tempcounter+1;

end

Policy=PolicyIndexes;



end