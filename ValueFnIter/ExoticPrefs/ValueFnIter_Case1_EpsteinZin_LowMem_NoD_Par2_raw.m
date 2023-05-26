function [VKron, Policy]=ValueFnIter_Case1_EpsteinZin_LowMem_NoD_Par2_raw(VKron, n_a, n_z, a_grid, z_grid, pi_z,DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, Howards,Howards2,Tolerance) % Verbose, ReturnFnParamNames, 
%Does pretty much exactly the same as ValueFnIter_Case1, only without any
%decision variable (n_d=0)

N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);


%%
l_z=length(n_z);
if all(size(z_grid)==[sum(n_z),1])
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
elseif all(size(z_grid)==[prod(n_z),l_z])
    z_gridvals=z_grid;
end

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn,0, n_a, ones(l_z,1),0, a_grid, zvals,ReturnFnParamsVec);
        ReturnMatrix_z(isfinite(ReturnMatrix_z))=ReturnMatrix_z(isfinite(ReturnMatrix_z)).^(1-1/DiscountFactorParamsVec(3));
        ReturnMatrix_z=(1-DiscountFactorParamsVec(1))*ReturnMatrix_z;
        
        temp=VKronold;
        temp(isfinite(VKronold))=VKronold(isfinite(VKronold)).^(1-DiscountFactorParamsVec(2));
        temp(VKronold==0)=0;
         % When using GPU matlab objects to switching between real and
         % complex numbers when evaluating powers. Using temp avoids this
         % issue.
        EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
                        
        temp3=EV_z;
        temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
        temp3(EV_z==0)=0;
        entireRHS=ReturnMatrix_z+DiscountFactorParamsVec(1)*temp3;
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS.^(1/(1-1/DiscountFactorParamsVec(3))),[],1);
        
        VKron(isfinite(Vtemp),z_c)=Vtemp(isfinite(Vtemp));
        VKron(~isfinite(Vtemp),z_c)=-Inf;
        PolicyIndexes(:,z_c)=maxindex;
        
        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
    end
    
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            EVKrontemp=VKron(PolicyIndexes,:);
            
            EVKrontemp(isfinite(EVKrontemp))=(EVKrontemp(isfinite(EVKrontemp)).^(1-DiscountFactorParamsVec(2)));
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            
            temp3=EVKrontemp;
            temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
            temp3(EVKrontemp==0)=0;
            
            % Note that Ftemp already includes all the relevant Epstein-Zin modifications
            VKron=(Ftemp+DiscountFactorParamsVec(1)*temp3); %.^(1/(1-1/DiscountFactorParamsVec(3))); 
            VKron(isfinite(VKron))=VKron(isfinite(VKron)).^(1/(1-1/DiscountFactorParamsVec(3)));
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