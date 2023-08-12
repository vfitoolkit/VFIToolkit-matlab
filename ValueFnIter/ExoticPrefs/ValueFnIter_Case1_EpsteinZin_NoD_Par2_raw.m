function [VKron, Policy]=ValueFnIter_Case1_EpsteinZin_NoD_Par2_raw(VKron, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, Howards,Howards2,Tolerance, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7)
% Does pretty much exactly the same as ValueFnIter_Case1_EpsteinZin, only without any decision variable (n_d=0)

N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Modify the Return Function appropriately for Epstein-Zin Preferences
becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
temp2=ReturnMatrix;
temp2(becareful)=ReturnMatrix(becareful).^ezc2;
temp2(ReturnMatrix==0)=-Inf; % matlab otherwise puts 0 to negative power to infinity

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
    for z_c=1:N_z
%         ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        temp2_z=temp2(:,:,z_c);
        
        % Part of Epstein-Zin is before taking expectation
        temp=VKronold;
        temp(isfinite(VKronold))=(ezc4*VKronold(isfinite(VKronold))).^ezc5;
        temp(VKronold==0)=0;
        
        %Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2); % sum over z', leaving a singular second dimension
        
        temp4=EV_z;
        temp4(isfinite(temp4))=temp4(isfinite(temp4)).^ezc6;
        temp4(EV_z==0)=0;
        
        entireRHS_z=ezc1*temp2_z+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,1);

        temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
        entireRHS_z(temp5)=ezc1*entireRHS_z(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        entireRHS_z(entireRHS_z==0)=-Inf;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
        
        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=temp2_z(tempmaxindex); % note that temp2_z is the EZ ReturnMatrix_z
    end
    
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            EVKrontemp=VKron(PolicyIndexes,:);

            % Part of Epstein-Zin is before taking expectation
            temp=EVKrontemp;
            temp(isfinite(EVKrontemp))=(ezc4*EVKrontemp(isfinite(EVKrontemp))).^ezc5;
            temp(EVKrontemp==0)=0;
            
            EVKrontemp=temp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            
            temp4=EVKrontemp;
            temp4(isfinite(temp4))=temp4(isfinite(temp4)).^ezc6;
            temp4(EVKrontemp==0)=0;
            
            % Note that Ftemp already includes all the relevant Epstein-Zin modifications
            VKron=ezc1*Ftemp+ezc3*DiscountFactorParamsVec*temp4; 

            temp5=logical(isfinite(VKron).*(VKron~=0));
            VKron(temp5)=ezc1*VKron(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
            VKron(VKron==0)=-Inf;
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