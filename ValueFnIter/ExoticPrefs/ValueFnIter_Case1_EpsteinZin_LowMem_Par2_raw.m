function [VKron, Policy]=ValueFnIter_Case1_EpsteinZin_LowMem_Par2_raw(VKron, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z,DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, Howards,Howards2,Tolerance) %Verbose,

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

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
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn,n_d, n_a, ones(l_z,1),d_grid, a_grid, zvals,ReturnFnParamsVec);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite but not zero
        temp2_z=ReturnMatrix_z;
        temp2_z(becareful)=ReturnMatrix_z(becareful).^ezc2; % Otherwise can get things like 0 to negative power equals infinity
        temp2_z(ReturnMatrix_z==0)=-Inf; % Otherwise these ReturnMatrix=zero points get a finite amount added to them (from expectations) and were mishandled later

        % Part of Epstein-Zin is before taking expectation
        temp=VKronold;
        temp(isfinite(VKronold))=(ezc4*VKronold(isfinite(VKronold))).^ezc5;
        temp(VKronold==0)=0;

        %Calc the expectation term (except beta)
        EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2); % sum over z', leaving a singular second dimension
        
        entireEV_z=repelem(EV_z,N_d,1);
        temp4=entireEV_z;
        temp4(isfinite(temp4))=temp4(isfinite(temp4)).^ezc6;
        temp4(entireEV_z==0)=0;
        
        entireRHS_z=ezc1*temp2_z+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,1);

        temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
        entireRHS_z(temp5)=ezc1*entireRHS_z(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        entireRHS_z(entireRHS_z==0)=-Inf; % Dont want to consider these

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
             
        tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
    end

    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?
    
    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            EVKrontemp=VKron(ceil(PolicyIndexes/N_d),:);

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
        % if rem(tempcounter,100)==0
        %     disp(tempcounter)
        %     disp(currdist)
        % end
        % tempcounter=tempcounter+1;
%     end
    tempcounter=tempcounter+1;
end

Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

end