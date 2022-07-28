function [VKron, Policy]=ValueFnIter_Case1_LowMem2_Refine_Par2_raw(VKron, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, beta, ReturnFn, ReturnFnParams, Howards,Howards2,Tolerance)
% Does pretty much exactly the same as ValueFnIter_Case1, only without any decision variable (n_d=0)
% Refinement is that we just solve for d*(aprime,a,z), then solve actual
% value fn without d, then add the optimal d at the end.

N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray'); % Note: This is just aprime (d is handled by refinement)

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

%%
l_a=length(n_a);
% l_z=length(n_z);

%%
a_gridvals=CreateGridvals(n_a,a_grid,1);
% z_gridvals=CreateGridvals(n_z,z_grid,1);

%% Refinement: calculate ReturnMatrix and 'remove' the d dimension
% ReturnMatrix=zeros(N_a,N_a*N_z); % 'refined' return matrix
% dstar=zeros(N_a,N_a*N_z);
% for az_c=1:N_a*N_z
%     a_c=rem(az_c-1,N_a)+1;
%     z_c=ceil(az_c/N_a);
%     avals=a_gridvals(a_c,:);
%     zvals=z_gridvals(z_c,:);
%     ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2_LowMem2(ReturnFn, n_d, n_a, ones(l_a,1),ones(l_z,1), d_grid, a_grid, avals, zvals,ReturnFnParams,1); % note: first n_a is n_aprime % the 1 at the end if to outpuit for refine
%     [ReturnMatrix_az,dstar_az]=max(ReturnMatrix_az,[],1); % solve for dstar
%     ReturnMatrix(:,az_c)=shiftdim(ReturnMatrix_az,1);
%     dstar(:,az_c)=shiftdim(dstar_az,1);
% end
% ReturnMatrix=reshape(ReturnMatrix,[N_a,N_a,N_z]);
% dstar=reshape(dstar,[N_a,N_a,N_z]);

ReturnMatrix=zeros(N_a,N_a,N_z); % 'refined' return matrix
dstar=zeros(N_a,N_a,N_z);
for a_c=1:N_a
    avals=a_gridvals(a_c,:);
    ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_Par2_LowMem2(ReturnFn, n_d, n_a, ones(l_a,1),n_z, d_grid, a_grid, avals, z_grid,ReturnFnParams,1); % note: first n_a is n_aprime % the 1 at the end if to outpuit for refine
    [ReturnMatrix_a,dstar_a]=max(ReturnMatrix_a,[],1); % solve for dstar
    ReturnMatrix(:,a_c,:)=shiftdim(ReturnMatrix_a,1);
    dstar(:,a_c,:)=shiftdim(dstar_a,1);
end



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
                
        entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a);%,1); %aprime by 1
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
        
        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
    end
    
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            
            EVKrontemp=VKrontemp(PolicyIndexes,:);
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*EVKrontemp;
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

% Add d back in (to implement refinement)
Policy=zeros(2,N_a,N_z);
Policy(2,:,:)=shiftdim(PolicyIndexes,-1);
temppolicyindex=reshape(PolicyIndexes,[1,N_a*N_z])+(0:1:N_a*N_z-1)*N_a;
Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]);


end