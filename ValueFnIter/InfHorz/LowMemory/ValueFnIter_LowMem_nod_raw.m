function [VKron, Policy]=ValueFnIter_LowMem_nod_raw(VKron, n_a, n_z, a_grid, z_gridvals, pi_z, beta, ReturnFn, ReturnFnParams, Howards,Howards2,Tolerance) % Verbose, ReturnFnParamNames,

l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

pi_z_howards=repelem(pi_z,N_a,1);

special_n_z=ones(l_z,1);

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_nod_Par2(ReturnFn, n_a, special_n_z, a_grid, zvals,ReturnFnParams);
        
        % Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=VKronold.*pi_z(z_c,:);
        EV_z(isnan(EV_z))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
                
        entireRHS=ReturnMatrix_z+beta*EV_z; %aprime by 1
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
        
        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
    end
    
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    if isfinite(currdist) && tempcounter<Howards2 % Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            
            EVKrontemp=VKrontemp(PolicyIndexes,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;
end

Policy=PolicyIndexes;



end
