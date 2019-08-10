function [VKron, Policy]=ValueFnIter_Case1_LowMem_NoD_Par1_raw(VKron, n_a, n_z, a_grid,z_grid, pi_z, beta, ReturnFn, ReturnFnParamsVec, Howards, Howards2,Tolerance, Verbose)
%Does pretty much exactly the same as ValueFnIter_Case1, only without any
%decision variable (n_d=0)

disp('WARNING: ValueFnIter_Case1_LowMem_NoD_Par1_raw is out of date')

N_a=prod(n_a);
N_z=prod(n_z);

if Verbose==1
    disp('Starting Value Fn Iteration')
    tempcounter=1;
end

PolicyIndexes=zeros(N_a,N_z);
currdist=Inf;

[~, a_gridvals, z_gridvals]=CreateGridvals_BasicVersion(0,n_a,n_z,0,a_grid,z_grid,1); % No d variables

tempcounter=1;
while currdist>Tolerance

    VKronold=VKron;

    parfor z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        
        pi_z_z=pi_z(z_c,:);
        VKron_z=zeros(N_a,1);
        PolicyIndexes_z=zeros(N_a,1);

        EV_z=VKronold.*kron(ones(N_a,1),pi_z_z(1,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        for a_c=1:N_a
            entireRHS=CreateReturnFnMatrix_Case1_LowMem_NoD_Disc(a_gridvals(a_c,:),z_val,ReturnFn, ReturnFnParamsVec, n_a, a_gridvals, 0)+beta*EV_z; %aprime by 1
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS);
            VKron_z(a_c)=Vtemp;
            PolicyIndexes_z(a_c)=maxindex;
        end
        
        VKron(:,z_c)=VKron_z;
        PolicyIndexes(:,z_c)=PolicyIndexes_z;
    end
        
    VKrondist=reshape(VKron-VKronold,[numel(VKron),1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        Ftemp=zeros(N_a,N_z);
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                Ftemp(a_c,z_c)=ReturnFn(a_gridvals(PolicyIndexes(a_c,z_c),:),a_gridvals(a_c,:),z_val);%FmatrixKron(PolicyIndexes1(a_c,z_c),PolicyIndexes2(a_c,z_c),a_c,z_c);
            end
        end
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            for z_c=1:N_z
                EVKrontemp_z=VKrontemp(PolicyIndexes(:,z_c),:).*kron(pi_z(z_c,:),ones(N_a,1)); %kron(pi_z(z_c,:),ones(nquad,1))
                EVKrontemp_z(isnan(EVKrontemp_z))=0; %Multiplying zero (transition prob) by -Inf (value fn) gives NaN
                VKron(:,z_c)=Ftemp(:,z_c)+beta*sum(EVKrontemp_z,2);
            end
        end
    end
    
    if Verbose==1
        if rem(tempcounter,100)==0
            disp(tempcounter)
            disp(currdist)
        end
    end

    tempcounter=tempcounter+1;
end
  
Policy=PolicyIndexes;

end