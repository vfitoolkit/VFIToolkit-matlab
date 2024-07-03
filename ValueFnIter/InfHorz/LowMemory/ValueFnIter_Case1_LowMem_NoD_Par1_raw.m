function [VKron, Policy]=ValueFnIter_Case1_LowMem_NoD_Par1_raw(VKron, n_a, n_z, a_grid,z_grid, pi_z, beta, ReturnFn, ReturnFnParamsVec, Howards, Howards2,Tolerance, Verbose)
%Does pretty much exactly the same as ValueFnIter_Case1, only without any
%decision variable (n_d=0)

% disp('WARNING: ValueFnIter_Case1_LowMem_NoD_Par1_raw is out of date')
% Note: pretty sure this warning was just about how Howards was being done
% and is now fixed.

N_a=prod(n_a);
N_z=prod(n_z);

if Verbose==1
    disp('Starting Value Fn Iteration')
    tempcounter=1;
end

PolicyIndexes=zeros(N_a,N_z);
currdist=Inf;

a_gridvals=CreateGridvals(n_a,a_grid,1);
l_z=length(n_z);
if all(size(z_grid)==[sum(n_z),1])
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
elseif all(size(z_grid)==[prod(n_z),l_z])
    z_gridvals=z_grid;
end

Ftemp_UsedForHowards=zeros([N_a,N_z]);

tempcounter=1;
while currdist>Tolerance

    VKronold=VKron;

    parfor z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        
        pi_z_z=pi_z(z_c,:);
        VKron_z=zeros(N_a,1);
        PolicyIndexes_z=zeros(N_a,1);
        Ftemp_UsedForHowards_z=zeros(N_a,1);

        EV_z=VKronold.*kron(ones(N_a,1),pi_z_z(1,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        for a_c=1:N_a
            %Calc the RHS
            a_val=a_gridvals(a_c,:);
            Fmatrix_az=CreateReturnFnMatrix_Case1_LowMem_NoD_Disc(a_val,z_val,ReturnFn, ReturnFnParamsVec, n_a, a_gridvals, 0);
            entireRHS=Fmatrix_az+beta*EV_z; %d by aprime by 1

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS);
            VKron_z(a_c)=Vtemp;
            PolicyIndexes_z(a_c)=maxindex;
            Ftemp_UsedForHowards_z(a_c,1)=Fmatrix_az(maxindex);
        end
        
        VKron(:,z_c)=VKron_z;
        PolicyIndexes(:,z_c)=PolicyIndexes_z;
        Ftemp_UsedForHowards(:,z_c)=Ftemp_UsedForHowards_z;
    end
    
    VKrondist=reshape(VKron-VKronold,[numel(VKron),1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            for z_c=1:N_z % WHY NOT USE parfor HERE?
                EVKrontemp_z=VKrontemp(PolicyIndexes(:,z_c),:).*kron(pi_z(z_c,:),ones(N_a,1)); %kron(pi_z(z_c,:),ones(nquad,1))
                EVKrontemp_z(isnan(EVKrontemp_z))=0; %Multiplying zero (transition prob) by -Inf (value fn) gives NaN
                VKron(:,z_c)=Ftemp_UsedForHowards(:,z_c)+beta*sum(EVKrontemp_z,2);
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