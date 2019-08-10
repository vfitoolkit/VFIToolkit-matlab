function [VKron, Policy]=ValueFnIter_Case1_LowMem_Par1_raw(VKron, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, beta, ReturnFn, ReturnFnParamsVec, Howards,Howards2, Tolerance,Verbose)

if Verbose==1
    disp('Starting Value Fn Iteration')
    tempcounter=1;
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

d_gridvals=CreateGridvals(n_d,d_grid,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);
z_gridvals=CreateGridvals(n_z,z_grid,1);

PolicyIndexes1=zeros(N_a,N_z);
PolicyIndexes2=zeros(N_a,N_z);

Ftemp_UsedForHowards=zeros([N_a,N_z]);

tempcounter=1;
currdist=Inf;

while currdist>Tolerance
    
    VKronold=VKron;
    
    parfor z_c=1:N_z
%         [z_c,N_z]
        z_val=z_gridvals(z_c,:);
        pi_z_z=pi_z(z_c,:);
        VKron_z=zeros(N_a,1);
        PolicyIndexes1_z=zeros(N_a,1);
        PolicyIndexes2_z=zeros(N_a,1);
        Ftemp_UsedForHowards_z=zeros(N_a,1);
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=VKronold.*kron(pi_z_z(1,:),ones(N_a,1));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        entireEV_z=kron(EV_z,ones(N_d,1));
        
        for a_c=1:N_a
%             [a_c, N_a]
            %Calc the RHS
            a_val=a_gridvals(a_c,:);
            Fmatrix_az=CreateReturnFnMatrix_Case1_LowMem_Disc(ReturnFn, ReturnFnParamsVec,a_val,z_val, n_d, n_a, d_gridvals, a_gridvals, 0);
            entireRHS=Fmatrix_az+beta*entireEV_z; %d by aprime by 1
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS);
            VKron_z(a_c)=Vtemp;
            PolInd_temp=ind2sub_homemade([N_d,N_a],maxindex); %[d;aprime]
            PolicyIndexes1_z(a_c,1)=PolInd_temp(1);
            PolicyIndexes2_z(a_c,1)=PolInd_temp(2);
            Ftemp_UsedForHowards_z(a_c,1)=Fmatrix_az(maxindex);
        end
        
        VKron(:,z_c)=VKron_z;
        PolicyIndexes1(:,z_c)=PolicyIndexes1_z;
        PolicyIndexes2(:,z_c)=PolicyIndexes2_z;
        Ftemp_UsedForHowards(:,z_c)=Ftemp_UsedForHowards_z;
    end
    
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    if isfinite(currdist) && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            for z_c=1:N_z
                EVKrontemp_z=VKrontemp(PolicyIndexes2(:,z_c),:).*kron(pi_z(z_c,:),ones(N_a,1)); %kron(pi_z(z_c,:),ones(nquad,1))
                EVKrontemp_z(isnan(EVKrontemp_z))=0; %Multiplying zero (transition prob) by -Inf (value fn) gives NaN
                VKron(:,z_c)=Ftemp_UsedForHowards(:,z_c)+beta*sum(EVKrontemp_z,2);
            end
        end
    end
    
    if Verbose==1
%         if rem(tempcounter,10)==0
            disp(tempcounter)
            disp(currdist)
%         end
    end
    tempcounter=tempcounter+1;

end

Policy=zeros(2,N_a,N_z);
Policy(1,:,:)=shiftdim(PolicyIndexes1,-1);
Policy(2,:,:)=shiftdim(PolicyIndexes2,-1);
% Policy(1,:,:)=permute(PolicyIndexes1,[3,1,2]);
% Policy(2,:,:)=permute(PolicyIndexes2,[3,1,2]);

end