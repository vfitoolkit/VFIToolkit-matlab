function [VKron, Policy]=ValueFnIter_Case1_LowMem_Par1_raw(VKron, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, beta, ReturnFn, Howards,Howards2, Tolerance)%,Verbose)

disp('WARNING: ValueFnIter_Case1_LowMem_raw is out of date')

% if Verbose==1
%     disp('Starting Value Fn Iteration')
%     tempcounter=1;
% end

PolicyIndexes1=zeros(N_a,N_z);
PolicyIndexes2=zeros(N_a,N_z);

z_gridvals=zeros(N_z,length(n_z));
for i1=1:N_z
    sub=zeros(1,length(n_z));
    sub(1)=rem(i1-1,n_z(1))+1;
    for ii=2:length(n_z)-1
        sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
    end
    sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
    
    if length(n_z)>1
        sub=sub+[0,cumsum(n_z(1:end-1))];
    end
    z_gridvals(i1,:)=z_grid(sub);
end
a_gridvals=zeros(N_a,length(n_a));
for i2=1:N_a
    sub=zeros(1,length(n_a));
    sub(1)=rem(i2-1,n_a(1))+1;
    for ii=2:length(n_a)-1
        sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
    end
    sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
    
    if length(n_a)>1
        sub=sub+[0,cumsum(n_a(1:end-1))];
    end
    a_gridvals(i2,:)=a_grid(sub);
end
d_gridvals=zeros(N_d,length(n_d));
for i3=1:N_d
    sub=zeros(1,length(n_d));
    sub(1)=rem(i1-1,n_d(1))+1;
    for ii=2:length(n_d)-1
        sub(ii)=rem(ceil(i3/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
    end
    sub(length(n_d))=ceil(i3/prod(n_d(1:length(n_d)-1)));
    
    if length(n_d)>1
        sub=sub+[0,cumsum(n_d(1:end-1))];
    end
    d_gridvals(i3,:)=d_grid(sub);
end


Ftemp_UsedForHowards=zeros(n_a,n_z);

tempcounter=1;
currdist=Inf;

while currdist>Tolerance
    
    VKronold=VKron;
    
    parfor z_c=1:N_z
        z=z_gridvals(z_c,:);
        pi_z_z=pi_z(z_c,:);
        VKron_z=zeros(N_a,1);
        PolicyIndexes1_z=zeros(N_a,1);
        PolicyIndexes2_z=zeros(N_a,1);
        Ftemp_UsedForHowards_z=zeros(n_a,1);
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=VKronold.*kron(pi_z_z(1,:),ones(N_a,1));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        entireEV_z=kron(EV_z,ones(N_d,1));
        
        for a_c=1:N_a
            %Calc the RHS
            a=a_gridvals(a_c,:);
            Fmatrix_az=CreateReturnFnMatrix_Case1_LowMem_Disc(ReturnFn,a,z, n_d, n_a, d_gridvals, a_gridvals, 0);
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
    
%     if Verbose==1
%         if rem(tempcounter,100)==0
%             disp(tempcounter)
%             disp(currdist)
%         end
%         tempcounter=tempcounter+1;
%     end
    tempcounter=tempcounter+1;

end

Policy=zeros(2,N_a,N_z);
Policy(1,:,:)=permute(PolicyIndexes1,[3,1,2]);
Policy(2,:,:)=permute(PolicyIndexes2,[3,1,2]);

end