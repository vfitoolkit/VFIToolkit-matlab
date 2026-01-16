function [VKron, Policy]=ValueFnIter_LowMem_sparse_nod_raw(VKron, n_a, n_z, a_grid, z_gridvals, pi_z, beta, ReturnFn, ReturnFnParams, Howards,MaxHowards,Tolerance,maxiter)

l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a,N_z,'gpuArray');
special_n_z=ones(l_z,1);

% Precompute variables for Howard improvement step
Ftemp          = zeros(N_a,N_z,'gpuArray');
pi_z_transpose = transpose(pi_z);
NaVec = gpuArray.colon(1,N_a)';
NzVec = gpuArray.colon(1,N_z)';
a_ind = repmat(NaVec,N_z,1);
z_ind = repelem(NzVec,N_a,1);
ind   = a_ind+(z_ind-1)*N_a;

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
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
        Policy(:,z_c)=maxindex;

        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 

    end %end z
    
    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    %disp(currdist)

    % Use Howards Policy Fn Iteration Improvement 
    % (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards
       
        Policy_vec = reshape(Policy,[N_a*N_z,1]);
        Ftemp_vec  = reshape(Ftemp,[N_a*N_z,1]);
        % Build large sparse matrix
        indp  = Policy_vec+(z_ind-1)*N_a;
        Qmat  = sparse(ind,indp,1,N_a*N_z,N_a*N_z);
        % Howard iterations, with Ftemp and Qmat precomputed
        for Howards_counter=1:Howards
            EV_howard = VKron*pi_z_transpose; % (a',z)
            EV_howard = reshape(EV_howard,[N_a*N_z,1]);
            VKron = Ftemp_vec+beta*Qmat*EV_howard;
            VKron = reshape(VKron,[N_a,N_z]);
        end
    end

    tempcounter=tempcounter+1;
end

Policy=reshape(Policy,[1,N_a,N_z]);



end %end function ValueFnIter_LowMem_nod_raw_sparse
