function [VKron, Policy]=ValueFnIter_Case1_QuasiHyperbolicNaive_LowMem_NoD_Par2_raw(VKron, n_a, n_z, pi_z, beta0beta, ReturnFn, ReturnFnParamsVec) 
% (last two entries of) DiscountFactorParamNames contains the names for the two parameters relating to
% Quasi-hyperbolic preferences.
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the
% future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j: Vtilde_j = u_t+ beta_0 *E[V_{j+1}]
% See documentation for a fuller explanation of this.

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a,N_z,'gpuArray');

l_z=length(n_z);
z_gridvals=CreateGridvals(n_z,z_grid,1);


VKronold=VKron;

for z_c=1:N_z
    zvals=z_gridvals(z_c,:);
    ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn,0, n_a, ones(l_z,1),0, a_grid, zvals,ReturnFnParamsVec);
    
    %Calc the condl expectation term (except beta), which depends on z but
    %not on control variables
    EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV_z=sum(EV_z,2);
    
    entireRHS=ReturnMatrix_z+beta0beta*EV_z*ones(1,N_a);%,1); %aprime by 1
    
    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    VKron(:,z_c)=Vtemp;
    Policy(:,z_c)=maxindex;
end

end