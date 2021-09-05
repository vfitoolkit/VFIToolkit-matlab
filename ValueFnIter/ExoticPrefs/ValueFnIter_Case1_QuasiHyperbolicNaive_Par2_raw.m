function [VKron, Policy]=ValueFnIter_Case1_QuasiHyperbolicNaive_Par2_raw(VKron, n_d,n_a,n_z, pi_z, beta0beta, ReturnMatrix) %Verbose,
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the
% future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j: Vtilde_j = u_t+ beta_0 *E[V_{j+1}]
% See documentation for a fuller explanation of this.

% Note that the inputed VKron is already V, so just calcule Vtilde and
% associated policy.

% Note that beta here is the 'this period to next period' discount function
% of the quasi-hyperbolic discounting.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a,N_z,'gpuArray');

%%
VKronold=VKron;

for z_c=1:N_z
    ReturnMatrix_z=ReturnMatrix(:,:,z_c);
    %Calc the condl expectation term (except beta), which depends on z but
    %not on control variables
    EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV_z=sum(EV_z,2);
    
    entireEV_z=kron(EV_z,ones(N_d,1));
    entireRHS=ReturnMatrix_z+beta0beta*entireEV_z*ones(1,N_a,1);
    
    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    VKron(:,z_c)=Vtemp;
    Policy(:,z_c)=maxindex;
end

Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(Policy/N_d),-1);

end