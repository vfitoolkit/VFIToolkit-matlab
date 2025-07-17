function [VKron,Policy]=ValueFnIter_Refine(V0,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions)
% When using refinement, lowmemory is implemented in the first state (return fn) but not the second (the actual iteration).

N_a=prod(n_a);
N_z=prod(n_z);

%% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
% Since the return function is independent of time creating it once and
% then using it every iteration is good for speed, but it does use a
% lot of memory.

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, ReturnFnParamsVec,1);
    
    % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
    if n_d(1)>0
        [ReturnMatrix,dstar]=max(ReturnMatrix,[],1);
        ReturnMatrix=shiftdim(ReturnMatrix,1);
    end
elseif vfoptions.lowmemory==1 % loop over z
    %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
    ReturnMatrix=zeros(N_a,N_a,N_z,'gpuArray'); % 'refined' return matrix
    dstar=zeros(N_a,N_a,N_z,'gpuArray');
    l_z=length(n_z);
    special_n_z=ones(1,l_z,'gpuArray');
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn,n_d, n_a, special_n_z,d_gridvals, a_grid, zvals,ReturnFnParamsVec,1); % the 1 at the end is to output for refine
        [ReturnMatrix_z,dstar_z]=max(ReturnMatrix_z,[],1); % solve for dstar
        ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end
end

%%
% V0=reshape(V0,[N_a,N_z]);

% Refinement essentially just ends up using the 'no decision variable (nod)' case to solve the value function once we have the return matrix and refine out d
if N_a<400 || N_z<20
    [VKron,Policy_a]=ValueFnIter_nod_HowardGreedy_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter);
else
    [VKron,Policy_a]=ValueFnIter_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter);
end

%% For refinement, add d into Policy
% Policy is currently
if n_d(1)>0
    Policy=zeros(2,N_a,N_z);
    Policy(2,:,:)=shiftdim(Policy_a,-1);
    temppolicyindex=reshape(Policy_a,[1,N_a*N_z])+(0:1:N_a*N_z-1)*N_a;
    Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]);
else
    Policy=Policy_a;
end



end
