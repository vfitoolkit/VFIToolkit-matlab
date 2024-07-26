function [VKron,Policy]=ValueFnIter_Case1_Refine(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions)
% When using refinement, lowmemory is implemented in the first state (return fn) but not the second (the actual iteration).

N_a=prod(n_a);
N_z=prod(n_z);

%% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
% Since the return function is independent of time creating it once and
% then using it every iteration is good for speed, but it does use a
% lot of memory.
if isfield(vfoptions,'statedependentparams')
    error('statedependentparams does not work with solnmethod purediscretization_refinement \n')
    dbstack
end

if vfoptions.lowmemory==0
    if vfoptions.returnmatrix==0     % On CPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec,1);
    elseif vfoptions.returnmatrix==1
        ReturnMatrix=ReturnFn;
    elseif vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec,1);
    end
    
    % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
    if n_d(1)>0
        [ReturnMatrix,dstar]=max(ReturnMatrix,[],1);
        ReturnMatrix=shiftdim(ReturnMatrix,1);
    end
elseif vfoptions.lowmemory==1 % loop over z
    %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
    ReturnMatrix=zeros(N_a,N_a,N_z); % 'refined' return matrix
    dstar=zeros(N_a,N_a,N_z);
    l_z=length(n_z);
    if all(size(z_grid)==[sum(n_z),1])
        z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(z_grid)==[prod(n_z),l_z])
        z_gridvals=z_grid;
    end
    n_z_temp=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn,n_d, n_a, n_z_temp,d_grid, a_grid, zvals,ReturnFnParamsVec,1); % the 1 at the end is to output for refine
        [ReturnMatrix_z,dstar_z]=max(ReturnMatrix_z,[],1); % solve for dstar
        ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end
elseif vfoptions.lowmemory==2 % loop over z and a
    ReturnMatrix=zeros(N_a,N_a,N_z); % 'refined' return matrix
    dstar=zeros(N_a,N_a,N_z);
    a_gridvals=CreateGridvals(n_a,a_grid,1);
    l_z=length(n_z);
    if all(size(z_grid)==[sum(n_z),1])
        z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(z_grid)==[prod(n_z),l_z])
        z_gridvals=z_grid;
    end
    n_a_temp=ones(1,l_a);
    n_z_temp=ones(1,l_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        for a_c=1:N_a
            avals=a_gridvals(a_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2_LowMem2(ReturnFn, n_d, n_a, n_a_temp, n_z_temp, d_grid, a_grid, avals, zvals,ReturnFnParamsVec,1); % the 1 at the end is to output for refine
            [ReturnMatrix_az,dstar_az]=max(ReturnMatrix_az,[],1); % solve for dstar
            ReturnMatrix(:,a_c,z_c)=shiftdim(ReturnMatrix_az,1);
            dstar(:,a_c,z_c)=shiftdim(dstar_az,1);
        end
    end

end

if vfoptions.verbose==1
    time=toc;
    fprintf('Time to create return fn matrix: %8.4f \n', time)
    fprintf('Starting Value Function \n')
    tic;
end

%%
% V0=reshape(V0,[N_a,N_z]);

% Refinement essentially just ends up using the NoD case to solve the value function once we have the return matrix
if vfoptions.parallel==0     % On CPU
    [VKron,Policy_a]=ValueFnIter_Case1_NoD_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
elseif vfoptions.parallel==1 % On Parallel CPU
    [VKron,Policy_a]=ValueFnIter_Case1_NoD_Par1_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
elseif vfoptions.parallel==2 % On GPU
    [VKron,Policy_a]=ValueFnIter_Case1_NoD_Par2_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance,vfoptions.maxiter); %  a_grid, z_grid,
end

%% For refinement, add d to Policy
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
