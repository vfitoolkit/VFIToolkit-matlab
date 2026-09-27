function    [VKron,Policy]=ValueFnIter_InfHorz_EndoVFI(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions,n_SDP,SDP1,SDP2,SDP3)
% Uses Endogenous VFI instead of VFI: see Bray (2019) - Markov Decision Processes with Exogenous Variables
% From that paper, point 15 on pg 4599: Vendogenous=Lambda * V is the
% endogenous value fn, the length-X*Z vector with ith element
% v(xi,yi)-sum_y v(xi,y)/Y.

if vfoptions.lowmemory==0
    %% CreateReturnFnMatrix_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a
    % lot of memory.

    d_gridvals=CreateGridvals(n_d,d_grid,1); % the GPU builder wants gridvals, not the stacked grid

    if vfoptions.verbose==1
        disp('Creating return fn matrix')
        tic;
    end

    if isfield(vfoptions,'statedependentparams')
        if n_SDP==3
            ReturnMatrix=CreateReturnFnMatrix_Disc_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec,SDP1,SDP2,SDP3);
        elseif n_SDP==2
            ReturnMatrix=CreateReturnFnMatrix_Disc_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec,SDP1,SDP2);
        elseif n_SDP==1
            ReturnMatrix=CreateReturnFnMatrix_Disc_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec,SDP1);
        end
    else % Following is the normal/standard behavior
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_grid, ReturnFnParamsVec,0);
    end

    if vfoptions.verbose==1
        time=toc;
        fprintf('Time to create return fn matrix: %8.4f \n', time)
        disp('Starting Value Function')
        tic;
    end

    %%
    if n_d(1)==0
        if vfoptions.parallel==2 % On GPU
            [VKron,Policy]=ValueFnIter_InfHorz_EndoVFI_Case1_nod_Par2_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance); %  a_grid, z_grid,
        end
    else
        if vfoptions.parallel==2 % On GPU
            [VKron, Policy]=ValueFnIter_InfHorz_EndoVFI_Case1_Par2_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        end
    end
end

end