function     [VKron,Policy]=ValueFnIter_Case1_RelativeVFI(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions,n_SDP,SDP1,SDP2,SDP3);
% Uses Relative VFI instead of VFI: see Bray (2019) - Strong Convergence and Dynamic Economic Models

if vfoptions.lowmemory==0
    %% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a
    % lot of memory.
    
    if vfoptions.verbose==1
        disp('Creating return fn matrix')
        tic;
        if vfoptions.returnmatrix==0
            fprintf('NOTE: When using CPU you can speed things up by giving return fn as a matrix; see vfoptions.returnmatrix=1 in VFI Toolkit documentation. \n')
        end
    end
    
    if isfield(vfoptions,'statedependentparams')
        if vfoptions.returnmatrix==2 % GPU
            if n_SDP==3
                ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec,SDP1,SDP2,SDP3);
            elseif n_SDP==2
                ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec,SDP1,SDP2);
            elseif n_SDP==1
                ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec,SDP1);
            end
        else
            fprintf('ERROR: statedependentparams only works with GPU (parallel=2) \n')
            dbstack
        end
    else % Following is the normal/standard behavior
        if vfoptions.returnmatrix==0
            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);
        elseif vfoptions.returnmatrix==1
            ReturnMatrix=ReturnFn;
        elseif vfoptions.returnmatrix==2 % GPU
            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
        end
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
            [VKron,Policy]=ValueFnIterRel_Case1_NoD_Par2_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance); %  a_grid, z_grid,
        end
    else
        if vfoptions.parallel==2 % On GPU
            [VKron, Policy]=ValueFnIterRel_Case1_Par2_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        end
    end
end


end