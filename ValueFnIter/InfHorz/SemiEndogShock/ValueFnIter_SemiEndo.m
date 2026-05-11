function [V,Policy]=ValueFnIter_SemiEndo(V0, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, DiscountFactorParamsVec, ReturnFn, vfoptions)

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, ReturnFnParamsVec);

if isa(vfoptions.SemiEndogShockFn,'function_handle')==0
    pi_z_semiendog=vfoptions.SemiEndogShockFn;
else
    if ~isfield(vfoptions,'SemiEndogShockFnParamNames')
        error('vfoptions.SemiEndogShockFnParamNames is missing (is needed for vfoptions.SemiEndogShockFn) \n')
    end
    pi_z_semiendog=zeros(N_a,N_z,N_z);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    SemiEndogParamsVec=CreateVectorFromParams(Parameters, vfoptions.SemiEndogShockFnParamNames);
    SemiEndogParamsCell=cell(length(SemiEndogParamsVec),1);
    for ii=1:length(SemiEndogParamsVec)
        SemiEndogParamsCell(ii,1)={SemiEndogParamsVec(ii)};
    end
    parfor ii=1:N_a
        a_ii=a_gridvals(ii,:)';
        a_ii_SemiEndogParamsCell=[a_ii;SemiEndogParamsCell];
        [~,temp_pi_z]=SemiEndogShockFn(a_ii_SemiEndogParamsCell{:});
        pi_z_semiendog(ii,:,:)=temp_pi_z;
        % Note that temp_z_grid is just the same things for all k, and same as
        % z_grid created about 10 lines above, so I don't bother keeping it.
        % I only create it so you can double-check it is same as z_grid
    end
end

if n_d(1)==0
    [VKron,Policy]=ValueFnIter_InfHorz_SemiEndog_nod_raw(V0, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
else
    [VKron, Policy]=ValueFnIter_InfHorz_SemiEndog_raw(V0, n_d, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
end

if vfoptions.outputkron==0
    V=reshape(VKron,[n_a,n_z]);
    Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
end


end
