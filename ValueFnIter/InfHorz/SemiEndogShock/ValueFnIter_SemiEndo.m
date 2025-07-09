function [V,Policy]=ValueFnIter_SemiEndo(V0, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, DiscountFactorParamsVec, ReturnFn, vfoptions)

if vfoptions.lowmemory~=0 || vfoptions.parallel<1 % GPU or parellel CPU are only things that I have created (email me if you want/need other options)
    error('Only lowmemory=0 and parallel=1 or 2 are currently possible when using vfoptions.SemiEndogShockFn \n')
end

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, ReturnFnParamsVec);

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

if vfoptions.parallel==2
    if n_d(1)==0
        [VKron,Policy]=ValueFnIter_Case1_NoD_SemiEndog_Par2_raw(V0, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
    else
        [VKron, Policy]=ValueFnIter_Case1_SemiEndog_Par2_raw(V0, n_d, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
    end
elseif vfoptions.parallel==1
    if n_d(1)==0
        [VKron,Policy]=ValueFnIter_Case1_NoD_SemiEndog_Par1_raw(V0, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
    else
        [VKron, Policy]=ValueFnIter_Case1_SemiEndog_Par1_raw(V0, n_d, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
    end
end
if vfoptions.outputkron==0
    V=reshape(VKron,[n_a,n_z]);
    Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
end

if vfoptions.polindorval==2
    Policy=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid, a_grid,vfoptions.parallel);
end

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    Policy=uint64(Policy);
    Policy=double(Policy);
end


end