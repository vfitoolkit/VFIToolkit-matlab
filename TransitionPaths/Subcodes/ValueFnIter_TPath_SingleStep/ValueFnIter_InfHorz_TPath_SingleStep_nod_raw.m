function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_nod_raw(Vnext,n_a,n_z, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,'gpuArray');
Policy=zeros(N_a,N_z,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

if vfoptions.lowmemory==0
    
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, [], a_grid, z_gridvals, ReturnFnParamsVec);

    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=Vnext.*shiftdim(pi_z',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2);

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);
    V=reshape(V,[N_a,N_z]); % otherwise it was outputing as [1,N_a,N_z]
    Policy=reshape(Policy,[N_a,N_z]); % otherwise it was outputing as [1,N_a,N_z]
    
elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
        
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=Vnext.*pi_z(z_c,:);
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c)=Vtemp;
        Policy(:,z_c)=maxindex;
    end
end

%% Policy in transition paths
Policy=UnKronPolicyIndexes_Case1(Policy,0,n_a,n_z,vfoptions);

end
