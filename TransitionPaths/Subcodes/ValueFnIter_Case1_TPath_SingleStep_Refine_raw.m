function [V,Policy]=ValueFnIter_Case1_TPath_SingleStep_Refine_raw(Vnext,n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
%% Refinement: calculate ReturnMatrix and 'remove' the d dimension

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,'gpuArray');
Policy_a=zeros(N_a,N_z,'gpuArray');

%%
if vfoptions.lowmemory>0
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end
if vfoptions.lowmemory>1
    a_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%%
l_z=length(n_z);

%%
if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
    [ReturnMatrix,dstar]=max(ReturnMatrix,[],1); % solve for dstar
    ReturnMatrix=shiftdim(ReturnMatrix,1);
    dstar=shiftdim(dstar,1);
    
if vfoptions.lowmemory==1
    %% Refinement: calculate ReturnMatrix and 'remove' the d dimension
    ReturnMatrix=zeros(N_a,N_a,N_z); % 'refined' return matrix
    dstar=zeros(N_a,N_a,N_z);
    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn,n_d, n_a, ones(l_z,1),d_grid, a_grid, zvals,ReturnFnParams,1); % the 1 at the end if to outpuit for refine
        [ReturnMatrix_z,dstar_z]=max(ReturnMatrix_z,[],1); % solve for dstar
        ReturnMatrix(:,:,z_c)=shiftdim(ReturnMatrix_z,1);
        dstar(:,:,z_c)=shiftdim(dstar_z,1);
    end
    
elseif vfoptions.lowmemory==2
    ReturnMatrix=zeros(N_a,N_a,N_z); % 'refined' return matrix
    dstar=zeros(N_a,N_a,N_z);
    for a_c=1:N_a
        avals=a_gridvals(a_c,:);
        ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_Par2_LowMem2(ReturnFn, n_d, n_a, ones(l_a,1),n_z, d_grid, a_grid, avals, z_grid,ReturnFnParams,1); % note: first n_a is n_aprime % the 1 at the end if to outpuit for refine
        [ReturnMatrix_a,dstar_a]=max(ReturnMatrix_a,[],1); % solve for dstar
        ReturnMatrix(:,a_c,:)=shiftdim(ReturnMatrix_a,1);
        dstar(:,a_c,:)=shiftdim(dstar_a,1);
    end
end


%%
if vfoptions.lowmemory==0
    
    for z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=Vnext.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c)=Vtemp;
        Policy_a(:,z_c)=maxindex;
    end
    
elseif vfoptions.lowmemory==1 || vfoptions.lowmemory==2
    for z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);

        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=Vnext.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c)=Vtemp;
        Policy_a(:,z_c)=maxindex;
    end
end

%%
Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(2,:,:)=shiftdim(Policy_a,-1); % aprime
temppolicyindex=reshape(Policy_a,[1,N_a*N_z])+(0:1:N_a*N_z-1)*N_a;
Policy(1,:,:)=reshape(dstar(temppolicyindex),[N_a,N_z]); % d

end