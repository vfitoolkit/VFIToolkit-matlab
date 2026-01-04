function [V,Policy2]=ValueFnIter_FHorz_Par1_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j);
Policy=zeros(N_a,N_z,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
    % Calc the max and its index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]); % Using V_Jplus1

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);

    parfor z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);

        %Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=EV.*pi_z(z_c,:);
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);

        entireEV_z=repelem(EV_z,N_d,1);
        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*entireEV_z; % autoexpand a into 2nd-dim of entireEV_z

        % Calc the max and its index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c,N_j)=Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end   
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=V(:,:,jj+1);
    
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
         
    parfor z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);

        %Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=EV.*pi_z(z_c,:);
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);

        entireEV_z=repelem(EV_z,N_d,1);
        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*entireEV_z; % autoexpand a into 2nd-dim of entireEV_z

        % Calc the max and its index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c,jj)=Vtemp;
        Policy(:,z_c,jj)=maxindex;
    end
end

%%
Policy2=zeros(2,N_a,N_z,N_j); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
