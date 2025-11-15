function [V, Policy]=ValueFnIter_FHorz_Par1_nod_raw(n_a,n_z,N_j, a_grid, z_grid,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j);
Policy=zeros(N_a,N_z,N_j); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
    % Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]);% Using V_Jplus1
    
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);

    parfor z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);

        % Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=EV.*(ones(N_a,1)*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);

        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z; %*ones(1,N_a,1);

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c,N_j)=Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=V(:,:,jj+1);
    
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);

    parfor z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);

        %Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=EV.*(ones(N_a,1)*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);

        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z; %*ones(1,N_a,1);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c,jj)=Vtemp;
        Policy(:,z_c,jj)=maxindex;
    end
end


end
