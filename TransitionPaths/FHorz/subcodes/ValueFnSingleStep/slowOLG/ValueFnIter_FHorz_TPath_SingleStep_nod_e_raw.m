function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_nod_e_raw(V,n_a,n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

Vnext=sum(V.*shiftdim(pi_e_J,-2),3); % Take expectations over e

%%
if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
end


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0
    
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, n_z,n_e, 0, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,:,N_j)=Vtemp;
    Policy(:,:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    
    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, n_z, special_n_e, 0, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
        V(:,:,e_c,N_j)=Vtemp;
        Policy(:,:,e_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2
    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, special_n_z, special_n_e, 0, a_grid, z_val, e_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
            V(:,z_c,e_c,N_j)=Vtemp;
            Policy(:,z_c,e_c,N_j)=maxindex;
        end
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    VKronNext_j=Vnext(:,:,1,jj+1);
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, n_z,n_e, 0, a_grid, z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj), ReturnFnParamsVec);

        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % autofill .*ones(1,N_a,1,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, n_z, special_n_e, 0, a_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*EV.*ones(1,N_a,1);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            
            %Calc the condl expectation term (except beta) which depends on z but not control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, special_n_z, special_n_e, 0, a_grid, z_val, e_val, ReturnFnParamsVec);

                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,jj)=Vtemp;
                Policy(:,z_c,e_c,jj)=maxindex;
            end
        end
        
    end
end




end
