function [V,Policy2]=ValueFnIter_Case1_FHorz_TPath_SingleStep_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_grid, a_grid, z_grid_J, e_grid_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
pi_e_J=shiftdim(pi_e_J,-2); 

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
    % e_gridvals is created below
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
    % z_gridvals is created below
end


%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0
    
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid_J(:,N_j), e_grid_J(:,N_j), ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,:,N_j)=Vtemp;
    Policy(:,:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    
    for e_c=1:N_e
        e_val=vfoptions.e_gridvals_J(e_c,:,N_j);
        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid_J(:,N_j), e_val, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
        V(:,:,e_c,N_j)=Vtemp;
        Policy(:,:,e_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2
    for z_c=1:N_z
        z_val=vfoptions.z_gridvals_J(z_c,:,N_j);
        for e_c=1:N_e
            e_val=vfoptions.e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z,special_n_e, d_grid, a_grid, z_val,e_val, ReturnFnParamsVec);
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

    if vfoptions.verbose==1
        sprintf('Finite horizon: %i of %i',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    VKronNext_j=Vtemp_j; % Has been presaved before it was
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3); % Take expectations over e

    Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated

    
    if vfoptions.lowmemory==0
                
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid_J(:,jj), e_grid_J(:,jj), ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repmat(entireEV,1,N_a,1,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
        
        for e_c=1:N_e
            e_val=vfoptions.e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid_J(:,jj), e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)
            
            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV.*ones(1,N_a,1);
            
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            
            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=vfoptions.z_gridvals_J(z_c,:,jj);
            
            %Calc the condl expectation term (except beta) which depends on z but not control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            entireEV_z=kron(EV_z,ones(N_d,1));

            for e_c=1:N_e
                e_val=vfoptions.e_gridvals_J(e_c,:,jj);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                
                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*entireEV_z*ones(1,N_a,1);
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,jj)=Vtemp;
                Policy(:,z_c,e_c,jj)=maxindex;
            end
        end
        
    end
end

%%
Policy2=zeros(2,N_a,N_z,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end