function [V, Policy]=ValueFnIter_Case1_FHorz_nod_noz_Par1_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j);
Policy=zeros(N_a,N_z,N_j); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
a_grid=gather(a_grid);
z_grid=gather(z_grid);

eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
    z_val=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

% Note: There is no z, so no need to deal with z_grid and pi_z depending on age

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1 || vfoptions.lowmemory==2

        parfor a_c=1:N_a
            a_val=a_gridvals(a_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, vfoptions.parallel, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_az);
            V(a_c,1,N_j)=Vtemp;
            Policy(a_c,1,N_j)=maxindex;
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);
        
        entireRHS_z=ReturnMatrix(:,:,1)+DiscountFactorParamsVec*V_Jplus1; %*ones(1,N_a,1);
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,1,N_j)=Vtemp;
        Policy(:,1,N_j)=maxindex;
        
    elseif vfoptions.lowmemory==1 || vfoptions.lowmemory==2
        DiscountedVKronNext_j=DiscountFactorParamsVec*V_Jplus1;
        parfor a_c=1:N_a
            a_val=a_gridvals(a_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, vfoptions.parallel,ReturnFnParamsVec);
            
            entireRHS_az=ReturnMatrix_az+DiscountedVKronNext_j;
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_az);
            V(a_c,1,N_j)=Vtemp;
            Policy(a_c,1,N_j)=maxindex;
        end
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
    
    % Note: There is no z, so no need to deal with z_grid and pi_z depending on age
    
    VKronNext_j=V(:,:,jj+1);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);
        
        entireRHS_z=ReturnMatrix(:,:,1)+DiscountFactorParamsVec*VKronNext_j; %*ones(1,N_a,1);
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,1,jj)=Vtemp;
        Policy(:,1,jj)=maxindex;
        
    elseif vfoptions.lowmemory==1 || vfoptions.lowmemory==2
        DiscountedVKronNext_j=DiscountFactorParamsVec*VKronNext_j;
        parfor a_c=1:N_a
            a_val=a_gridvals(a_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, vfoptions.parallel,ReturnFnParamsVec);
            
            entireRHS_az=ReturnMatrix_az+DiscountedVKronNext_j;
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_az);
            V(a_c,1,jj)=Vtemp;
            Policy(a_c,1,jj)=maxindex;
        end
    end
        
end


end