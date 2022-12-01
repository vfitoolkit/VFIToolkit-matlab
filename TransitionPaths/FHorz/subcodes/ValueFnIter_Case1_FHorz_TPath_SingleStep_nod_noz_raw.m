function [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
if vfoptions.lowmemory>0
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end

%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    %if vfoptions.returnmatrix==2 % GPU
    for a_c=1:N_a
        a_val=a_gridvals(a_c,:);
        ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, special_n_a, 0, a_val, ReturnFnParamsVec,0);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_a);
        V(a_c,N_j)=Vtemp;
        Policy(a_c,N_j)=maxindex;
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        sprintf('Finite horizon: %i of %i (counting backwards to 1)',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    VKronNext_j=Vtemp_j; % Has been presaved before it was
%     VKronNext_j=V(:,:,j+1);
    Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
            
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*VKronNext_j*ones(1,N_a,1);
            
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,jj)=Vtemp;
        Policy(:,jj)=maxindex;
        
    elseif vfoptions.lowmemory==1
        for a_c=1:N_a
            a_val=a_gridvals(a_c,:);
            ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, special_n_a, 0, a_val, ReturnFnParamsVec,0);
            
            entireRHS_a=ReturnMatrix_a+DiscountFactorParamsVec*EV_z;
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_a);
            V(a_c,jj)=Vtemp;
            Policy(a_c,jj)=maxindex;
        end
        
    end
end




end