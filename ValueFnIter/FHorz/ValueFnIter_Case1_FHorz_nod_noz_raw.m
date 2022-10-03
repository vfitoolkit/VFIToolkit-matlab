function [V, Policy]=ValueFnIter_Case1_FHorz_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
a_grid=gpuArray(a_grid);

if vfoptions.lowmemory==1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0
    
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    
    for a_c=1:N_a
        a_val=a_gridvals(a_c,:);
        ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, special_n_a, 0, a_val, ReturnFnParamsVec,0);
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_a);
        V(a_c,N_j)=Vtemp;
        Policy(a_c,N_j)=maxindex;

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

    
    VKronNext_j=V(:,jj+1);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
        
        % (aprime,a)
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*VKronNext_j.*ones(1,N_a);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,jj)=shiftdim(Vtemp,1);
        Policy(:,jj)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
        for a_c=1:N_a
            a_val=a_gridvals(a_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, special_n_a, 0, a_val, ReturnFnParamsVec,0);

            entireRHS_a=ReturnMatrix_az+DiscountFactorParamsVec*VKronNext_j;
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_a);
            V(a_c,jj)=Vtemp;
            Policy(a_c,jj)=maxindex;
        end

    end
end


end