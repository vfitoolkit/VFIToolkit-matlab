function [V,Policy]=ValueFnIter_Case1_FHorz_DC1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
% divide-and-conquer for length(n_a)==1

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
a_grid=gpuArray(a_grid);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
Policytemp=zeros(N_a,1,'gpuArray');

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, n_a, a_grid(level1ii), a_grid, ReturnFnParamsVec);

    %Calc the max and it's index
    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);

    V(level1ii,N_j)=shiftdim(Vtempii,1);
    Policytemp(level1ii)=shiftdim(maxindex,1);

    for ii=1:(vfoptions.level1n-1)
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, n_a, a_grid(level1ii(ii)+1:level1ii(ii+1)-1), a_grid(Policytemp(level1ii(ii)):Policytemp(level1ii(ii+1))), ReturnFnParamsVec);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,N_j)=shiftdim(Vtempii,1);
        Policytemp(level1ii(ii)+1:level1ii(ii+1)-1)=shiftdim(maxindex,1)+Policytemp(level1ii(ii))-1;
    end
  
    Policy(:,N_j)=Policytemp;

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, n_a, a_grid(level1ii), a_grid, ReturnFnParamsVec);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*V_Jplus1;
    %Calc the max and it's index
    [Vtempii,maxindex]=max(entireRHS_ii,[],1);

    V(level1ii,N_j)=shiftdim(Vtempii,1);
    Policytemp(level1ii)=shiftdim(maxindex,1);

    for ii=1:(vfoptions.level1n-1)
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, n_a, a_grid(level1ii(ii)+1:level1ii(ii+1)-1), a_grid(Policytemp(level1ii(ii)):Policytemp(level1ii(ii+1))), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*V_Jplus1(Policytemp(level1ii(ii)):Policytemp(level1ii(ii+1)));
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,N_j)=shiftdim(Vtempii,1);
        Policytemp(level1ii(ii)+1:level1ii(ii+1)-1)=shiftdim(maxindex,1)+Policytemp(level1ii(ii))-1;
    end

    Policy(:,N_j)=Policytemp;
        
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

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, n_a, a_grid(level1ii), a_grid, ReturnFnParamsVec);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*VKronNext_j;
    %Calc the max and it's index
    [Vtempii,maxindex]=max(entireRHS_ii,[],1);

    V(level1ii,jj)=shiftdim(Vtempii,1);
    Policytemp(level1ii)=shiftdim(maxindex,1);
    
    for ii=1:(vfoptions.level1n-1)
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, n_a, a_grid(level1ii(ii)+1:level1ii(ii+1)-1), a_grid(Policytemp(level1ii(ii)):Policytemp(level1ii(ii+1))), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*VKronNext_j(Policytemp(level1ii(ii)):Policytemp(level1ii(ii+1)));
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,jj)=shiftdim(Vtempii,1);
        Policytemp(level1ii(ii)+1:level1ii(ii+1)-1)=shiftdim(maxindex,1)+Policytemp(level1ii(ii))-1;
    end

    Policy(:,jj)=Policytemp;

end










end