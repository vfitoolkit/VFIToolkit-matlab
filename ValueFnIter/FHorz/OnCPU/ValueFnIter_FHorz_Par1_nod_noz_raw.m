function [V, Policy]=ValueFnIter_FHorz_Par1_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);

V=zeros(N_a,N_j);
Policy=zeros(N_a,N_j); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
a_grid=gather(a_grid);

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

% Note: There is no z, so no need to deal with z_grid and pi_z depending on age

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, n_a, 0, [], a_grid, [], vfoptions.parallel, ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;
else
    % Using V_Jplus1
    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, n_a, 0, [], a_grid, [], vfoptions.parallel, ReturnFnParamsVec);

    entireRHS_z=ReturnMatrix+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS_z,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

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

    EV=V(:,jj+1);

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, n_a, 0, [], a_grid, [], vfoptions.parallel, ReturnFnParamsVec);

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,jj)=Vtemp;
    Policy(:,jj)=maxindex;

end


end
