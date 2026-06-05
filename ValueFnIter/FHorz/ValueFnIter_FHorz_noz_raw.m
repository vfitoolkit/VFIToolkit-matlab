function [V,Policy]=ValueFnIter_FHorz_noz_raw(n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

if isUnderlyingType(a_grid,'single')
    precision='single';
else
    precision='double';
end

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,precision,'gpuArray');
Policy=zeros(1,N_a,N_j,precision,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
if isUnderlyingType(a_grid,'single')
    ReturnFnParamsVec=single(ReturnFnParamsVec);
end

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(1,:,N_j)=maxindex;
else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);
    % (d,aprime,a)

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,N_j)=Vtemp;
    Policy(1,:,N_j)=maxindex;
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
    if isUnderlyingType(a_grid,'single')
        ReturnFnParamsVec=single(ReturnFnParamsVec);
        DiscountFactorParamsVec=single(DiscountFactorParamsVec);
    end

    EV=V(:,jj+1);

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);
    % (d,aprime,a)

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,jj)=Vtemp;
    Policy(1,:,jj)=maxindex;
end


end
