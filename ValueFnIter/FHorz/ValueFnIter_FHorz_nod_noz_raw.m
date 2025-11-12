function [V, Policy]=ValueFnIter_FHorz_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;
else
    
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]); % Using V_Jplus1

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % autoexpand a in 3rd-dim

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,N_j)=shiftdim(Vtemp,1);
    Policy(:,N_j)=shiftdim(maxindex,1);
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
    
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % autoexpand a in 3rd-dim

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,jj)=shiftdim(Vtemp,1);
    Policy(:,jj)=shiftdim(maxindex,1);
end


end
