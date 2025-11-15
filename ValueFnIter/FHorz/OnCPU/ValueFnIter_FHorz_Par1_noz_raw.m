function [V,Policy2]=ValueFnIter_FHorz_Par1_noz_raw(n_d,n_a,N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j);
Policy=zeros(N_a,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz(ReturnFn, n_d, n_a, d_grid, a_grid, vfoptions.parallel,ReturnFnParamsVec,0);
    % Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

else
    
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]); % Using V_Jplus1

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz(ReturnFn, n_d, n_a, d_grid, a_grid, vfoptions.parallel, ReturnFnParamsVec,0);

    entireEV=repelem(EV,N_d,1);
    entireRHS=ReturnMatrix+DiscountFactorParamsVec*entireEV; % autoexpand a into the 2nd-dim of entireEV

    % Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;
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
    
    EV=V(:,jj+1);

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz(ReturnFn, n_d, n_a, d_grid, a_grid, vfoptions.parallel, ReturnFnParamsVec,0);

    entireEV=repelem(EV,N_d,1);
    entireRHS=ReturnMatrix+DiscountFactorParamsVec*entireEV; % autoexpand a into the 2nd-dim of entireEV

    % Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,jj)=Vtemp;
    Policy(:,jj)=maxindex;
end

%%
Policy2=zeros(2,N_a,N_j); %NOTE: this is not actually in Kron form
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
