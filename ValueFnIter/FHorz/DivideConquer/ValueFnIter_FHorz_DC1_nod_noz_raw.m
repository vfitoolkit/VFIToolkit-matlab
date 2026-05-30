function [V,Policy]=ValueFnIter_FHorz_DC1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% divide-and-conquer for length(n_a)==1

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(1,N_a,N_j,'gpuArray'); % indexes the optimal choice for aprime rest of dimensions a,z

%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

    %Calc the max and it's index
    [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);

    V(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy(1,level1ii,N_j)=maxindex1;

    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(maxindex1(ii):maxindex1(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        V(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(1,curraindex,N_j)=maxindex+maxindex1(ii)-1;
    end

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]); % Using V_Jplus1

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;
    %Calc the max and it's index
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

    V(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy(1,level1ii,N_j)=maxindex1;

    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(maxindex1(ii):maxindex1(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(maxindex1(ii):maxindex1(ii+1));
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(1,curraindex,N_j)=maxindex+maxindex1(ii)-1;
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

    EV=V(:,jj+1);

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;
    %Calc the max and it's index
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

    V(level1ii,jj)=shiftdim(Vtempii,1);
    Policy(1,level1ii,jj)=maxindex1;

    % Note: Did a runtime test, this simple version is faster than actually checking if maxgap(ii)=0 like in all the other DC1 codes.
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(maxindex1(ii):maxindex1(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(maxindex1(ii):maxindex1(ii+1));
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(curraindex,jj)=shiftdim(Vtempii,1);
        Policy(1,curraindex,jj)=maxindex+maxindex1(ii)-1;
    end

end




end
