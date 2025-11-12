function  [V,Policy]=ValueFnIter_FHorz_nod_noz_e_raw(n_a, n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Note: have no z variable, do have e variables

N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); % no d variable

if ~isfield(vfoptions,'parallel_e')
    vfoptions.parallel_e=zeros(1,length(n_e));
    % Parallel e can have some elements (starting from the front end) equal to 1. I will parallelize over these.
end


%%
if all(vfoptions.parallel_e==0)
    %% Loop over all e
    special_n_e=ones(1,length(n_e));

    %% N_j
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

    pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)
    
    if ~isfield(vfoptions,'V_Jplus1')
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec); % Because no z, can treat e like z and call Par2 rather than Par2e
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        EV=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form
        EV=sum(EV.*pi_e_J(:,:,N_j),2);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec);

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*EV; %.*ones(1,N_a,1);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,e_c,N_j)=shiftdim(maxindex,1);
        end
    end

    %% Loop backward over age
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;

        if vfoptions.verbose==1
            fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
        end

        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        EV=V(:,:,jj+1);
        EV=sum(EV.*pi_e_J(:,:,jj),2);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec);

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*EV; %.*ones(1,N_a,1);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,e_c,jj)=shiftdim(maxindex,1);
        end
    end

elseif all(vfoptions.parallel_e==1)
    % Parallel over all e

    %% N_j
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

    pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimensionfor e_c=1:n_e (normally -2, but no z so -1)
    
    if ~isfield(vfoptions,'V_Jplus1')
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % Because no z, can treat e like z and call Par2 rather than Par2e
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        EV=reshape(vfoptions.V_Jplus1,[N_a,N_e]); % Using V_Jplus1
        EV=sum(EV.*pi_e_J(1,:,N_j),2);

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %.*ones(1,N_a,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
    end


    %% Loop backward over age
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;

        if vfoptions.verbose==1
            fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
        end

        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        EV=V(:,:,jj+1);
        EV=sum(EV.*pi_e_J(1,:,jj),2);

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %.*ones(1,N_a,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
    end
    
else
    % Split e into e1 which I parallel over, and e2 which I loop over
    % Assumes that e1 and e2 are uncorrelated/independently distributed
    n_e1=n_e(logical(vfoptions.parallel_e));
    n_e2=n_e(~logical(vfoptions.parallel_e));
    N_e1=prod(n_e1);
    N_e2=prod(n_e2);

    e1_gridvals_J=e_gridvals_J(1:1:N_e1,1:sum(n_e1),:); % Note, allows for dependence on age j
    e2_gridvals_J=e_gridvals_J(N_e1*(0:1:N_e2-1),sum(n_e1)+1:end,:); % Note, allows for dependence on age j
    temp=reshape(pi_e_J,[N_e1,N_e2,N_j]);
    pi_e1_J=reshape(sum(temp,2),[N_e1,N_j]); % Assumes that e1 and e2 are uncorrelated/independently distributed
    pi_e2_J=reshape(sum(temp,1),[N_e2,N_j]); % Assumes that e1 and e2 are uncorrelated/independently distributed

    % Need to be a different size when do a mix of parallel and loop for e
    V=zeros(N_a,N_e1,N_e2,N_j,'gpuArray');
    Policy=zeros(N_a,N_e1,N_e2,N_j,'gpuArray');
    
    pi_e1_J=shiftdim(pi_e1_J,-1); % Move to second dimension
    pi_e2_J=shiftdim(pi_e2_J,-2); % Move to thrid dimension

    %% j=N_j

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);


    % Now loop over e2, and within that parallelize over e1
    special_n_e2=ones(1,length(n_e2));

    if ~isfield(vfoptions,'V_Jplus1')
        for e2_c=1:N_e2
            e2_val=e2_gridvals_J(e2_c,:,N_j);
            ReturnMatrix_e2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, n_e1, special_n_e2, 0, a_grid, e1_gridvals_J(:,:,N_j), e2_val, ReturnFnParamsVec); % Just treat e1 like z and e2 like e
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e2,[],1);
            V(:,:,e2_c,N_j)=Vtemp;
            Policy(:,:,e2_c,N_j)=maxindex;
        end
    else

        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        EV=reshape(vfoptions.V_Jplus1,[N_a,N_e]); % Using V_Jplus1
        EV=sum(EV.*pi_e2_J,3);

        for e2_c=1:N_e2
            e2_val=e2_gridvals_J(e2_c,:,N_j);
            ReturnMatrix_e2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, n_e1, special_n_e2, 0, a_grid, e1_gridvals_J(:,:,N_j), e2_val, ReturnFnParamsVec); % Just treat e1 as z and e2 as e

            EV_e2=EV.*pi_e1_J(1,:,N_j);
            EV_e2(isnan(EV_e2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_e2=sum(EV_e2,2); % sum over z', leaving a singular second dimension

            entireRHS_e2=ReturnMatrix_e2+DiscountFactorParamsVec*EV_e2; %.*ones(1,N_a,1);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e2,[],1);

            V(:,:,e2_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e2_c,N_j)=shiftdim(maxindex,1);
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

        EV=V(:,:,:,jj+1);
        EV=sum(EV.*pi_e2_J(1,1,:,jj),3);

        for e2_c=1:N_e2
            e2_val=e2_gridvals_J(e2_c,:,jj);
            ReturnMatrix_e2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, 0, n_a, n_e1, special_n_e2, 0, a_grid, e1_gridvals_J(:,:,jj), e2_val, ReturnFnParamsVec); % Just treat e1 as z and e2 as e

            EV_e2=EV.*pi_e1_J(1,:,jj);
            EV_e2(isnan(EV_e2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_e2=sum(EV_e2,2); % sum over z', leaving a singular second dimension

            entireRHS_e2=ReturnMatrix_e2+DiscountFactorParamsVec*EV_e2; %.*ones(1,N_a,1);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e2,[],1);

            V(:,:,e2_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e2_c,jj)=shiftdim(maxindex,1);
        end
        
    end

    V=reshape(V,[N_a,N_e,N_j]);
    Policy=reshape(Policy,[N_a,N_e,N_j]);

end
