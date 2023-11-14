function [V,Policy]=ValueFnIter_Case1_FHorz_RiskyAsset_noz_e_raw(n_d,n_a1,n_a2,n_e,n_u, N_j, d_grid, a1_grid, a2_grid, e_gridvals_J, u_grid, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d=prod(n_d);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_e=prod(n_e);
N_u=prod(n_u);

N_a=N_a1*N_a2;

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);
u_grid=gpuArray(u_grid);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], n_e, [d_grid; a1_grid], [a1_grid; a2_grid],e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,N_j)=Vtemp;
        Policy(:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(a_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], special_n_e, [d_grid; a1_grid], [a1_grid; a2_grid], e_val, ReturnFnParamsVec,0);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=sum(reshape(vfoptions.V_Jplus1,[N_a,N_e]).*pi_e_J(:,N_j)',2);    % First, switch V_Jplus1 into Kron form ,take expecation over e

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, [n_d,n_a1], n_a2, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

    aprimeIndex=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),(a2primeIndex-1)); % [N_d*N_a1,N_u]
    aprimeplus1Index=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),a2primeIndex); % [N_d*N_a1,N_u]
    aprimeProbs=kron(ones(N_a1,1),a2primeProbs);  % [N_d*N_a1,N_u]
    % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)

    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EV1=aprimeProbs.*reshape(V_Jplus1(aprimeIndex),[N_d*N_a1,N_u]); % (d&a1prime,u), the lower aprime
    EV2=(1-aprimeProbs).*reshape(V_Jplus1(aprimeplus1Index),[N_d*N_a1,N_u]); % (d&a1prime,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid
    
    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d&a1prime,u), sum over u
    % EV is over (d&a1prime,1)
    
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2],n_e, [d_grid; a1_grid], [a1_grid; a2_grid],e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,a,e)
  
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV.*ones(1,N_a,N_e); % d-by-a-by-e
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1

        betaEV=DiscountFactorParamsVec*EV.*ones(1,N_a,1);

       for e_c=1:N_e
           e_val=e_gridvals_J(e_c,:,N_j);
           ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], n_e, special_n_e, [d_grid; a1_grid], [a1_grid; a2_grid], e_val, ReturnFnParamsVec);
           
           entireRHS_e=ReturnMatrix_e+betaEV; % (d&a1prime,a)
           
           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_e,[],1);
           V(:,e_c,N_j)=Vtemp;
           Policy(:,e_c,N_j)=maxindex;
        end
        
    end
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
    
    VKronNext_j=sum(V(:,:,jj+1).*pi_e_J(:,jj)',2); % Expectation over e
    
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, [n_d,n_a1], n_a2, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

    aprimeIndex=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),(a2primeIndex-1)); % [N_d*N_a1,N_u]
    aprimeplus1Index=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),a2primeIndex); % [N_d*N_a1,N_u]
    aprimeProbs=kron(ones(N_a1,1),a2primeProbs);  % [N_d*N_a1,N_u]
    % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)

    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EV1=aprimeProbs.*reshape(VKronNext_j(aprimeIndex),[N_d*N_a1,N_u]); % (d&a1prime,u), the lower aprime
    EV2=(1-aprimeProbs).*reshape(VKronNext_j(aprimeplus1Index),[N_d*N_a1,N_u]); % (d&a1prime,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid
    
    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d&a1prime,u), sum over u
    % EV is over (d&a1prime,1)
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], n_e, [d_grid; a1_grid], [a1_grid; a2_grid], e_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,a,e)
        
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV.*ones(1,N_a,N_e); % d-by-a-by-e
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
       
       betaEV=DiscountFactorParamsVec*EV.*ones(1,N_a,1);
        
       for e_c=1:N_e
           e_val=e_gridvals_J(e_c,:,jj);
           ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], special_n_e, [d_grid; a1_grid], [a1_grid; a2_grid], e_val, ReturnFnParamsVec);
           
           entireRHS_e=ReturnMatrix_e+betaEV; % d-by-a
           
           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_e,[],1);
           V(:,e_c,jj)=Vtemp;
           Policy(:,e_c,jj)=maxindex;
        end
        
    end
end


end