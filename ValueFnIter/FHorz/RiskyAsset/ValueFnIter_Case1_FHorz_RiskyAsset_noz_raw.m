function [V,Policy]=ValueFnIter_Case1_FHorz_RiskyAsset_noz_raw(n_d,n_a1,n_a2,n_u, N_j, d_grid, a1_grid,a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d=prod(n_d);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_u=prod(n_u);

N_a=N_a1*N_a2;

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);
u_grid=gpuArray(u_grid);

if vfoptions.lowmemory>0
    special_n_a1=ones(1,length(n_a1));
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1); % The 1 at end indicates want output in form of matrix.
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], [d_grid; a1_grid], [a1_grid; a2_grid], ReturnFnParamsVec);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,N_j)=Vtemp;
        Policy(:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1
        %if vfoptions.returnmatrix==2 % GPU
        for a1_c=1:N_a1
            a1_val=a1_gridvals(a1_c,:);
            ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn, [n_d,n_a1], [special_n_a1,n_a2], [d_grid; a1_grid], [a1_val; a2_grid], ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_a,[],1);
            V(a1_c+N_a1*(0:1:N_a2-1),N_j)=Vtemp;
            Policy(a1_c+N_a1*(0:1:N_a2-1),N_j)=maxindex;
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a2,1]);    % First, switch V_Jplus1 into Kron form

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
    EV1=aprimeProbs.*reshape(V_Jplus1(aprimeIndex),[N_d*N_a1,N_u]); % (d,u), the lower aprime
    EV2=(1-aprimeProbs).*reshape(V_Jplus1(aprimeplus1Index),[N_d*N_a1,N_u]); % (d,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid
    
    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,z), sum over u
    % EV is over (d,1)
    
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], [d_grid; a1_grid], [a1_grid; a2_grid], ReturnFnParamsVec);
        % (d,aprime,a)
  
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV.*ones(1,N_a,1); % d-by-a
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,N_j)=Vtemp;
        Policy(:,N_j)=maxindex;
        
    elseif vfoptions.lowmemory==1
       for a1_c=1:N_a1
           a1_val=a1_gridvals(a1_c,:);
           ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn, [n_d,n_a1], [special_n_a1,n_a2],  [d_grid; a1_grid],  [a1_val; a2_grid], ReturnFnParamsVec);
           
           entireRHS_a=ReturnMatrix_a+DiscountFactorParamsVec*EV; % d-by-1
           
           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_a,[],1);
           V(a1_c+N_a1*(0:1:N_a2-1),N_j)=Vtemp;
           Policy(a1_c+N_a1*(0:1:N_a2-1),N_j)=maxindex;
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
    
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, [n_d,n_a1], n_a2, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

    aprimeIndex=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),(a2primeIndex-1)); % [N_d*N_a1,N_u]
    aprimeplus1Index=kron((1:1:N_a1)',ones(N_d,N_u))+N_a1*kron(ones(N_a1,1),a2primeIndex); % [N_d*N_a1,N_u]
    aprimeProbs=kron(ones(N_a1,1),a2primeProbs);  % [N_d*N_a1,N_u]
    % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)

    VKronNext_j=V(:,jj+1);
    
    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EV1=aprimeProbs.*reshape(VKronNext_j(aprimeIndex),[N_d*N_a1,N_u]); % (d,u), the lower aprime
    EV2=(1-aprimeProbs).*reshape(VKronNext_j(aprimeplus1Index),[N_d*N_a1,N_u]); % (d,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid
    
    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,z), sum over u
    % EV is over (d,1)
    
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn, [n_d,n_a1], [n_a1,n_a2], [d_grid; a1_grid], [a1_grid; a2_grid], ReturnFnParamsVec);
        % (d,aprime,a)
        
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV.*ones(1,N_a,1); % aprime-by-a
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,jj)=Vtemp;
        Policy(:,jj)=maxindex;
        
    elseif vfoptions.lowmemory==1
       
       for a1_c=1:N_a1
           a1_val=a1_gridvals(a1_c,:);
           ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn, [n_d,n_a1], [special_n_a1,n_a2],  [d_grid; a1_grid],  [a1_val; a2_grid], ReturnFnParamsVec);
           
           entireRHS_a=ReturnMatrix_a+DiscountFactorParamsVec*EV; % aprime-by-1
           
           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_a,[],1);
           V(a1_c+N_a1*(0:1:N_a2-1),jj)=Vtemp;
           Policy(a1_c+N_a1*(0:1:N_a2-1),jj)=maxindex;
        end
        
    end
end


end