function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_QHS_nod_noz_e_raw(V,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_e=prod(n_e);

Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

Vnext=sum(V.*shiftdim(pi_e_J,-1),2); % Take expectations over e

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=K not supported for ValueFnIter_FHorz_TPath_SingleStep_QHS_nod_noz_e_raw')
end


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a,n_e, 0, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
    % Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1

    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec,0); % Because no z, can treat e like z and call Par2 rather than Par2e
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
        V(:,e_c,N_j)=Vtemp;
        Policy(:,e_c,N_j)=maxindex;
    end

end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    EV=Vnext(:,1,jj+1);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a,n_e, 0, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

        % First Policy
        entireRHS=ReturnMatrix+beta0beta*EV; % Use the today-to-tomorrow discount factor
        [~,maxindex]=max(entireRHS,[],1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
        % Now Vunderbar
        entireRHS=ReturnMatrix+beta*EV; % Use the two-future-periods discount factor
        maxindexfull=maxindex+N_a*(0:1:N_a-1)+shiftdim(N_a*N_a*(0:1:N_e-1),-1);
        V(:,:,jj)=entireRHS(maxindexfull);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec,0);

            % First Policy
            entireRHS_e=ReturnMatrix_e+beta0beta*EV; %.*ones(1,N_a,1);
            [~,maxindex]=max(entireRHS_e,[],1);
            Policy(:,e_c,jj)=shiftdim(maxindex,1);
            % Now Vunderbar
            entireRHS_e=ReturnMatrix_e+beta*EV; %.*ones(1,N_a,1);
            maxindexfull=maxindex+N_a*(0:1:N_a-1);
            V(:,e_c,jj)=entireRHS_e(maxindexfull);
        end

    end
end

%% Output shape for policy
Policy=shiftdim(Policy,-1);


end
