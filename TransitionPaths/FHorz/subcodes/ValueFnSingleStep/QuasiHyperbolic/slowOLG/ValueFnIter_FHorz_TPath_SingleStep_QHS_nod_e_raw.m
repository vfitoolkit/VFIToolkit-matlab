function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_nod_e_raw(V,n_a,n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z
Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray'); % agent's-perspective value at QH-optimal policy under beta0beta

Vnext=sum(V.*shiftdim(pi_e_J,-2),3); % Take expectations over e

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
end
if vfoptions.lowmemory>=3
    error('vfoptions.lowmemory=K not supported for ValueFnIter_FHorz_TPath_SingleStep_QHS_nod_e_raw')
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, n_z,n_e, 0, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,:,N_j)=Vtemp;
    Policy(:,:,:,N_j)=maxindex;
    Vhat(:,:,:,N_j)=V(:,:,:,N_j); % terminal: no continuation, Vhat=V

elseif vfoptions.lowmemory==1

    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, n_z, special_n_e, 0, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
        V(:,:,e_c,N_j)=Vtemp;
        Policy(:,:,e_c,N_j)=maxindex;
        Vhat(:,:,e_c,N_j)=V(:,:,e_c,N_j); % terminal: no continuation, Vhat=V
    end

elseif vfoptions.lowmemory==2

    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, special_n_z, special_n_e, 0, a_grid, z_val, e_val, ReturnFnParamsVec,0);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
            V(:,z_c,e_c,N_j)=Vtemp;
            Policy(:,z_c,e_c,N_j)=maxindex;
            Vhat(:,z_c,e_c,N_j)=V(:,z_c,e_c,N_j); % terminal: no continuation, Vhat=V
        end
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

    VKronNext_j=Vnext(:,:,1,jj+1);

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, n_z,n_e, 0, a_grid, z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

        % First Policy
        entireRHS=ReturnMatrix+beta0beta*EV; % Use the today-to-tomorrow discount factor
        [Vtilde_tmp,maxindex]=max(entireRHS,[],1);
        Vhat(:,:,:,jj)=shiftdim(Vtilde_tmp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);
        % Now Vunderbar
        entireRHS=ReturnMatrix+beta*EV; % Use the two-future-periods discount factor
        maxindexfull=maxindex+N_a*(0:1:N_a-1)+shiftdim(N_a*N_a*(0:1:N_z-1),-1)+shiftdim(N_a*N_a*N_z*(0:1:N_e-1),-2);
        V(:,:,:,jj)=entireRHS(maxindexfull);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, n_z, special_n_e, 0, a_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0);

            % First Policy
            entireRHS_e=ReturnMatrix_e+beta0beta*EV; %.*ones(1,N_a,1);
            [Vtilde_tmp,maxindex]=max(entireRHS_e,[],1);
            Vhat(:,:,e_c,jj)=shiftdim(Vtilde_tmp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
            % Now Vunderbar
            entireRHS_e=ReturnMatrix_e+beta*EV; %.*ones(1,N_a,1);
            maxindexfull=maxindex+N_a*(0:1:N_a-1)+shiftdim(N_a*N_a*(0:1:N_z-1),-1);
            V(:,:,e_c,jj)=entireRHS_e(maxindexfull);
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, special_n_z, special_n_e, 0, a_grid, z_val, e_val, ReturnFnParamsVec,0);

                % First Policy
                entireRHS_ze=ReturnMatrix_ze+beta0beta*EV_z; %*ones(1,N_a,1);
                [Vtilde_tmp,maxindex]=max(entireRHS_ze,[],1);
                Vhat(:,z_c,e_c,jj)=Vtilde_tmp;
                Policy(:,z_c,e_c,jj)=maxindex;
                % Now Vunderbar
                entireRHS_ze=ReturnMatrix_ze+beta*EV_z; %*ones(1,N_a,1);
                maxindexfull=maxindex+N_a*(0:1:N_a-1);
                V(:,z_c,e_c,jj)=entireRHS_ze(maxindexfull);
            end
        end

    end
end

%% Output shape for policy
Policy=shiftdim(Policy,-1);


end
