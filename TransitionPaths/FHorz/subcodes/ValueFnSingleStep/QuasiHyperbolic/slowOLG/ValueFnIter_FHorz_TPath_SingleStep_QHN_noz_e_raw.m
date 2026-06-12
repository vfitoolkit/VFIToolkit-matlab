function [V,Policy,Policyalt,Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
Policyalt=zeros(N_a,N_e,N_j,'gpuArray'); % exponential discounter optimal choice (Valt is computed at this)
Vtilde=zeros(N_a,N_e,N_j,'gpuArray'); % agent's-perspective value at QH-optimal policy under beta0beta

Vnext=sum(V.*shiftdim(pi_e_J(:,[1,1:end-1]),-1),2); % Take expectations over e: Vnext(...,jj+1) is read for current age jj, so weight V at age jj+1 by pi_e_J(:,jj) [same timing as standard ValueFnIter commands]; first column is padding, never read

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=K not supported for ValueFnIter_FHorz_TPath_SingleStep_QHN_noz_e_raw')
end

%% j=N_j
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;
    Policyalt(:,:,N_j)=maxindex; % terminal period: QH and exponential discounter coincide
    Vtilde(:,:,N_j)=V(:,:,N_j); % terminal: no continuation, Vtilde=Valt

elseif vfoptions.lowmemory==1

    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,0);  % Because no z, can treat e like z and call Par2 rather than Par2e
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
        V(:,e_c,N_j)=Vtemp;
        Policy(:,e_c,N_j)=maxindex;
        Policyalt(:,e_c,N_j)=maxindex;
        Vtilde(:,e_c,N_j)=V(:,e_c,N_j); % terminal: no continuation, Vtilde=Valt
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
    entireEV=repelem(EV,N_d,1,1);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);
        % (d,aprime,a,e)

        % First Valt
        entireRHS_alt=ReturnMatrix+beta*entireEV;
        [Vtemp,maxindex_alt]=max(entireRHS_alt,[],1);
        V(:,:,jj)=shiftdim(Vtemp,1);
        Policyalt(:,:,jj)=shiftdim(maxindex_alt,1);
        % Now Policy
        entireRHS=ReturnMatrix+beta0beta*entireEV;
        [Vtilde_tmp,maxindex]=max(entireRHS,[],1);
        Vtilde(:,:,jj)=shiftdim(Vtilde_tmp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,0);
            % (d,aprime,a)

            % First Valt
            entireRHS_alt_e=ReturnMatrix_e+beta*entireEV;
            [Vtemp,maxindex_alt]=max(entireRHS_alt_e,[],1);
            V(:,e_c,jj)=shiftdim(Vtemp,1);
            Policyalt(:,e_c,jj)=shiftdim(maxindex_alt,1);
            % Now Policy
            entireRHS_e=ReturnMatrix_e+beta0beta*entireEV;
            [Vtilde_tmp,maxindex]=max(entireRHS_e,[],1);
            Vtilde(:,e_c,jj)=shiftdim(Vtilde_tmp,1);
            Policy(:,e_c,jj)=shiftdim(maxindex,1);
        end

    end
end

Policy=shiftdim(Policy,-1);
Policyalt=shiftdim(Policyalt,-1);

end
