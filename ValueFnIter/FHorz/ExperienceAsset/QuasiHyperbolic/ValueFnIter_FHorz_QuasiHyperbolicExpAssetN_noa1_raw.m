function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetN_noa1_raw(n_d1,n_d2,n_a2,n_z,N_j, d_gridvals, d2_gridvals, a2_grid,z_gridvals_J,pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);

Valt=zeros(N_a,N_z,N_j,'gpuArray');
Vtilde=zeros(N_a,N_z,N_j,'gpuArray');
Policyalt=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z
Policy=zeros(N_a,N_z,N_j,'gpuArray');

%%
d2_gridvals=gpuArray(d2_gridvals);
a2_grid=gpuArray(a2_grid);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, n_z, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,N_j)=Vtemp;
        Policyalt(:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, d_gridvals, a2_grid, z_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Valt(:,z_c,N_j)=Vtemp;
            Policyalt(:,z_c,N_j)=maxindex;
        end
    end
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy(:,:,N_j)=Policyalt(:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]
    a2primeProbs=repmat(a2primeProbs,1,1,N_z);  % [N_d2,N_a2,N_z]

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]);

    Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_z]);
    Vupper=reshape(EV(a2primeIndex+1,:),[N_d2,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    EVbase_qh=repelem(EV,N_d1,1);
    DiscountedEV_alt=beta*EVbase_qh;
    DiscountedEV_tilde=beta0beta*EVbase_qh;

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, n_z, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS_alt=ReturnMatrix+DiscountedEV_alt;
        [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
        Valt(:,:,N_j)=shiftdim(Vtemp_alt,1);
        Policyalt(:,:,N_j)=shiftdim(maxindex_alt,1);
        entireRHS_tilde=ReturnMatrix+DiscountedEV_tilde;
        [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
        Vtilde(:,:,N_j)=shiftdim(Vtemp_tilde,1);
        Policy(:,:,N_j)=shiftdim(maxindex_tilde,1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z_alt=DiscountedEV_alt(:,:,z_c);
            DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,z_c);

            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, d_gridvals, a2_grid, z_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS_alt=ReturnMatrix+DiscountedEV_z_alt;
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            Valt(:,z_c,N_j)=shiftdim(Vtemp_alt,1);
            Policyalt(:,z_c,N_j)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix+DiscountedEV_z_tilde;
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            Vtilde(:,z_c,N_j)=shiftdim(Vtemp_tilde,1);
            Policy(:,z_c,N_j)=shiftdim(maxindex_tilde,1);
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
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]
    a2primeProbs=repmat(a2primeProbs,1,1,N_z);  % [N_d2,N_a2,N_z]

    Vlower=reshape(Valt(a2primeIndex,:,jj+1),[N_d2,N_a2,N_z]);
    Vupper=reshape(Valt(a2primeIndex+1,:,jj+1),[N_d2,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    EVbase_qh=repelem(EV,N_d1,1);
    DiscountedEV_alt=beta*EVbase_qh;
    DiscountedEV_tilde=beta0beta*EVbase_qh;

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, n_z, d_gridvals, a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS_alt=ReturnMatrix+DiscountedEV_alt;
        [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
        Valt(:,:,jj)=shiftdim(Vtemp_alt,1);
        Policyalt(:,:,jj)=shiftdim(maxindex_alt,1);
        entireRHS_tilde=ReturnMatrix+DiscountedEV_tilde;
        [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
        Vtilde(:,:,jj)=shiftdim(Vtemp_tilde,1);
        Policy(:,:,jj)=shiftdim(maxindex_tilde,1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z_alt=DiscountedEV_alt(:,:,z_c);
            DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,z_c);

            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, d_gridvals, a2_grid, z_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS_alt=ReturnMatrix+DiscountedEV_z_alt;
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            Valt(:,z_c,jj)=shiftdim(Vtemp_alt,1);
            Policyalt(:,z_c,jj)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix+DiscountedEV_z_tilde;
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            Vtilde(:,z_c,jj)=shiftdim(Vtemp_tilde,1);
            Policy(:,z_c,jj)=shiftdim(maxindex_tilde,1);
        end
    end

end

%%
Policy=shiftdim(Policy,-1);
Policyalt=shiftdim(Policyalt,-1);


end
