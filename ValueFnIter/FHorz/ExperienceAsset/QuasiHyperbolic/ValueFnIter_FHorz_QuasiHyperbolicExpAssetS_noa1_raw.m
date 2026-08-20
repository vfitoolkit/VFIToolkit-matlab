function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetS_noa1_raw(n_d1,n_d2,n_a2,n_z,N_j, d_gridvals, d2_gridvals, a2_grid,z_gridvals_J,pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2; % argmax row dim: the ReturnMatrix is built over [n_d1,n_d2]
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);

Vhat=zeros(N_a,N_z,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

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
        Vhat(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, d_gridvals, a2_grid, z_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Vhat(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end
    end
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);
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
    DiscountedEV_under=beta*EVbase_qh;
    DiscountedEV_hat=beta0beta*EVbase_qh;

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, n_z, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        % hat: argmax of F + beta0*beta*EV; under: the beta-RHS gathered at that argmax
        entireRHS_hat=ReturnMatrix+DiscountedEV_hat;
        [Vtemp,maxindex]=max(entireRHS_hat,[],1);
        entireRHS_under=ReturnMatrix+DiscountedEV_under;
        maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_z-1),-1);
        Vhat(:,:,N_j)=shiftdim(Vtemp,1);
        Vunderbar(:,:,N_j)=shiftdim(entireRHS_under(maxindexfull),1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z_hat=DiscountedEV_hat(:,:,z_c);
            DiscountedEV_z_under=DiscountedEV_under(:,:,z_c);

            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, d_gridvals, a2_grid, z_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            % hat: argmax of F + beta0*beta*EV; under: the beta-RHS gathered at that argmax
            entireRHS_hat=ReturnMatrix+DiscountedEV_z_hat;
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            entireRHS_under=ReturnMatrix+DiscountedEV_z_under;
            maxindexfull=maxindex+N_d12*(0:1:N_a-1);
            Vhat(:,z_c,N_j)=shiftdim(Vtemp,1);
            Vunderbar(:,z_c,N_j)=shiftdim(entireRHS_under(maxindexfull),1);
            Policy(:,z_c,N_j)=shiftdim(maxindex,1);
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

    Vlower=reshape(Vunderbar(a2primeIndex,:,jj+1),[N_d2,N_a2,N_z]);
    Vupper=reshape(Vunderbar(a2primeIndex+1,:,jj+1),[N_d2,N_a2,N_z]);
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
    DiscountedEV_under=beta*EVbase_qh;
    DiscountedEV_hat=beta0beta*EVbase_qh;

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, n_z, d_gridvals, a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        % hat: argmax of F + beta0*beta*EV; under: the beta-RHS gathered at that argmax
        entireRHS_hat=ReturnMatrix+DiscountedEV_hat;
        [Vtemp,maxindex]=max(entireRHS_hat,[],1);
        entireRHS_under=ReturnMatrix+DiscountedEV_under;
        maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_z-1),-1);
        Vhat(:,:,jj)=shiftdim(Vtemp,1);
        Vunderbar(:,:,jj)=shiftdim(entireRHS_under(maxindexfull),1);
        Policy(:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z_hat=DiscountedEV_hat(:,:,z_c);
            DiscountedEV_z_under=DiscountedEV_under(:,:,z_c);

            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, d_gridvals, a2_grid, z_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            % hat: argmax of F + beta0*beta*EV; under: the beta-RHS gathered at that argmax
            entireRHS_hat=ReturnMatrix+DiscountedEV_z_hat;
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            entireRHS_under=ReturnMatrix+DiscountedEV_z_under;
            maxindexfull=maxindex+N_d12*(0:1:N_a-1);
            Vhat(:,z_c,jj)=shiftdim(Vtemp,1);
            Vunderbar(:,z_c,jj)=shiftdim(entireRHS_under(maxindexfull),1);
            Policy(:,z_c,jj)=shiftdim(maxindex,1);
        end
    end

end

%%
Policy=shiftdim(Policy,-1);


end
