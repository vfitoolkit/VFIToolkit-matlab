function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_nod1_noa1_e_raw(n_d2,n_a2,n_z,n_e,N_j, d2_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive QH + ExpAssete (e dep aprimeFn), baseline (no DC, no GI), no d1, no a1, with z.
% Experience asset a2 is the only endogenous state, so d2 is the only choice.
% lowmemory=0 full vectorization; lowmemory=1 loops e; lowmemory=2 nested z+e.

N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

Valt=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d, rest of dimensions a,z,e
Policyalt=zeros(N_a,N_z,N_e,N_j,'gpuArray');

%%
d2_gridvals=gpuArray(d2_gridvals);
a2_grid=gpuArray(a2_grid);


if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
end


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_d2, n_a2, n_z,n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policyalt(:,:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_d2, n_a2, n_z,special_n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(e_c,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Valt(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policyalt(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            for e_c=1:N_e
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_d2, n_a2, special_n_z,special_n_e, d2_gridvals, a2_grid, z_gridvals_J(z_c,:,N_j), e_gridvals_J(e_c,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Valt(:,z_c,e_c,N_j)=Vtemp;
                Policyalt(:,z_c,e_c,N_j)=maxindex;
            end
        end
    end
    Vtilde(:,:,:,N_j)=Valt(:,:,:,N_j); % no future, so nothing is discounted
    Policy(:,:,:,N_j)=Policyalt(:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2*N_e,1], whereas aprimeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)
    a2primeProbs=repmat(a2primeProbs,1,1,1,N_z);  % [N_d2,N_a2,N_e,N_z]   (replicate over zprime)

    EVpre=sum(shiftdim(pi_e_J(:,N_j+1),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3); % Integrate out eprime first

    Vlower=reshape(EVpre(a2primeIndex,:),[N_d2,N_a2,N_e,N_z]); % (d2,a2,e_cur,zprime)
    Vupper=reshape(EVpre(a2primeIndex+1,:),[N_d2,N_a2,N_e,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a2,e_cur,zprime)

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-3); % pi'(zprime,zcur) shaped [1,1,1,zprime,zcur]
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=reshape(sum(EV,4),[N_d2,N_a2,N_e,N_z]); % sum zprime -> (d2,a2,e_cur,zcur)
    EV=permute(EV,[1,2,4,3]); % (d2,a2,zcur,e_cur)  -- match ReturnMatrix dim order
    % EV is left undiscounted; beta/beta0beta are applied below

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+beta*EV;
        %Calc the max and it's index
        [Vtemp,maxindexalt]=max(entireRHS,[],1);
        Valt(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policyalt(:,:,:,N_j)=shiftdim(maxindexalt,1);

        entireRHS=ReturnMatrix+beta0beta*EV;
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vtilde(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_d2, n_a2, n_z, special_n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS_e=ReturnMatrix_e+beta*EV(:,:,:,e_c);
            %Calc the max and it's index
            [Vtemp,maxindexalt]=max(entireRHS_e,[],1);
            Valt(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policyalt(:,:,e_c,N_j)=shiftdim(maxindexalt,1);

            entireRHS_e=ReturnMatrix_e+beta0beta*EV(:,:,:,e_c);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            Vtilde(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_d2, n_a2, special_n_z, special_n_e, d2_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

                entireRHS_ze=ReturnMatrix_ze+beta*EV(:,:,z_c,e_c);
                %Calc the max and it's index
                [Vtemp,maxindexalt]=max(entireRHS_ze,[],1);
                Valt(:,z_c,e_c,N_j)=Vtemp;
                Policyalt(:,z_c,e_c,N_j)=maxindexalt;

                entireRHS_ze=ReturnMatrix_ze+beta0beta*EV(:,:,z_c,e_c);
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                Vtilde(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2*N_e,1], whereas aprimeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)
    a2primeProbs=repmat(a2primeProbs,1,1,1,N_z);  % [N_d2,N_a2,N_e,N_z]   (replicate over zprime)

    EVpre=sum(shiftdim(pi_e_J(:,jj+1),-2).*Valt(:,:,:,jj+1),3); % Integrate out eprime first (naive: continuation is the exponential Valt)

    Vlower=reshape(EVpre(a2primeIndex,:),[N_d2,N_a2,N_e,N_z]); % (d2,a2,e_cur,zprime)
    Vupper=reshape(EVpre(a2primeIndex+1,:),[N_d2,N_a2,N_e,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a2,e_cur,zprime)

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-3); % pi'(zprime,zcur) shaped [1,1,1,zprime,zcur]
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=reshape(sum(EV,4),[N_d2,N_a2,N_e,N_z]); % sum zprime -> (d2,a2,e_cur,zcur)
    EV=permute(EV,[1,2,4,3]); % (d2,a2,zcur,e_cur)  -- match ReturnMatrix dim order
    % EV is left undiscounted; beta/beta0beta are applied below

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+beta*EV;
        %Calc the max and it's index
        [Vtemp,maxindexalt]=max(entireRHS,[],1);
        Valt(:,:,:,jj)=shiftdim(Vtemp,1);
        Policyalt(:,:,:,jj)=shiftdim(maxindexalt,1);

        entireRHS=ReturnMatrix+beta0beta*EV;
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vtilde(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_d2, n_a2, n_z, special_n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS_e=ReturnMatrix_e+beta*EV(:,:,:,e_c);
            %Calc the max and it's index
            [Vtemp,maxindexalt]=max(entireRHS_e,[],1);
            Valt(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policyalt(:,:,e_c,jj)=shiftdim(maxindexalt,1);

            entireRHS_e=ReturnMatrix_e+beta0beta*EV(:,:,:,e_c);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            Vtilde(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,n_d2, n_a2, special_n_z, special_n_e, d2_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

                entireRHS_ze=ReturnMatrix_ze+beta*EV(:,:,z_c,e_c);
                %Calc the max and it's index
                [Vtemp,maxindexalt]=max(entireRHS_ze,[],1);
                Valt(:,z_c,e_c,jj)=Vtemp;
                Policyalt(:,z_c,e_c,jj)=maxindexalt;

                entireRHS_ze=ReturnMatrix_ze+beta0beta*EV(:,:,z_c,e_c);
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                Vtilde(:,z_c,e_c,jj)=Vtemp;
                Policy(:,z_c,e_c,jj)=maxindex;
            end
        end
    end

end


%%
Policy=shiftdim(Policy,-1);
Policyalt=shiftdim(Policyalt,-1);

end
