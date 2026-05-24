function [V,Policy]=ValueFnIter_FHorz_ExpAssetze_noa1_e_raw(n_d1,n_d2,n_a2,n_z,n_e,N_j, d_gridvals, d2_gridvals, a2_grid,z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d, rest of dimensions a,z,e

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
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, special_n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, special_n_e, d_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2*N_z*N_e,1], whereas aprimeProbs is [N_d2,N_a2,N_z,N_e]   (z and e are both the current values)
    a2primeProbs=repmat(a2primeProbs,1,1,1,1,N_z);  % [N_d2,N_a2,N_z,N_e,N_z]   (replicate over zprime)

    EVpre=sum(shiftdim(pi_e_J(:,N_j),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3); % Integrate out eprime first

    Vlower=reshape(EVpre(a2primeIndex,:),[N_d2,N_a2,N_z,N_e,N_z]); % (d2,a2,z_cur,e_cur,zprime)
    Vupper=reshape(EVpre(a2primeIndex+1,:),[N_d2,N_a2,N_z,N_e,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a2,z_cur,e_cur,zprime)

    EV=EV.*reshape(pi_z_J(:,:,N_j),[1,1,N_z,1,N_z]); % pi[z_cur,z_prime] reshaped to broadcast: z_cur at dim 3, z_prime at dim 5
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=reshape(sum(EV,5),[N_d2,N_a2,N_z,N_e]); % sum zprime -> (d2,a2,z_cur,e_cur)

    DiscountedEV=DiscountFactorParamsVec*repelem(EV,N_d1,1,1,1);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+DiscountedEV;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, special_n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS=ReturnMatrix_e+DiscountedEV(:,:,:,e_c);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_ze=DiscountedEV(:,:,z_c,e_c);

                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, special_n_e, d_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

                entireRHS=ReturnMatrix_ze+DiscountedEV_ze;

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V(:,z_c,e_c,N_j)=shiftdim(Vtemp,1);
                Policy(:,z_c,e_c,N_j)=shiftdim(maxindex,1);
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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2*N_z*N_e,1], whereas aprimeProbs is [N_d2,N_a2,N_z,N_e]   (z and e are both the current values)
    a2primeProbs=repmat(a2primeProbs,1,1,1,1,N_z);  % [N_d2,N_a2,N_z,N_e,N_z]   (replicate over zprime)

    EVpre=sum(shiftdim(pi_e_J(:,jj),-2).*V(:,:,:,jj+1),3); % Integrate out eprime first

    Vlower=reshape(EVpre(a2primeIndex,:),[N_d2,N_a2,N_z,N_e,N_z]); % (d2,a2,z_cur,e_cur,zprime)
    Vupper=reshape(EVpre(a2primeIndex+1,:),[N_d2,N_a2,N_z,N_e,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a2,z_cur,e_cur,zprime)

    EV=EV.*reshape(pi_z_J(:,:,jj),[1,1,N_z,1,N_z]); % pi[z_cur,z_prime] reshaped to broadcast: z_cur at dim 3, z_prime at dim 5
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=reshape(sum(EV,5),[N_d2,N_a2,N_z,N_e]); % sum zprime -> (d2,a2,z_cur,e_cur)

    DiscountedEV=DiscountFactorParamsVec*repelem(EV,N_d1,1,1,1);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+DiscountedEV;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, special_n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS=ReturnMatrix_e+DiscountedEV(:,:,:,e_c);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_ze=DiscountedEV(:,:,z_c,e_c);

                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, special_n_e, d_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

                entireRHS=ReturnMatrix_ze+DiscountedEV_ze;

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V(:,z_c,e_c,jj)=shiftdim(Vtemp,1);
                Policy(:,z_c,e_c,jj)=shiftdim(maxindex,1);
            end
        end
    end

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron



end
