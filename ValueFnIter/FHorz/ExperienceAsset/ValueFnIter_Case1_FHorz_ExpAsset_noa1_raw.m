function [V,Policy]=ValueFnIter_Case1_FHorz_ExpAsset_noa1_raw(n_d1,n_d2,n_a2,n_z,N_j, d1_grid, d2_grid, a2_grid,z_gridvals_J,pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a2_grid=gpuArray(a2_grid);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, n_z, [d1_grid; d2_grid], a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, [d1_grid; d2_grid], a2_grid, z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]
    a2primeProbs=repmat(a2primeProbs,1,1,N_z);  % [N_d2,N_a2,N_z]
    
    Vnext=reshape(vfoptions.V_Jplus1,[N_a,N_z]);

    Vlower=reshape(Vnext(a2primeIndex,:),[N_d2,N_a2,N_z]);
    Vupper=reshape(Vnext(a2primeIndex+1,:),[N_d2,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    
    if vfoptions.lowmemory==0
        EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
        EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
        EV=squeeze(sum(EV,3));
        % EV is over (d2,a1prime,a2,z)

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, n_z, [d1_grid; d2_grid], a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            EV_z=EV.*shiftdim(pi_z_J(z_c,:,N_j)',-2);
            EV_z(isnan(EV_z))=0; % remove nan created where value fn is -Inf but probability is zero
            EV_z=squeeze(sum(EV_z,3));
            % EV is over (d2,a1prime,a2,z)

            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, [d1_grid; d2_grid], a2_grid, z_gridvals_J(z_c,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV_z,N_d1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,z_c,N_j)=shiftdim(Vtemp,1);
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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]
    a2primeProbs=repmat(a2primeProbs,1,1,N_z);  % [N_d2,N_a2,N_z]

    Vlower=reshape(V(a2primeIndex,:,jj+1),[N_d2,N_a2,N_z]);
    Vupper=reshape(V(a2primeIndex+1,:,jj+1),[N_d2,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    
    if vfoptions.lowmemory==0
        EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
        EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
        EV=squeeze(sum(EV,3));
        % EV is over (d2,a1prime,a2,z)

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, n_z, [d1_grid; d2_grid], a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            EV_z=EV.*shiftdim(pi_z_J(z_c,:,jj)',-2);
            EV_z(isnan(EV_z))=0; % remove nan created where value fn is -Inf but probability is zero
            EV_z=squeeze(sum(EV_z,3));
            % EV is over (d2,a1prime,a2,z)

            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, [d1_grid; d2_grid], a2_grid, z_gridvals_J(z_c,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV_z,N_d1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,z_c,jj)=shiftdim(Vtemp,1);
            Policy(:,z_c,jj)=shiftdim(maxindex,1);
        end
    end

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron



end