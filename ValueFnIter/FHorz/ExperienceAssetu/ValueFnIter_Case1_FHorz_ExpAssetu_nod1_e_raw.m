function [V,Policy]=ValueFnIter_Case1_FHorz_ExpAssetu_nod1_e_raw(n_d2,n_a1,n_a2,n_z,n_e,n_u,N_j, d2_grid, a1_grid, a2_grid, z_gridvals_J, e_gridvals_J, u_grid, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);
N_u=prod(n_u);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d2_grid=gpuArray(d2_grid);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);

pi_u=shiftdim(pi_u,-2); % put it into third dimension

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
        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d2, n_a1,n_a2,n_z,n_e, d2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d2, n_a1,n_a2,n_z,special_n_e, d2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(e_c,:,N_j), ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            for e_c=1:N_e
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d2, n_a1,n_a2,special_n_z,special_n_e, d2_grid, a1_grid, a2_grid, z_gridvals_J(z_c,:,N_j), e_gridvals_J(e_c,:,N_j), ReturnFnParamsVec);
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

    Vnext=sum(shiftdim(pi_e_J(:,N_j),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3); % Expectations over e

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_u)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_z); % [N_d2*N_a1,N_a2,N_u,N_z]

    Vlower=reshape(Vnext(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_u,N_z]);
    Vupper=reshape(Vnext(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_u,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    % Already applied the probabilities from interpolating onto grid
    EV=sum((EV.*pi_u),3); % (d2,a1prime,a2,zprime)

    if vfoptions.lowmemory==0
        EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
        EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
        EV=squeeze(sum(EV,3));
        % EV is over (d2,a1prime,a2,z)

        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d2, n_a1,n_a2,n_z,n_e, d2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,1,N_a1,1); % should autofill e dimension

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
        EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
        EV=squeeze(sum(EV,3));
        % EV is over (d2,a1prime,a2,z)

        for e_c=1:N_e
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d2, n_a1,n_a2,n_z,special_n_e, d2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(e,:,N_j), ReturnFnParamsVec);

            entireRHS=ReturnMatrix_e+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            % Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=EV.*shiftdim(pi_z_J(z_c,:,N_j)',-2);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,3);

            for e_c=1:N_e
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d2, n_a1,n_a2,special_n_z,special_n_e, d2_grid, a1_grid, a2_grid, z_gridvals_J(z_c,:,N_j), e_gridvals_J(e,:,N_j), ReturnFnParamsVec);

                entireRHS=ReturnMatrix_ze+DiscountFactorParamsVec*repelem(EV_z,1,N_a1,1);

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

    Vnext=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_u)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_z); % [N_d2*N_a1,N_a2,N_u,N_z]

    Vlower=reshape(Vnext(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_u,N_z]);
    Vupper=reshape(Vnext(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_u,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    % Already applied the probabilities from interpolating onto grid
    EV=sum((EV.*pi_u),3); % (d2,a1prime,a2,zprime)

    if vfoptions.lowmemory==0
        EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
        EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
        EV=squeeze(sum(EV,3));
        % EV is over (d2,a1prime,a2,z)

        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d2, n_a1,n_a2,n_z,n_e, d2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,1,N_a1,1); % should autofill e dimension

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
        EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
        EV=squeeze(sum(EV,3));
        % EV is over (d2,a1prime,a2,z)

        for e_c=1:N_e
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d2, n_a1,n_a2,n_z,special_n_e, d2_grid, a1_grid, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(e,:,jj), ReturnFnParamsVec);

            entireRHS=ReturnMatrix_e+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            % Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=EV.*shiftdim(pi_z_J(z_c,:,jj)',-2);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,3);

            for e_c=1:N_e
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d2, n_a1,n_a2,special_n_z,special_n_e, d2_grid, a1_grid, a2_grid, z_gridvals_J(z_c,:,jj), e_gridvals_J(e,:,jj), ReturnFnParamsVec);

                entireRHS=ReturnMatrix_ze+DiscountFactorParamsVec*repelem(EV_z,1,N_a1,1);

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