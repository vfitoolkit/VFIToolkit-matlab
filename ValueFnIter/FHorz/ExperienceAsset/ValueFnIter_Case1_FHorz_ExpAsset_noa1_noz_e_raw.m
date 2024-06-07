function [V,Policy]=ValueFnIter_Case1_FHorz_ExpAsset_noa1_noz_e_raw(n_d1,n_d2,n_a2,n_e,N_j, d1_grid, d2_grid, a2_grid,e_gridvals_J,pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a2_grid=gpuArray(a2_grid);

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, n_e, [d1_grid; d2_grid], a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, special_n_e, [d1_grid; d2_grid], a2_grid, e_gridvals_J(e_c,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    % Using V_Jplus1
    Vnext=sum(pi_e_J(:,N_j)'.*reshape(vfoptions.V_Jplus1,[N_a,N_e]),2);    % First, switch V_Jplus1 into Kron form

    Vlower=reshape(Vnext(a2primeIndex),[N_d2,N_a2]);
    Vupper=reshape(Vnext(a2primeIndex+1),[N_d2,N_a2]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, n_e, [d1_grid; d2_grid], a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1,1); % Should autofill e dimension

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, special_n_e, [d1_grid; d2_grid], a2_grid, e_gridvals_J(e_c,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS=ReturnMatrix_e+DiscountFactorParamsVec*repelem(EV,N_d1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,e_c,N_j)=shiftdim(maxindex,1);
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

    Vnext=sum(pi_e_J(:,jj)'.*V(:,:,jj+1),2);

    Vlower=reshape(Vnext(a2primeIndex),[N_d2,N_a2]);
    Vupper=reshape(Vnext(a2primeIndex+1),[N_d2,N_a2]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, n_e, [d1_grid; d2_grid], a2_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1,1); % Should autofill e dimension

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, special_n_e, [d1_grid; d2_grid], a2_grid, e_gridvals_J(e_c,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS=ReturnMatrix_e+DiscountFactorParamsVec*repelem(EV,N_d1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,e_c,jj)=shiftdim(maxindex,1);
        end
    end
end


%% For experience asset, just output Policy as is and then use Case2 to UnKron
% Policy2=zeros(2,N_a,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
% Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d2)+1,-1);
% Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d2),-1);

end