function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_raw(V,n_d1,n_d2,n_a1,n_a2,n_z,N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory==3
    special_n_z=ones(1,length(n_z));
    special_n_ea=ones(1,length(n_a2));
end

%% j=N_j
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
    %Calc the max and its index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;
elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2, special_n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);
        % Calc the max and its index
        [Vtemp,maxindex]=max(ReturnMatrix_z);
        V(:,z_c,N_j)=Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end
elseif vfoptions.lowmemory==3
    for ea_c=1:N_a2
        ea_val=a2_gridvals(ea_c);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d1,n_d2,n_a1,n_a1, special_n_ea, special_n_z, d_gridvals, a1_gridvals, a1_gridvals, ea_val, z_val, ReturnFnParamsVec,0,0);
            % Calc the max and its index
            [Vtemp,maxindex]=max(ReturnMatrix_z);
            V(1+(ea_c-1)*N_a1:ea_c*N_a1,z_c,N_j)=Vtemp;
            Policy(1+(ea_c-1)*N_a1:ea_c*N_a1,z_c,N_j)=maxindex;
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z); % [N_d2*N_a1,N_a2,N_z]

    Vlower=reshape(V(aprimeIndex(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(V(aprimeplus1Index(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    if vfoptions.lowmemory==0
                
        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,N_a1);

        % Calc the max and its index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*repelem(EV(:,:,z_c),N_d1,N_a1);

            %Calc the max and its index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);

            V(:,z_c,jj)=shiftdim(Vtemp,1);
            Policy(:,z_c,jj)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==3
        for ea_c=1:N_a2
            ea_val=a2_gridvals(ea_c);
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,jj);
                ReturnMatrix_ea_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d1,n_d2,n_a1,n_a1,special_n_ea,special_n_z, d_gridvals, a1_gridvals, a1_gridvals, ea_val, z_val, ReturnFnParamsVec,0,0);
    
                entireRHS_ea_z=ReturnMatrix_ea_z+DiscountFactorParamsVec*repelem(EV(:,ea_c,z_c),N_d1,N_a1);
    
                %Calc the max and its index
                [Vtemp,maxindex]=max(entireRHS_ea_z,[],1);
    
                V(1+(ea_c-1)*N_a1:ea_c*N_a1,z_c,jj)=shiftdim(Vtemp,1);
                Policy(1+(ea_c-1)*N_a1:ea_c*N_a1,z_c,jj)=shiftdim(maxindex,1);
            end
        end
    end
end

%% Separate d and aprime
if vfoptions.gridinterplayer
    error("gridinterp not implemented with ExpAsset yet")
end


end
