function [V,Policy]=ValueFnIter_FHorz_ExpAsset_noa1_noz_raw(n_d1,n_d2,n_a2,N_j, d_gridvals, d2_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d2_grid=gpuArray(d2_grid);
a2_grid=gpuArray(a2_grid);

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn,[n_d1,n_d2], n_a2, d_gridvals, a2_grid, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);

    Vlower=reshape(EVpre(a2primeIndex),[N_d2,N_a2]);
    Vupper=reshape(EVpre(a2primeIndex+1),[N_d2,N_a2]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn,[n_d1,n_d2], n_a2, d_gridvals, a2_grid, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,N_j)=shiftdim(Vtemp,1);
    Policy(:,N_j)=shiftdim(maxindex,1);

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

    Vlower=reshape(V(a2primeIndex,jj+1),[N_d2,N_a2]);
    Vupper=reshape(V(a2primeIndex+1,jj+1),[N_d2,N_a2]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn,[n_d1,n_d2], n_a2, d_gridvals, a2_grid, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,jj)=shiftdim(Vtemp,1);
    Policy(:,jj)=shiftdim(maxindex,1);

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron



end
