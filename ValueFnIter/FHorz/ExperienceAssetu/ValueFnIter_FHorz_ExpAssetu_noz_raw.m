function [V,Policy]=ValueFnIter_FHorz_ExpAssetu_noz_raw(n_d1, n_d2,n_a1,n_a2,n_u,N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_u=prod(n_u);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

pi_u=shiftdim(pi_u,-2); % put it into third dimension

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_u]

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);

    Vlower=reshape(EVpre(aprimeIndex(:)),[N_d2*N_a1,N_a2,N_u]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1,N_a2,N_u]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u)
    % Already applied the probabilities from interpolating onto grid
    EV=sum((EV.*pi_u),3); % (d2,a1prime,a2)

    ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0);

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,N_a1);

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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_u]

    Vlower=reshape(V(aprimeIndex(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
    Vupper=reshape(V(aprimeplus1Index(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u)
    % Already applied the probabilities from interpolating onto grid
    EV=sum((EV.*pi_u),3); % (d2,a1prime,a2)

    ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0);

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,N_a1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    
    V(:,jj)=shiftdim(Vtemp,1);
    Policy(:,jj)=shiftdim(maxindex,1);

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron



end
