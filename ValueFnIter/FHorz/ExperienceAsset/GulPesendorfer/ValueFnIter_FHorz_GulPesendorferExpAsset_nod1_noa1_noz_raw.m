function [V,Policy]=ValueFnIter_FHorz_GulPesendorferExpAsset_nod1_noa1_noz_raw(n_d2,n_a2,N_j, d2_gridvals, a2_grid, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions)
% Gul-Pesendorfer with an experience asset: V_j = max_{d2} [u+v+beta*E V_{j+1}] - max_{d2} v.
% The tempted objective goes through the standard ExpAsset machinery (the continuation is at
% a2prime=aprimeFn(d2,a2), interpolated onto the a2 grid); the most-tempting term is a max of v
% over the FULL d2 choice set and is subtracted AFTER the max (v never touches
% the continuation, so the temptation side needs none of the aprime-probs machinery).

N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d2_gridvals=gpuArray(d2_gridvals);
a2_grid=gpuArray(a2_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % the CreateReturnFnMatrix_Case2_Disc* commands want gridvals ([N_a2-by-l_a2]), not the stacked a2_grid.
% (These are the same array when there is only one experience asset, which is why passing a2_grid worked until l_a2=2.)

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz(ReturnFn,n_d2, n_a2, d2_gridvals, a2_gridvals, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    TemptationMatrix=CreateReturnFnMatrix_Case2_Disc_noz(TemptationFn,n_d2, n_a2, d2_gridvals, a2_gridvals, TemptationFnParamsVec); % with only the experience asset, can just use Case2 command
    MostTempting=max(TemptationMatrix,[],1);
    MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
    entireRHS=ReturnMatrix+TemptationMatrix;
    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,N_j)=Vtemp-MostTempting;
    Policy(:,N_j)=maxindex;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]); % First, switch V_Jplus1 into Kron form

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    if length(n_a2)==1
        Vlower=reshape(EVpre(a2primeIndex),[N_d2,N_a2]);
        Vupper=reshape(EVpre(a2primeIndex+1),[N_d2,N_a2]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        a2primeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
        EV(a2primeProbs==0)=Vupper(a2primeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(a2primeProbs==1)=Vlower(a2primeProbs==1);
    else
        % l_a2==2: a2primeIndex is [l_a2,N_d2*N_a2] and a2primeProbs is [l_a2,N_d2,N_a2],
        % per-dim factored rather than a single lower corner. With no a1, the aprime index is
        % just the Kron index in the a2 product space. Nested 2-corner interp with skipinterp
        % at each level, and per-contribution NaN cleanup so that 0*(-Inf) at a zero-prob
        % corner does not poison the sum.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:),[N_d2,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:),[N_d2,N_a2]);
        prob_1=reshape(a2primeProbs(1,:,:),[N_d2,N_a2]);
        prob_2=reshape(a2primeProbs(2,:,:),[N_d2,N_a2]);
        aprime_ll=loIdx_1+n_a2_1*(loIdx_2-1);
        aprime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        aprime_lh=loIdx_1+n_a2_1*loIdx_2;
        aprime_hh=(loIdx_1+1)+n_a2_1*loIdx_2;
        V_ll=reshape(EVpre(aprime_ll(:)),[N_d2,N_a2]);
        V_hl=reshape(EVpre(aprime_hl(:)),[N_d2,N_a2]);
        V_lh=reshape(EVpre(aprime_lh(:)),[N_d2,N_a2]);
        V_hh=reshape(EVpre(aprime_hh(:)),[N_d2,N_a2]);
        % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
        p1_lo=prob_1; p1_lo(V_ll==V_hl)=0;
        c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
        EV_lo=c_ll+c_hl;
        p1_hi=prob_1; p1_hi(V_lh==V_hh)=0;
        c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hi=c_lh+c_hh;
        % outer level: interpolate those two over the a2_2 dimension
        p2=prob_2; p2(EV_lo==EV_hi)=0;
        c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
        c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
        EV=c_lo+c_hi;
    end
    EV(isnan(EV))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz(ReturnFn,n_d2, n_a2, d2_gridvals, a2_gridvals, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    TemptationMatrix=CreateReturnFnMatrix_Case2_Disc_noz(TemptationFn,n_d2, n_a2, d2_gridvals, a2_gridvals, TemptationFnParamsVec); % with only the experience asset, can just use Case2 command
    MostTempting=max(TemptationMatrix,[],1);
    MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

    entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,N_j)=shiftdim(Vtemp-MostTempting,1);
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
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    if length(n_a2)==1
        Vlower=reshape(V(a2primeIndex,jj+1),[N_d2,N_a2]);
        Vupper=reshape(V(a2primeIndex+1,jj+1),[N_d2,N_a2]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        a2primeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
        EV(a2primeProbs==0)=Vupper(a2primeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(a2primeProbs==1)=Vlower(a2primeProbs==1);
    else
        % l_a2==2: a2primeIndex is [l_a2,N_d2*N_a2] and a2primeProbs is [l_a2,N_d2,N_a2],
        % per-dim factored rather than a single lower corner. With no a1, the aprime index is
        % just the Kron index in the a2 product space. Nested 2-corner interp with skipinterp
        % at each level, and per-contribution NaN cleanup so that 0*(-Inf) at a zero-prob
        % corner does not poison the sum.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:),[N_d2,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:),[N_d2,N_a2]);
        prob_1=reshape(a2primeProbs(1,:,:),[N_d2,N_a2]);
        prob_2=reshape(a2primeProbs(2,:,:),[N_d2,N_a2]);
        aprime_ll=loIdx_1+n_a2_1*(loIdx_2-1);
        aprime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        aprime_lh=loIdx_1+n_a2_1*loIdx_2;
        aprime_hh=(loIdx_1+1)+n_a2_1*loIdx_2;
        V_ll=reshape(V(aprime_ll(:),jj+1),[N_d2,N_a2]);
        V_hl=reshape(V(aprime_hl(:),jj+1),[N_d2,N_a2]);
        V_lh=reshape(V(aprime_lh(:),jj+1),[N_d2,N_a2]);
        V_hh=reshape(V(aprime_hh(:),jj+1),[N_d2,N_a2]);
        % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
        p1_lo=prob_1; p1_lo(V_ll==V_hl)=0;
        c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
        EV_lo=c_ll+c_hl;
        p1_hi=prob_1; p1_hi(V_lh==V_hh)=0;
        c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hi=c_lh+c_hh;
        % outer level: interpolate those two over the a2_2 dimension
        p2=prob_2; p2(EV_lo==EV_hi)=0;
        c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
        c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
        EV=c_lo+c_hi;
    end
    EV(isnan(EV))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz(ReturnFn,n_d2, n_a2, d2_gridvals, a2_gridvals, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    TemptationMatrix=CreateReturnFnMatrix_Case2_Disc_noz(TemptationFn,n_d2, n_a2, d2_gridvals, a2_gridvals, TemptationFnParamsVec); % with only the experience asset, can just use Case2 command
    MostTempting=max(TemptationMatrix,[],1);
    MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

    entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,jj)=shiftdim(Vtemp-MostTempting,1);
    Policy(:,jj)=shiftdim(maxindex,1);

end

%%
Policy=shiftdim(Policy,-1);


end
