function [V,Policy]=ValueFnIter_FHorz_ExpAssetu_nod1_noz_raw(n_d2,n_a1,n_a2,n_u,N_j, d2_gridvals, a1_gridvals, a2_grid, u_gridvals, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_u=prod(n_u);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

pi_u=shiftdim(pi_u,-2); % put it into third dimension

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a2, n_u, d2_gridvals, a2_grid, u_gridvals, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    if length(n_a2)==2
        % l_a2==2: the builder returns PER-DIM FACTORED output with the a2-dim index leading.
        % Pull out the two lower indexes and their probs, each shaped exactly as the l_a2==1
        % a2primeIndex/a2primeProbs would be, then Kron-fold the four corners of the a2 lattice.
        % a2k_ll/_hl/_lh/_hh are drop-in replacements for (a2primeIndex) and (a2primeIndex+1),
        % and pr_1/pr_2 for a2primeProbs, so every site below reads the same as the l_a2==1 one.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:),[N_d2,N_a2,N_u]);
        loIdx_2=reshape(a2primeIndex(2,:),[N_d2,N_a2,N_u]);
        pr_1=reshape(a2primeProbs(1,:),[N_d2,N_a2,N_u]);
        pr_2=reshape(a2primeProbs(2,:),[N_d2,N_a2,N_u]);
        a2k_ll=loIdx_1+n_a2_1*(loIdx_2-1);
        a2k_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        a2k_lh=loIdx_1+n_a2_1*loIdx_2;
        a2k_hh=(loIdx_1+1)+n_a2_1*loIdx_2;
        prob_1=pr_1;
        prob_2=pr_2;
    end
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
        aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_u]
    else
        aprime_ll=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_ll-1),N_a1,1);
        aprime_hl=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_hl-1),N_a1,1);
        aprime_lh=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_lh-1),N_a1,1);
        aprime_hh=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_hh-1),N_a1,1);
        prob_1=repmat(pr_1,N_a1,1,1);
        prob_2=repmat(pr_2,N_a1,1,1);
    end

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);

    if length(n_a2)==1
        Vlower=reshape(EVpre(aprimeIndex(:)),[N_d2*N_a1,N_a2,N_u]);
        Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1,N_a2,N_u]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u)
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
        % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
        % does not poison the sum.
        V_ll=reshape(EVpre(aprime_ll(:)),[N_d2*N_a1,N_a2,N_u]);
        V_hl=reshape(EVpre(aprime_hl(:)),[N_d2*N_a1,N_a2,N_u]);
        V_lh=reshape(EVpre(aprime_lh(:)),[N_d2*N_a1,N_a2,N_u]);
        V_hh=reshape(EVpre(aprime_hh(:)),[N_d2*N_a1,N_a2,N_u]);
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
    % Already applied the probabilities from interpolating onto grid
    EV=EV.*pi_u; EV(isnan(EV))=0; % a zero pi_u against an infinite node gives 0*(-Inf)=NaN, so zero the term BEFORE summing
    EV=sum(EV,3); % (d2,a1prime,a2)

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0,n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a2, n_u, d2_gridvals, a2_grid, u_gridvals, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    if length(n_a2)==2
        % l_a2==2: the builder returns PER-DIM FACTORED output with the a2-dim index leading.
        % Pull out the two lower indexes and their probs, each shaped exactly as the l_a2==1
        % a2primeIndex/a2primeProbs would be, then Kron-fold the four corners of the a2 lattice.
        % a2k_ll/_hl/_lh/_hh are drop-in replacements for (a2primeIndex) and (a2primeIndex+1),
        % and pr_1/pr_2 for a2primeProbs, so every site below reads the same as the l_a2==1 one.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:),[N_d2,N_a2,N_u]);
        loIdx_2=reshape(a2primeIndex(2,:),[N_d2,N_a2,N_u]);
        pr_1=reshape(a2primeProbs(1,:),[N_d2,N_a2,N_u]);
        pr_2=reshape(a2primeProbs(2,:),[N_d2,N_a2,N_u]);
        a2k_ll=loIdx_1+n_a2_1*(loIdx_2-1);
        a2k_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        a2k_lh=loIdx_1+n_a2_1*loIdx_2;
        a2k_hh=(loIdx_1+1)+n_a2_1*loIdx_2;
        prob_1=pr_1;
        prob_2=pr_2;
    end
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
        aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_u]
    else
        aprime_ll=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_ll-1),N_a1,1);
        aprime_hl=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_hl-1),N_a1,1);
        aprime_lh=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_lh-1),N_a1,1);
        aprime_hh=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_hh-1),N_a1,1);
        prob_1=repmat(pr_1,N_a1,1,1);
        prob_2=repmat(pr_2,N_a1,1,1);
    end

    if length(n_a2)==1
        Vlower=reshape(V(aprimeIndex(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
        Vupper=reshape(V(aprimeplus1Index(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u)
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
        % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
        % does not poison the sum.
        V_ll=reshape(V(aprime_ll(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
        V_hl=reshape(V(aprime_hl(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
        V_lh=reshape(V(aprime_lh(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
        V_hh=reshape(V(aprime_hh(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
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
    % Already applied the probabilities from interpolating onto grid
    EV=EV.*pi_u; EV(isnan(EV))=0; % a zero pi_u against an infinite node gives 0*(-Inf)=NaN, so zero the term BEFORE summing
    EV=sum(EV,3); % (d2,a1prime,a2)

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0, n_d2, n_a1,n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,jj)=shiftdim(Vtemp,1);
    Policy(:,jj)=shiftdim(maxindex,1);

end


%%
Policy=shiftdim(Policy,-1);


end
