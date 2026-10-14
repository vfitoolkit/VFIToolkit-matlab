function [V,Policy]=ValueFnIter_FHorz_ExpAssetu_noa1_noz_e_raw(n_d1,n_d2,n_a2,n_e,n_u, N_j, d_gridvals, d2_gridvals, a2_grid, e_gridvals_J, u_gridvals, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;
N_e=prod(n_e);
N_u=prod(n_u);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d2_gridvals=gpuArray(d2_gridvals);
a2_grid=gpuArray(a2_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % the CreateReturnFnMatrix_* commands want gridvals ([N_a2-by-l_a2]), not the stacked a2_grid.
% (These are the same array when there is only one experience asset, which is why passing a2_grid worked until l_a2=2.)

pi_u=shiftdim(pi_u,-2); % put it into third dimension

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, n_e, d_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, special_n_e, d_gridvals, a2_gridvals, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=sum(pi_e_J(:,N_j+1)'.*reshape(vfoptions.V_Jplus1,[N_a,N_e]),2); % First, switch V_Jplus1 into Kron form

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
        Vlower=reshape(EVpre(a2primeIndex(:)),[N_d2,N_a2,N_u]);
        Vupper=reshape(EVpre(a2primeIndex(:)+1),[N_d2,N_a2,N_u]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        a2primeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u)
        EV(a2primeProbs==0)=Vupper(a2primeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(a2primeProbs==1)=Vlower(a2primeProbs==1);
    else
        % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
        % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
        % does not poison the sum.
        V_ll=reshape(EVpre(a2k_ll(:)),[N_d2,N_a2,N_u]);
        V_hl=reshape(EVpre(a2k_hl(:)),[N_d2,N_a2,N_u]);
        V_lh=reshape(EVpre(a2k_lh(:)),[N_d2,N_a2,N_u]);
        V_hh=reshape(EVpre(a2k_hh(:)),[N_d2,N_a2,N_u]);
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

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, n_e, d_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1); % should autofill e dimension

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, special_n_e, d_gridvals, a2_gridvals, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

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

    EVpre=sum(pi_e_J(:,jj+1)'.*V(:,:,jj+1),2);

    if length(n_a2)==1
        Vlower=reshape(EVpre(a2primeIndex(:)),[N_d2,N_a2,N_u]);
        Vupper=reshape(EVpre(a2primeIndex(:)+1),[N_d2,N_a2,N_u]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        a2primeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u)
        EV(a2primeProbs==0)=Vupper(a2primeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(a2primeProbs==1)=Vlower(a2primeProbs==1);
    else
        % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
        % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
        % does not poison the sum.
        V_ll=reshape(EVpre(a2k_ll(:)),[N_d2,N_a2,N_u]);
        V_hl=reshape(EVpre(a2k_hl(:)),[N_d2,N_a2,N_u]);
        V_lh=reshape(EVpre(a2k_lh(:)),[N_d2,N_a2,N_u]);
        V_hh=reshape(EVpre(a2k_hh(:)),[N_d2,N_a2,N_u]);
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

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, n_e, d_gridvals, a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1); % should autofill e dimension

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn,[n_d1,n_d2], n_a2, special_n_e, d_gridvals, a2_gridvals, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS=ReturnMatrix_e+DiscountFactorParamsVec*repelem(EV,N_d1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,e_c,jj)=shiftdim(maxindex,1);
        end
    end
end

%%
Policy=shiftdim(Policy,-1);



end
