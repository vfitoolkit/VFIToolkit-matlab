function [V,Policy]=ValueFnIter_FHorz_ExpAssete_nod1_noa1_noz_e_raw(n_d2,n_a2,n_e,N_j, d2_gridvals, a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d, rest of dimensions a,e

%%
d2_gridvals=gpuArray(d2_gridvals);
a2_grid=gpuArray(a2_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % the CreateReturnFnMatrix_* commands want gridvals ([N_a2-by-l_a2]), not the stacked a2_grid.
% (These are the same array when there is only one experience asset, which is why passing a2_grid worked until l_a2=2.)

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_d2, n_a2, n_e, d2_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_d2, n_a2, special_n_e, d2_gridvals, a2_gridvals, e_gridvals_J(e_c,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2*N_e,1], whereas aprimeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)

    EVpre=sum(pi_e_J(:,N_j+1)'.*reshape(vfoptions.V_Jplus1,[N_a,N_e]),2);    % Integrate out eprime: [N_a,1]

    if length(n_a2)==1
        Vlower=reshape(EVpre(a2primeIndex),[N_d2,N_a2,N_e]); % (d2,a2,e_cur)
        Vupper=reshape(EVpre(a2primeIndex+1),[N_d2,N_a2,N_e]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        a2primeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a2,e_cur)
        EV(a2primeProbs==0)=Vupper(a2primeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(a2primeProbs==1)=Vlower(a2primeProbs==1);
    else
        % l_a2==2: a2primeIndex/a2primeProbs are per-dim factored rather than a single lower
        % corner. Fold the two per-dim lower indexes into the four corners of the a2 lattice and
        % do a nested 2-corner interp, with skipinterp at each level and per-contribution NaN
        % cleanup so that 0*(-Inf) does not poison the sum.
        n_a2_1=n_a2(1);
        loIdx_1=a2primeIndex(1,:)';
        loIdx_2=a2primeIndex(2,:)';
        prob_1=reshape(a2primeProbs(1,:),[N_d2,N_a2,N_e]);
        prob_2=reshape(a2primeProbs(2,:),[N_d2,N_a2,N_e]);
        aprime_ll= loIdx_1   +n_a2_1*(loIdx_2-1);
        aprime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        aprime_lh= loIdx_1   +n_a2_1* loIdx_2;
        aprime_hh=(loIdx_1+1)+n_a2_1* loIdx_2;
        V_ll=reshape(EVpre(aprime_ll),[N_d2,N_a2,N_e]);
        V_hl=reshape(EVpre(aprime_hl),[N_d2,N_a2,N_e]);
        V_lh=reshape(EVpre(aprime_lh),[N_d2,N_a2,N_e]);
        V_hh=reshape(EVpre(aprime_hh),[N_d2,N_a2,N_e]);
        % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
        p1_loy=prob_1; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        % outer level: interpolate those two over the a2_2 dimension
        p2=prob_2; p2(EV_loy==EV_hiy)=0;
        c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_d2, n_a2, n_e, d2_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_d2, n_a2, special_n_e, d2_gridvals, a2_gridvals, e_gridvals_J(e_c,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS=ReturnMatrix_e+DiscountFactorParamsVec*EV(:,:,e_c);

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
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2*N_e,1], whereas aprimeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)

    EVpre=sum(pi_e_J(:,jj+1)'.*V(:,:,jj+1),2);    % Integrate out eprime: [N_a,1]

    if length(n_a2)==1
        Vlower=reshape(EVpre(a2primeIndex),[N_d2,N_a2,N_e]); % (d2,a2,e_cur)
        Vupper=reshape(EVpre(a2primeIndex+1),[N_d2,N_a2,N_e]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        a2primeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a2,e_cur)
        EV(a2primeProbs==0)=Vupper(a2primeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(a2primeProbs==1)=Vlower(a2primeProbs==1);
    else
        % l_a2==2: a2primeIndex/a2primeProbs are per-dim factored rather than a single lower
        % corner. Fold the two per-dim lower indexes into the four corners of the a2 lattice and
        % do a nested 2-corner interp, with skipinterp at each level and per-contribution NaN
        % cleanup so that 0*(-Inf) does not poison the sum.
        n_a2_1=n_a2(1);
        loIdx_1=a2primeIndex(1,:)';
        loIdx_2=a2primeIndex(2,:)';
        prob_1=reshape(a2primeProbs(1,:),[N_d2,N_a2,N_e]);
        prob_2=reshape(a2primeProbs(2,:),[N_d2,N_a2,N_e]);
        aprime_ll= loIdx_1   +n_a2_1*(loIdx_2-1);
        aprime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        aprime_lh= loIdx_1   +n_a2_1* loIdx_2;
        aprime_hh=(loIdx_1+1)+n_a2_1* loIdx_2;
        V_ll=reshape(EVpre(aprime_ll),[N_d2,N_a2,N_e]);
        V_hl=reshape(EVpre(aprime_hl),[N_d2,N_a2,N_e]);
        V_lh=reshape(EVpre(aprime_lh),[N_d2,N_a2,N_e]);
        V_hh=reshape(EVpre(aprime_hh),[N_d2,N_a2,N_e]);
        % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
        p1_loy=prob_1; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        % outer level: interpolate those two over the a2_2 dimension
        p2=prob_2; p2(EV_loy==EV_hiy)=0;
        c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_d2, n_a2, n_e, d2_gridvals, a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_d2, n_a2, special_n_e, d2_gridvals, a2_gridvals, e_gridvals_J(e_c,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

            entireRHS=ReturnMatrix_e+DiscountFactorParamsVec*EV(:,:,e_c);

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
