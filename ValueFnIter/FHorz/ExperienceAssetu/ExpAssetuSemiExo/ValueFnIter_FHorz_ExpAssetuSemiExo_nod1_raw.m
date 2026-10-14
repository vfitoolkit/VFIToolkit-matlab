function [V,Policy3]=ValueFnIter_FHorz_ExpAssetuSemiExo_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, u_gridvals, pi_z_J, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% z is exogenous state, semiz is semi-exog state

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_u=prod(n_u);

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,a1prime seperately
Policy3=zeros(3,N_a,N_semiz*N_z,N_j,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put it into third dimension

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% For the return function we just want (I'm just guessing that as I need them N_j times it will be fractionally faster to put them together now)
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_semiz=[n_semiz,ones(1,length(n_z))]; % semiz vectorised, z scalar (lowmemory=1 split over z)
end

% Preallocate
V_ford3_jj=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,n_bothz, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % [N_d*N_a1,N_a1*N_a2,N_z]; Level=0, Refine=0
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1; % Do I need this shiftdim(), can probably delete all these
        Policy3(1,:,:,N_j)=rem(d_ind-1,N_d2)+1;
        Policy3(2,:,:,N_j)=ceil(d_ind/N_d2);
        Policy3(3,:,:,N_j)=ceil(maxindex/N_d23);

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for z_c=1:N_z
            zind=(1:1:N_semiz)+N_semiz*(z_c-1);
            z_val=bothz_gridvals_J(zind,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,special_n_semiz, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,zind,N_j)=shiftdim(Vtemp,1);
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy3(1,:,zind,N_j)=rem(d_ind-1,N_d2)+1;
            Policy3(2,:,zind,N_j)=ceil(d_ind/N_d2);
            Policy3(3,:,zind,N_j)=ceil(maxindex/N_d23);
        end
    elseif vfoptions.lowmemory==2
        % joint: loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy3(1,:,z_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy3(2,:,z_c,N_j)=ceil(d_ind/N_d2);
            Policy3(3,:,z_c,N_j)=ceil(maxindex/N_d23);
        end
    end
else
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
    else
        aprime_ll=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_ll-1),N_a1,1);
        aprime_hl=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_hl-1),N_a1,1);
        aprime_lh=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_lh-1),N_a1,1);
        aprime_hh=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_hh-1),N_a1,1);
        prob_1=pr_1;
        prob_2=pr_2;
    end
    if vfoptions.lowmemory==0
        if length(n_a2)==1
            aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_bothz);  % [N_d2*N_a1,N_a2,N_u,N_bothz]
        else
            prob_1=repmat(pr_1,N_a1,1,1,N_bothz);
            prob_2=repmat(pr_2,N_a1,1,1,N_bothz);
        end
    else
        if length(n_a2)==1
            aprimeProbs=repmat(a2primeProbs,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
        else
            prob_1=repmat(pr_1,N_a1,1);
            prob_2=repmat(pr_2,N_a1,1);
        end
    end

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz_d3=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % Level=0, Refine=0
            % (d,aprime,a,z)

            EV=EVpre.*shiftdim(pi_bothz_d3',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs; % fresh per d3: skipinterp varies with d3_c, so the zeroing must not accumulate
                aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
                % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
                % does not poison the sum.
                V_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
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
            EV=squeeze(sum(EV,3)); % (d2,a1prime,a2,both)

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);

            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);

        end

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz_d3=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz_d3',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the upper aprime
                aprimeProbs_full=repmat(a2primeProbs,N_a1,1,1,N_bothz);
                skipinterp=(EV1==EV2);
                aprimeProbs_full(skipinterp)=0;
                entireEV=EV1.*aprimeProbs_full+EV2.*(1-aprimeProbs_full);
                entireEV(aprimeProbs_full==0)=EV2(aprimeProbs_full==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_full==1)=EV1(aprimeProbs_full==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
                % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
                % does not poison the sum.
                V_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                prob_1=repmat(pr_1,N_a1,1,1,N_bothz);
                prob_2=repmat(pr_2,N_a1,1,1,N_bothz);
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
                entireEV=c_lo+c_hi;
            end
            entireEV=entireEV.*pi_u; entireEV(isnan(entireEV))=0; % a zero pi_u against an infinite node gives 0*(-Inf)=NaN, so zero the term BEFORE summing
            entireEV=squeeze(sum(entireEV,3)); % integrate out u -> (d2*a1prime,a2,bothz)

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                entireEV_z=entireEV(:,:,zind);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,1,N_a1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,zind,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,zind,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        % joint: loop over bothz
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz_d3=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=EVpre.*(ones(N_a,1,'gpuArray')*pi_bothz_d3(z_c,:));
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                if length(n_a2)==1
                    EV1=reshape(EV_z(aprimeIndex),[N_d2*N_a1,N_a2,N_u]); % (d2,a1prime,a2), the lower aprime
                    EV2=reshape(EV_z(aprimeplus1Index),[N_d2*N_a1,N_a2,N_u]); % (d2,a1prime,a2), the upper aprime

                    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                    skipinterp=(EV1==EV2);
                    aprimeProbs_d3=aprimeProbs; % fresh per d3: skipinterp varies with d3_c, so the zeroing must not accumulate
                    aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                    % Apply the aprimeProbs
                    EV_z=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                    EV_z(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    EV_z(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
                else
                    % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
                    % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
                    % does not poison the sum.
                    V_ll=reshape(EV_z(aprime_ll),[N_d2*N_a1,N_a2,N_u]);
                    V_hl=reshape(EV_z(aprime_hl),[N_d2*N_a1,N_a2,N_u]);
                    V_lh=reshape(EV_z(aprime_lh),[N_d2*N_a1,N_a2,N_u]);
                    V_hh=reshape(EV_z(aprime_hh),[N_d2*N_a1,N_a2,N_u]);
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
                    EV_z=c_lo+c_hi;
                end
                % Already applied the probabilities from interpolating onto grid
                EV_z=EV_z.*pi_u; EV_z(isnan(EV_z))=0; % a zero pi_u against an infinite node gives 0*(-Inf)=NaN, so zero the term BEFORE summing
                EV_z=sum(EV_z,3); % (d2,a1prime,a2)

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*kron(EV_z,ones(1,N_a1));

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d3
    V(:,:,N_j)=V_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy3(1,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,N_j)=ceil(d2a1prime_ind/N_d2); % a1prime

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
    else
        aprime_ll=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_ll-1),N_a1,1);
        aprime_hl=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_hl-1),N_a1,1);
        aprime_lh=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_lh-1),N_a1,1);
        aprime_hh=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2k_hh-1),N_a1,1);
        prob_1=pr_1;
        prob_2=pr_2;
    end
    if vfoptions.lowmemory==0
        if length(n_a2)==1
            aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_bothz);  % [N_d2*N_a1,N_a2,N_u,N_bothz]
        else
            prob_1=repmat(pr_1,N_a1,1,1,N_bothz);
            prob_2=repmat(pr_2,N_a1,1,1,N_bothz);
        end
    else
        if length(n_a2)==1
            aprimeProbs=repmat(a2primeProbs,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
        else
            prob_1=repmat(pr_1,N_a1,1);
            prob_2=repmat(pr_2,N_a1,1);
        end
    end

    EVpre=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz_d3=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0); % Level=0, Refine=0
            % (d,aprime,a,z)

            EV=EVpre.*shiftdim(pi_bothz_d3',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs; % fresh per d3: skipinterp varies with d3_c, so the zeroing must not accumulate
                aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
                % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
                % does not poison the sum.
                V_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
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
            EV=squeeze(sum(EV,3)); % (d2,a1prime,a2,bothz)

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz_d3=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz_d3',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the upper aprime
                aprimeProbs_full=repmat(a2primeProbs,N_a1,1,1,N_bothz);
                skipinterp=(EV1==EV2);
                aprimeProbs_full(skipinterp)=0;
                entireEV=EV1.*aprimeProbs_full+EV2.*(1-aprimeProbs_full);
                entireEV(aprimeProbs_full==0)=EV2(aprimeProbs_full==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_full==1)=EV1(aprimeProbs_full==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
                % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
                % does not poison the sum.
                V_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                V_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_u,N_bothz]);
                prob_1=repmat(pr_1,N_a1,1,1,N_bothz);
                prob_2=repmat(pr_2,N_a1,1,1,N_bothz);
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
                entireEV=c_lo+c_hi;
            end
            entireEV=entireEV.*pi_u; entireEV(isnan(entireEV))=0; % a zero pi_u against an infinite node gives 0*(-Inf)=NaN, so zero the term BEFORE summing
            entireEV=squeeze(sum(entireEV,3)); % integrate out u -> (d2*a1prime,a2,bothz)

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                entireEV_z=entireEV(:,:,zind);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                entireRHS_z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,1,N_a1,1);
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V_ford3_jj(:,zind,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,zind,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        % joint: loop over bothz
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz_d3=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d3_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=EVpre.*shiftdim(pi_bothz_d3(z_c,:)',-1);
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                if length(n_a2)==1
                    EV1=reshape(EV_z(aprimeIndex),[N_d2*N_a1,N_a2,N_u]); % (d2,a1prime,a2), the lower aprime
                    EV2=reshape(EV_z(aprimeplus1Index),[N_d2*N_a1,N_a2,N_u]); % (d2,a1prime,a2), the upper aprime

                    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                    skipinterp=(EV1==EV2);
                    aprimeProbs_d3=aprimeProbs; % fresh per d3: skipinterp varies with d3_c, so the zeroing must not accumulate
                    aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                    % Apply the aprimeProbs
                    EV_z=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                    EV_z(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    EV_z(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
                else
                    % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
                    % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
                    % does not poison the sum.
                    V_ll=reshape(EV_z(aprime_ll),[N_d2*N_a1,N_a2,N_u]);
                    V_hl=reshape(EV_z(aprime_hl),[N_d2*N_a1,N_a2,N_u]);
                    V_lh=reshape(EV_z(aprime_lh),[N_d2*N_a1,N_a2,N_u]);
                    V_hh=reshape(EV_z(aprime_hh),[N_d2*N_a1,N_a2,N_u]);
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
                    EV_z=c_lo+c_hi;
                end
                % Already applied the probabilities from interpolating onto grid
                EV_z=EV_z.*pi_u; EV_z(isnan(EV_z))=0; % a zero pi_u against an infinite node gives 0*(-Inf)=NaN, so zero the term BEFORE summing
                EV_z=sum(EV_z,3); % (d2,a1prime,a2)

                entireRHS_z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(EV_z,1,N_a1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);

                V_ford3_jj(:,z_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,z_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d3
    V(:,:,jj)=V_jj;
    Policy3(2,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy3(1,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,jj)=ceil(d2a1prime_ind/N_d2); % a1prime

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron

end
