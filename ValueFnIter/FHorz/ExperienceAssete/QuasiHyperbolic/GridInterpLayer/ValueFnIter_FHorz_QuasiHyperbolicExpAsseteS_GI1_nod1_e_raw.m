function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_GI1_nod1_e_raw(n_d2,n_a1,n_a2,n_z,n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAssete_GI1_nod1_e_raw.
% ExperienceAssete (aprimeFn depends on e) with grid interpolation layer on a1. GPU only.
%
% Sophisticated: Vhat_j      = max_{d,a1'} u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EVfine_at_optimal_choice
% EVfine is the (undiscounted) interpolated continuation actually added to the layer-2
% RHS, so the a2 lottery is already baked in and the gather needs no lottery handling.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z
PolicyL2flag=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
% n_a1prime=n_a1;
% a1prime_gridvals=a1_gridvals;

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
end

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1
zind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1
zindB=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
zeindB=zindB+N_z*shiftdim((0:1:N_e-1),-2); % already includes -1

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0
        % Calc the max and it's index
        [~,maxindex]=max(ReturnMatrix,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z,n_e, d2_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_z,N_e]; Level=2, Refine=0
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Vhat(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*zeindB; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policy(1,:,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offset      = ceil(maxindexL2/N_d2);
        linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*zeindB;
        linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zeindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
            % Calc the max and it's index
            [~,maxindex]=max(ReturnMatrix_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_z
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z,special_n_e, d2_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_z]; Level=2, Refine=0
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            Vhat(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*zindB; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_z
            Policy(1,:,:,e_c,N_j)=d_ind; % d2
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offset      = ceil(maxindexL2/N_d2);
            linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
            linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
                % Calc the max and it's index
                [~,maxindex]=max(ReturnMatrix_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_z]; Level=2, Refine=0
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                Vhat(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
                Policy(1,:,z_c,e_c,N_j)=d_ind; % d2
                Policy(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
                L2offset      = ceil(maxindexL2/N_d2);
                linidx_lower  = d_ind                   + N_d2*n2long*aind;
                linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
            end
        end

    end

    Vunderbar=Vhat;
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=sum(shiftdim(pi_e_J(:,N_j+1),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3); % Integrate out eprime first

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_e], whereas aprimeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2,N_e]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_e]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_z); % [N_d2*N_a1,N_a2,N_e,N_z]   (replicate over zprime)

    Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_z]); % (d2*a1prime,a2,e_cur,zprime)
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2*a1prime,a2,e_cur,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-3); % pi'(zprime,zcur) shaped [1,1,1,zprime,zcur]
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_d2*N_a1,N_a2,N_e,N_z]); % sum zprime -> (d2*a1prime,a2,e_cur,zcur)
    EV=permute(EV,[1,2,4,3]); % (d2*a1prime,a2,zcur,e_cur)  -- match (z,e) dim order

    entireEV=reshape(EV,[N_d2,N_a1,1,N_a2,N_z,N_e]); % undiscounted; beta/beta0beta applied at use sites
    % Interpolate EV over aprime_grid
    entireEVinterp=permute(interp1(a1_gridvals,permute(entireEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]); % [N_d2,N_a1prime,1,N_a2,N_z,N_e]

    Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        % --- Vhat search (beta0*beta) ---
        entireRHS=ReturnMatrix+beta0beta*entireEV; % autofill 3rd dim to N_a1

        % Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z,n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
        d2a1primea2z=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind+N_d2*N_a1prime*N_a2*N_z*shiftdim((0:1:N_e-1),-4);
        EVfine=reshape(entireEVinterp(d2a1primea2z(:)),[N_d2*n2long,N_a1*N_a2,N_z,N_e]);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vhat(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policy(1,:,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offset      = ceil(maxindexL2/N_d2);
        linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*zeindB;
        linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zeindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        linidx=reshape(maxindexL2,[1,N_a*N_z*N_e])+N_d2*n2long*(0:N_a*N_z*N_e-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_z,N_e]);
        Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            entireEV_e=entireEV(:,:,:,:,:,e_c);
            entireEVinterp_e=entireEVinterp(:,:,:,:,:,e_c);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            % --- Vhat search (beta0*beta) ---
            entireRHS_e=ReturnMatrix_e+beta0beta*entireEV_e; % autofill 3rd dim to N_a1

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
            d2a1primea2z=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind;
            EVfine_e=reshape(entireEVinterp_e(d2a1primea2z(:)),[N_d2*n2long,N_a1*N_a2,N_z]);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine_e;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            Vhat(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
            Policy(1,:,:,e_c,N_j)=d_ind; % d2
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offset      = ceil(maxindexL2/N_d2);
            linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
            linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
            linidx_e=reshape(maxindexL2,[1,N_a*N_z])+N_d2*n2long*(0:N_a*N_z-1);
            EV_at_policy_e=reshape(EVfine_e(linidx_e),[N_a,N_z]);
            Vunderbar(:,:,e_c,N_j)=Vhat(:,:,e_c,N_j)+(beta-beta0beta)*EV_at_policy_e;
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                entireEV_ze=entireEV(:,:,:,:,z_c,e_c);
                entireEVinterp_ze=entireEVinterp(:,:,:,:,z_c,e_c);
                ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                % --- Vhat search (beta0*beta) ---
                entireRHS=ReturnMatrix+beta0beta*entireEV_ze; % autofill 3rd dim to N_a1

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
                d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                EVfine_ze=reshape(entireEVinterp_ze(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine_ze;
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                Vhat(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
                Policy(1,:,z_c,e_c,N_j)=d_ind; % d2
                Policy(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
                L2offset      = ceil(maxindexL2/N_d2);
                linidx_lower  = d_ind                   + N_d2*n2long*aind;
                linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
                linidx_ze=reshape(maxindexL2,[1,N_a])+N_d2*n2long*(0:N_a-1);
                EV_at_policy_ze=reshape(EVfine_ze(linidx_ze),[N_a,1]);
                Vunderbar(:,z_c,e_c,N_j)=Vhat(:,z_c,e_c,N_j)+(beta-beta0beta)*EV_at_policy_ze;
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
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_e], whereas aprimeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2,N_e]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_e]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_z); % [N_d2*N_a1,N_a2,N_e,N_z]   (replicate over zprime)

    EVpre=sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3); % Integrate out eprime

    Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_z]); % (d2*a1prime,a2,e_cur,zprime)
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2*a1prime,a2,e_cur,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-3); % pi'(zprime,zcur) shaped [1,1,1,zprime,zcur]
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_d2*N_a1,N_a2,N_e,N_z]); % sum zprime -> (d2*a1prime,a2,e_cur,zcur)
    EV=permute(EV,[1,2,4,3]); % (d2*a1prime,a2,zcur,e_cur)  -- match (z,e) dim order

    entireEV=reshape(EV,[N_d2,N_a1,1,N_a2,N_z,N_e]); % undiscounted; beta/beta0beta applied at use sites
    % Interpolate EV over aprime_grid
    entireEVinterp=permute(interp1(a1_gridvals,permute(entireEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]); % [N_d2,N_a1prime,1,N_a2,N_z,N_e]

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        % --- Vhat search (beta0*beta) ---
        entireRHS=ReturnMatrix+beta0beta*entireEV; % autofill 3rd dim to N_a1

        % Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z,n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
        d2a1primea2z=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind+N_d2*N_a1prime*N_a2*N_z*shiftdim((0:1:N_e-1),-4);
        EVfine=reshape(entireEVinterp(d2a1primea2z(:)),[N_d2*n2long,N_a1*N_a2,N_z,N_e]);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vhat(:,:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policy(1,:,:,:,jj)=d_ind; % d2
        Policy(2,:,:,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,:,jj)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offset      = ceil(maxindexL2/N_d2);
        linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*zeindB;
        linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zeindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,:,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        linidx=reshape(maxindexL2,[1,N_a*N_z*N_e])+N_d2*n2long*(0:N_a*N_z*N_e-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_z,N_e]);
        Vunderbar(:,:,:,jj)=Vhat(:,:,:,jj)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            entireEV_e=entireEV(:,:,:,:,:,e_c);
            entireEVinterp_e=entireEVinterp(:,:,:,:,:,e_c);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            % --- Vhat search (beta0*beta) ---
            entireRHS_e=ReturnMatrix_e+beta0beta*entireEV_e; % autofill 3rd dim to N_a1

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
            d2a1primea2z=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind;
            EVfine_e=reshape(entireEVinterp_e(d2a1primea2z(:)),[N_d2*n2long,N_a1*N_a2,N_z]);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine_e;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            Vhat(:,:,e_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
            Policy(1,:,:,e_c,jj)=d_ind; % d2
            Policy(2,:,:,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offset      = ceil(maxindexL2/N_d2);
            linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
            linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,:,e_c,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
            linidx_e=reshape(maxindexL2,[1,N_a*N_z])+N_d2*n2long*(0:N_a*N_z-1);
            EV_at_policy_e=reshape(EVfine_e(linidx_e),[N_a,N_z]);
            Vunderbar(:,:,e_c,jj)=Vhat(:,:,e_c,jj)+(beta-beta0beta)*EV_at_policy_e;
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                entireEV_ze=entireEV(:,:,:,:,z_c,e_c);
                entireEVinterp_ze=entireEVinterp(:,:,:,:,z_c,e_c);
                ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                % --- Vhat search (beta0*beta) ---
                entireRHS=ReturnMatrix+beta0beta*entireEV_ze; % autofill 3rd dim to N_a1

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
                d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                EVfine_ze=reshape(entireEVinterp_ze(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine_ze;
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                Vhat(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
                Policy(1,:,z_c,e_c,jj)=d_ind; % d2
                Policy(2,:,z_c,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy(3,:,z_c,e_c,jj)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
                L2offset      = ceil(maxindexL2/N_d2);
                linidx_lower  = d_ind                   + N_d2*n2long*aind;
                linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
                linidx_ze=reshape(maxindexL2,[1,N_a])+N_d2*n2long*(0:N_a-1);
                EV_at_policy_ze=reshape(EVfine_ze(linidx_ze),[N_a,1]);
                Vunderbar(:,z_c,e_c,jj)=Vhat(:,z_c,e_c,jj)+(beta-beta0beta)*EV_at_policy_ze;
            end
        end
    end
end



%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy;PolicyL2flag];



end
