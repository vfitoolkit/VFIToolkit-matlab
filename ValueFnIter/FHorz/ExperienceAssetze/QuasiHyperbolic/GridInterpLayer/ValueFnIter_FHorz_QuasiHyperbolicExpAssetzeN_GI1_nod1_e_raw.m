function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_GI1_nod1_e_raw(n_d2,n_a1,n_a2,n_z,n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAssetze_GI1_nod1_e_raw.
% ExperienceAssetze with grid interpolation layer on a1. GPU only.
% Note: in this family aprimeFn(d2,a2,z,e) depends on BOTH the markov z and the iid e;
% both are integrated out when EV is formed (a2 lottery, then pi_z over zprime, with
% pi_e applied to V_{j+1} in EVpre), upstream of both maximisation passes.
% The _e in the filename is a FURTHER iid shock, carried by every variant of this family.
%
% Naive:  Valt_j   = max_{d,a1'} F + beta*E[Valt_{j+1}]         (exponential discounter)
%         Vtilde_j = max_{d,a1'} F + beta_0*beta*E[Valt_{j+1}]  (agent's perceived choice)
% where F is the return function.
% The two discount factors generally pick different GI midpoints, so each pass
% re-derives its own midpoint, its own layer-2 return matrix, and its own L2 flag.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

Valt=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z,e
Policy(4,:,:,:,:)=2; % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
Policyalt=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); % exponential discounter optimal choice
Policyalt(4,:,:,:,:)=2;

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
        Valt(:,:,:,N_j)=shiftdim(Vtempii,1);
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
        Policy(4,:,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
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
            Valt(:,:,e_c,N_j)=shiftdim(Vtempii,1);
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
            Policy(4,:,:,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
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
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                Valt(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
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
                Policy(4,:,z_c,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
            end
        end
    end

    Vtilde=Valt;
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j); % terminal: QH and exp discounter coincide
    Policyalt(4,:,:,:,N_j)=Policy(4,:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=sum(shiftdim(pi_e_J(:,N_j+1),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3); % integrate out eprime first

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % l_a2==1: a2primeIndex/a2primeProbs are [N_d2,N_a2,N_z,N_e] (legacy lower-corner)
    % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2,N_z,N_e] (per-dim factored)

    if length(n_a2)==1
        aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat((a2primeIndex-1),N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_z,N_e]
        aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_z,N_e]
        aprimeProbs=repmat(a2primeProbs,N_a1,1,1,1,N_z); % [N_d2*N_a1,N_a2,N_z,N_e,N_z]   (replicate over zprime)

        Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;

        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % l_a2==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:,:,:),[N_d2,N_a2,N_z,N_e]);
        loIdx_2=reshape(a2primeIndex(2,:,:,:,:),[N_d2,N_a2,N_z,N_e]);
        prob_1_exp=repmat(reshape(a2primeProbs(1,:,:,:,:),[N_d2,N_a2,N_z,N_e]),N_a1,1,1,1,N_z);
        prob_2_exp=repmat(reshape(a2primeProbs(2,:,:,:,:),[N_d2,N_a2,N_z,N_e]),N_a1,1,1,1,N_z);

        a1prime_offsets=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z,N_e);
        aprime_ll=a1prime_offsets+N_a1*repmat( loIdx_1   +n_a2_1*(loIdx_2-1)-1,N_a1,1,1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1,1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat( loIdx_1   +n_a2_1* loIdx_2   -1,N_a1,1,1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1* loIdx_2   -1,N_a1,1,1,1);
        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end

    EV=EV.*reshape(pi_z_J(:,:,N_j),[1,1,N_z,1,N_z]); % pi[z_cur,z_prime] reshaped to broadcast: z_cur at dim 3, z_prime at dim 5
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,5),[N_d2*N_a1,N_a2,N_z,N_e]); % sum zprime -> (d2*a1prime,a2,z_cur,e_cur)

    entireEV=reshape(EV,[N_d2,N_a1,1,N_a2,N_z,N_e]); % (d2,a1prime,1,a2,z,e) -- undiscounted; beta/beta0beta applied at use sites
    % Interpolate EV over aprime_grid
    entireEVinterp=permute(interp1(a1_gridvals,permute(entireEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]); % [N_d2,N_a1prime,1,N_a2,N_z,N_e]

    Vtilde=zeros(N_a,N_z,N_e,N_j,'gpuArray');

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHSalt=ReturnMatrix+beta*entireEV; % autofill 3rd dim to N_a1

        % Calc the max and it's index
        [~,maxindexalt]=max(entireRHSalt,[],2);

        % Turn this into the 'midpoint'
        midpointalt=max(min(maxindexalt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z,n_e, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
        d2a1primea2zalt=(1:1:N_d2)'+N_d2*(a1primeindexesfinealt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind+N_d2*N_a1prime*N_a2*N_z*shiftdim((0:1:N_e-1),-4);
        entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(entireEVinterp(d2a1primea2zalt(:)),[N_d2*n2long,N_a1*N_a2,N_z,N_e]);
        [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
        Valt(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_indalt=rem(maxindexL2alt-1,N_d2)+1;
        allindalt=d_indalt+N_d2*aind+N_d2*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policyalt(1,:,:,:,N_j)=d_indalt; % d2
        Policyalt(2,:,:,:,N_j)=shiftdim(squeeze(midpointalt(allindalt)),-1); % a1prime midpoint
        Policyalt(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offsetalt      = ceil(maxindexL2alt/N_d2);
        linidx_loweralt  = d_indalt                   + N_d2*n2long*aind + N_d2*n2long*N_a*zeindB;
        linidx_upperalt  = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zeindB;
        isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
        inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
        inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
        Policyalt(4,:,:,:,N_j) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), -1);
        %% Vtilde (beta0*beta)
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
        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(entireEVinterp(d2a1primea2z(:)),[N_d2*n2long,N_a1*N_a2,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,:,:,N_j)=shiftdim(Vtempii,1);
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
        Policy(4,:,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            entireEV_e=entireEV(:,:,:,:,:,e_c);
            entireEVinterp_e=entireEVinterp(:,:,:,:,:,e_c);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS_ealt=ReturnMatrix_e+beta*entireEV_e; % autofill 3rd dim to N_a1

            % Calc the max and it's index
            [~,maxindexalt]=max(entireRHS_ealt,[],2);

            % Turn this into the 'midpoint'
            midpointalt=max(min(maxindexalt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
            d2a1primea2zalt=(1:1:N_d2)'+N_d2*(a1primeindexesfinealt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(entireEVinterp_e(d2a1primea2zalt(:)),[N_d2*n2long,N_a1*N_a2,N_z]);
            [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
            Valt(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_indalt=rem(maxindexL2alt-1,N_d2)+1;
            allindalt=d_indalt+N_d2*aind+N_d2*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
            Policyalt(1,:,:,e_c,N_j)=d_indalt; % d2
            Policyalt(2,:,:,e_c,N_j)=shiftdim(squeeze(midpointalt(allindalt)),-1); % a1prime midpoint
            Policyalt(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offsetalt      = ceil(maxindexL2alt/N_d2);
            linidx_loweralt  = d_indalt                   + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
            linidx_upperalt  = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
            isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
            inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
            inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
            Policyalt(4,:,:,e_c,N_j) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), -1);
            %% Vtilde (beta0*beta)
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
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(entireEVinterp_e(d2a1primea2z(:)),[N_d2*n2long,N_a1*N_a2,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            Vtilde(:,:,e_c,N_j)=shiftdim(Vtempii,1);
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
            Policy(4,:,:,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                entireEV_ze=entireEV(:,:,:,:,z_c,e_c);
                entireEVinterp_ze=entireEVinterp(:,:,:,:,z_c,e_c);
                ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
                entireRHSalt=ReturnMatrix+beta*entireEV_ze; % autofill 3rd dim to N_a1

                % Calc the max and it's index
                [~,maxindexalt]=max(entireRHSalt,[],2);

                % Turn this into the 'midpoint'
                midpointalt=max(min(maxindexalt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
                d2a1primea2alt=(1:1:N_d2)'+N_d2*(a1primeindexesfinealt-1)+N_d2*N_a1prime*a2ind;
                entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(entireEVinterp_ze(d2a1primea2alt(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
                Valt(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_indalt=rem(maxindexL2alt-1,N_d2)+1;
                allindalt=d_indalt+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
                Policyalt(1,:,z_c,e_c,N_j)=d_indalt; % d2
                Policyalt(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpointalt(allindalt)),-1); % a1prime midpoint
                Policyalt(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
                L2offsetalt      = ceil(maxindexL2alt/N_d2);
                linidx_loweralt  = d_indalt                   + N_d2*n2long*aind;
                linidx_upperalt  = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind;
                isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
                isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
                inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
                inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
                Policyalt(4,:,z_c,e_c,N_j) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), -1);
                %% Vtilde (beta0*beta)
                entireRHS=ReturnMatrix+beta0beta*entireEV_ze; % autofill 3rd dim to N_a1

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
                d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(entireEVinterp_ze(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                Vtilde(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
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
                Policy(4,:,z_c,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
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
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % l_a2==1: a2primeIndex/a2primeProbs are [N_d2,N_a2,N_z,N_e] (legacy lower-corner)
    % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2,N_z,N_e] (per-dim factored)

    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3); % integrate out eprime

    if length(n_a2)==1
        aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat((a2primeIndex-1),N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_z,N_e]
        aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_z,N_e]
        aprimeProbs=repmat(a2primeProbs,N_a1,1,1,1,N_z); % [N_d2*N_a1,N_a2,N_z,N_e,N_z]   (replicate over zprime)

        Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;

        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % l_a2==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:,:,:),[N_d2,N_a2,N_z,N_e]);
        loIdx_2=reshape(a2primeIndex(2,:,:,:,:),[N_d2,N_a2,N_z,N_e]);
        prob_1_exp=repmat(reshape(a2primeProbs(1,:,:,:,:),[N_d2,N_a2,N_z,N_e]),N_a1,1,1,1,N_z);
        prob_2_exp=repmat(reshape(a2primeProbs(2,:,:,:,:),[N_d2,N_a2,N_z,N_e]),N_a1,1,1,1,N_z);

        a1prime_offsets=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z,N_e);
        aprime_ll=a1prime_offsets+N_a1*repmat( loIdx_1   +n_a2_1*(loIdx_2-1)-1,N_a1,1,1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1,1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat( loIdx_1   +n_a2_1* loIdx_2   -1,N_a1,1,1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1* loIdx_2   -1,N_a1,1,1,1);
        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end

    EV=EV.*reshape(pi_z_J(:,:,jj),[1,1,N_z,1,N_z]); % pi[z_cur,z_prime] reshaped to broadcast: z_cur at dim 3, z_prime at dim 5
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,5),[N_d2*N_a1,N_a2,N_z,N_e]); % sum zprime -> (d2*a1prime,a2,z_cur,e_cur)

    entireEV=reshape(EV,[N_d2,N_a1,1,N_a2,N_z,N_e]); % (d2,a1prime,1,a2,z,e) -- undiscounted; beta/beta0beta applied at use sites
    % Interpolate EV over aprime_grid
    entireEVinterp=permute(interp1(a1_gridvals,permute(entireEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]); % [N_d2,N_a1prime,1,N_a2,N_z,N_e]

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHSalt=ReturnMatrix+beta*entireEV; % autofill 3rd dim to N_a1

        % Calc the max and it's index
        [~,maxindexalt]=max(entireRHSalt,[],2);

        % Turn this into the 'midpoint'
        midpointalt=max(min(maxindexalt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z,n_e, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
        d2a1primea2zalt=(1:1:N_d2)'+N_d2*(a1primeindexesfinealt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind+N_d2*N_a1prime*N_a2*N_z*shiftdim((0:1:N_e-1),-4);
        entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(entireEVinterp(d2a1primea2zalt(:)),[N_d2*n2long,N_a1*N_a2,N_z,N_e]);
        [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
        Valt(:,:,:,jj)=shiftdim(Vtempii,1);
        d_indalt=rem(maxindexL2alt-1,N_d2)+1;
        allindalt=d_indalt+N_d2*aind+N_d2*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policyalt(1,:,:,:,jj)=d_indalt; % d2
        Policyalt(2,:,:,:,jj)=shiftdim(squeeze(midpointalt(allindalt)),-1); % a1prime midpoint
        Policyalt(3,:,:,:,jj)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offsetalt      = ceil(maxindexL2alt/N_d2);
        linidx_loweralt  = d_indalt                   + N_d2*n2long*aind + N_d2*n2long*N_a*zeindB;
        linidx_upperalt  = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zeindB;
        isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
        inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
        inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
        Policyalt(4,:,:,:,jj) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), -1);
        %% Vtilde (beta0*beta)
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
        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(entireEVinterp(d2a1primea2z(:)),[N_d2*n2long,N_a1*N_a2,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,:,:,jj)=shiftdim(Vtempii,1);
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
        Policy(4,:,:,:,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            entireEV_e=entireEV(:,:,:,:,:,e_c);
            entireEVinterp_e=entireEVinterp(:,:,:,:,:,e_c);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS_ealt=ReturnMatrix_e+beta*entireEV_e; % autofill 3rd dim to N_a1

            % Calc the max and it's index
            [~,maxindexalt]=max(entireRHS_ealt,[],2);

            % Turn this into the 'midpoint'
            midpointalt=max(min(maxindexalt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
            d2a1primea2zalt=(1:1:N_d2)'+N_d2*(a1primeindexesfinealt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(entireEVinterp_e(d2a1primea2zalt(:)),[N_d2*n2long,N_a1*N_a2,N_z]);
            [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
            Valt(:,:,e_c,jj)=shiftdim(Vtempii,1);
            d_indalt=rem(maxindexL2alt-1,N_d2)+1;
            allindalt=d_indalt+N_d2*aind+N_d2*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
            Policyalt(1,:,:,e_c,jj)=d_indalt; % d2
            Policyalt(2,:,:,e_c,jj)=shiftdim(squeeze(midpointalt(allindalt)),-1); % a1prime midpoint
            Policyalt(3,:,:,e_c,jj)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offsetalt      = ceil(maxindexL2alt/N_d2);
            linidx_loweralt  = d_indalt                   + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
            linidx_upperalt  = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
            isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
            inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
            inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
            Policyalt(4,:,:,e_c,jj) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), -1);
            %% Vtilde (beta0*beta)
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
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(entireEVinterp_e(d2a1primea2z(:)),[N_d2*n2long,N_a1*N_a2,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            Vtilde(:,:,e_c,jj)=shiftdim(Vtempii,1);
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
            Policy(4,:,:,e_c,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                entireEV_ze=entireEV(:,:,:,:,z_c,e_c);
                entireEVinterp_ze=entireEVinterp(:,:,:,:,z_c,e_c);
                ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
                entireRHSalt=ReturnMatrix+beta*entireEV_ze; % autofill 3rd dim to N_a1

                % Calc the max and it's index
                [~,maxindexalt]=max(entireRHSalt,[],2);

                % Turn this into the 'midpoint'
                midpointalt=max(min(maxindexalt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
                d2a1primea2alt=(1:1:N_d2)'+N_d2*(a1primeindexesfinealt-1)+N_d2*N_a1prime*a2ind;
                entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(entireEVinterp_ze(d2a1primea2alt(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
                Valt(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                d_indalt=rem(maxindexL2alt-1,N_d2)+1;
                allindalt=d_indalt+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
                Policyalt(1,:,z_c,e_c,jj)=d_indalt; % d2
                Policyalt(2,:,z_c,e_c,jj)=shiftdim(squeeze(midpointalt(allindalt)),-1); % a1prime midpoint
                Policyalt(3,:,z_c,e_c,jj)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
                L2offsetalt      = ceil(maxindexL2alt/N_d2);
                linidx_loweralt  = d_indalt                   + N_d2*n2long*aind;
                linidx_upperalt  = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind;
                isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
                isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
                inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
                inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
                Policyalt(4,:,z_c,e_c,jj) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), -1);
                %% Vtilde (beta0*beta)
                entireRHS=ReturnMatrix+beta0beta*entireEV_ze; % autofill 3rd dim to N_a1

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
                d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(entireEVinterp_ze(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                Vtilde(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
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
                Policy(4,:,z_c,e_c,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
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
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

adjustalt=(Policyalt(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policyalt(2,:,:,:,:)=Policyalt(2,:,:,:,:)-adjustalt; % lower grid point
Policyalt(3,:,:,:,:)=Policyalt(3,:,:,:,:)-(n2short+1)*(~adjustalt); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)



end
