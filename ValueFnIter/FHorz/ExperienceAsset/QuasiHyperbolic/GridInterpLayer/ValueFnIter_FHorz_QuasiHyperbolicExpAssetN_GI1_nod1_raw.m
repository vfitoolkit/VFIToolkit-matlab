function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetN_GI1_nod1_raw(n_d2,n_a1,n_a2,n_z,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAsset_GI1_nod1_raw.
% ExperienceAsset with grid interpolation layer on a1. GPU only.
%
% Naive:  Valt_j   = max_{d,a1'} u + beta*E[Valt_{j+1}]         (exponential discounter)
%         Vtilde_j = max_{d,a1'} u + beta_0*beta*E[Valt_{j+1}]  (agent's perceived choice)
% The two discount factors generally pick different GI midpoints, so each pass
% re-derives its own midpoint and layer-2 return matrix.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);

Valt=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(3,N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
Policyalt=zeros(3,N_a,N_z,N_j,'gpuArray'); % exponential discounter optimal choice
PolicyL2flagalt=2*ones(1,N_a,N_z,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
% n_a1prime=n_a1;
% a1prime_gridvals=a1_gridvals;

if vfoptions.lowmemory>0
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

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, n_z, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0
        % Calc the max and it's index
        [~,maxindex]=max(ReturnMatrix,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_z
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z, d2_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Valt(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*zindB; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_z
        Policy(1,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offset      = ceil(maxindexL2/N_d2);
        linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
        linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, special_n_z, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
            % Calc the max and it's index
            [~,maxindex]=max(ReturnMatrix_z,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,special_n_z, d2_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            Valt(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
            Policy(1,:,z_c,N_j)=d_ind; % d2
            Policy(2,:,z_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,z_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offset      = ceil(maxindexL2/N_d2);
            linidx_lower  = d_ind                   + N_d2*n2long*aind;
            linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,z_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        end
    end

    Vtilde=Valt;
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j); % terminal: QH and exp discounter coincide
    PolicyL2flagalt(1,:,:,N_j)=PolicyL2flag(1,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]

    Vlower=reshape(EV(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(EV(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=squeeze(sum(EV,3)); % sum over z', leaving a singular third dimension

    entireEV=reshape(EV,[N_d2,N_a1,1,N_a2,N_z]); % undiscounted; beta/beta0beta applied at use sites
    % Interpolate EV over aprime_grid
    entireEVinterp=permute(interp1(a1_gridvals,permute(entireEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_z]

    Vtilde=zeros(N_a,N_z,N_j,'gpuArray');

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, n_z, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHSalt=ReturnMatrix+beta*entireEV; % autofill 3rd dim to N_a1

        % Calc the max and it's index
        [~,maxindexalt]=max(entireRHSalt,[],2);

        % Turn this into the 'midpoint'
        midpointalt=max(min(maxindexalt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
        a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
        ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
        d2a1primea2zalt=(1:1:N_d2)'+N_d2*(a1primeindexesfinealt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind;
        entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(entireEVinterp(d2a1primea2zalt(:)),[N_d2*n2long,N_a1*N_a2,N_z]);
        [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
        Valt(:,:,N_j)=shiftdim(Vtempii,1);
        d_indalt=rem(maxindexL2alt-1,N_d2)+1;
        allindalt=d_indalt+N_d2*aind+N_d2*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
        Policyalt(1,:,:,N_j)=d_indalt; % d2
        Policyalt(2,:,:,N_j)=shiftdim(squeeze(midpointalt(allindalt)),-1); % a1prime midpoint
        Policyalt(3,:,:,N_j)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offsetalt      = ceil(maxindexL2alt/N_d2);
        linidx_loweralt  = d_indalt                   + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
        linidx_upperalt  = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
        isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
        inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
        inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
        PolicyL2flagalt(1,:,:,N_j) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), -1);
        %% Vtilde (beta0*beta)
        entireRHS=ReturnMatrix+beta0beta*entireEV; % autofill 3rd dim to N_a1

        % Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
        d2a1primea2z=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind;
        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(entireEVinterp(d2a1primea2z(:)),[N_d2*n2long,N_a1*N_a2,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
        Policy(1,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offset      = ceil(maxindexL2/N_d2);
        linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
        linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            entireEV_z=entireEV(:,:,:,:,z_c);
            entireEVinterp_z=entireEVinterp(:,:,:,:,z_c);


            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, special_n_z, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS_zalt=ReturnMatrix_z+beta*entireEV_z; % autofill 3rd dim to N_a1

            % Calc the max and it's index
            [~,maxindexalt]=max(entireRHS_zalt,[],2);

            % Turn this into the 'midpoint'
            midpointalt=max(min(maxindexalt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2
            a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
            ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, special_n_z, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            d2a1primea2alt=(1:1:N_d2)'+N_d2*(a1primeindexesfinealt-1)+N_d2*N_a1prime*a2ind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(entireEVinterp_z(d2a1primea2alt(:)),[N_d2*n2long,N_a1*N_a2]);
            [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
            Valt(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_indalt=rem(maxindexL2alt-1,N_d2)+1;
            allindalt=d_indalt+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
            Policyalt(1,:,z_c,N_j)=d_indalt; % d2
            Policyalt(2,:,z_c,N_j)=shiftdim(squeeze(midpointalt(allindalt)),-1); % a1prime midpoint
            Policyalt(3,:,z_c,N_j)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offsetalt      = ceil(maxindexL2alt/N_d2);
            linidx_loweralt  = d_indalt                   + N_d2*n2long*aind;
            linidx_upperalt  = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind;
            isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
            inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
            inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
            PolicyL2flagalt(1,:,z_c,N_j) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), -1);
            %% Vtilde (beta0*beta)
            entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z; % autofill 3rd dim to N_a1

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_z,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, special_n_z, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(entireEVinterp_z(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            Vtilde(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
            Policy(1,:,z_c,N_j)=d_ind; % d2
            Policy(2,:,z_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,z_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offset      = ceil(maxindexL2/N_d2);
            linidx_lower  = d_ind                   + N_d2*n2long*aind;
            linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,z_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]

    Vlower=reshape(Valt(aprimeIndex(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(Valt(aprimeplus1Index(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=squeeze(sum(EV,3)); % sum over z', leaving a singular third dimension

    entireEV=reshape(EV,[N_d2,N_a1,1,N_a2,N_z]); % undiscounted; beta/beta0beta applied at use sites
    % Interpolate EV over aprime_grid
    entireEVinterp=permute(interp1(a1_gridvals,permute(entireEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_z]

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, n_z, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHSalt=ReturnMatrix+beta*entireEV; % autofill 3rd dim to N_a1

        % Calc the max and it's index
        [~,maxindexalt]=max(entireRHSalt,[],2);

        % Turn this into the 'midpoint'
        midpointalt=max(min(maxindexalt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
        a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
        ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
        d2a1primea2zalt=(1:1:N_d2)'+N_d2*(a1primeindexesfinealt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind;
        entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(entireEVinterp(d2a1primea2zalt(:)),[N_d2*n2long,N_a1*N_a2,N_z]);
        [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
        Valt(:,:,jj)=shiftdim(Vtempii,1);
        d_indalt=rem(maxindexL2alt-1,N_d2)+1;
        allindalt=d_indalt+N_d2*aind+N_d2*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
        Policyalt(1,:,:,jj)=d_indalt; % d2
        Policyalt(2,:,:,jj)=shiftdim(squeeze(midpointalt(allindalt)),-1); % a1prime midpoint
        Policyalt(3,:,:,jj)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offsetalt      = ceil(maxindexL2alt/N_d2);
        linidx_loweralt  = d_indalt                   + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
        linidx_upperalt  = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
        isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
        inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
        inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
        PolicyL2flagalt(1,:,:,jj) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), -1);
        %% Vtilde (beta0*beta)
        entireRHS=ReturnMatrix+beta0beta*entireEV; % autofill 3rd dim to N_a1

        % Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n2long, n_a1,n_a2,n_z, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
        d2a1primea2z=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*zind;
        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(entireEVinterp(d2a1primea2z(:)),[N_d2*n2long,N_a1*N_a2,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vtilde(:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
        Policy(1,:,:,jj)=d_ind; % d2
        Policy(2,:,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,jj)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offset      = ceil(maxindexL2/N_d2);
        linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
        linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*zindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            entireEV_z=entireEV(:,:,:,:,z_c);
            entireEVinterp_z=entireEVinterp(:,:,:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, special_n_z, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS_zalt=ReturnMatrix_z+beta*entireEV_z; % autofill 3rd dim to N_a1

            % Calc the max and it's index
            [~,maxindexalt]=max(entireRHS_zalt,[],2);

            % Turn this into the 'midpoint'
            midpointalt=max(min(maxindexalt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2
            a1primeindexesfinealt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
            ReturnMatrix_iialt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, special_n_z, d2_gridvals, a1prime_grid(a1primeindexesfinealt), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            d2a1primea2alt=(1:1:N_d2)'+N_d2*(a1primeindexesfinealt-1)+N_d2*N_a1prime*a2ind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(entireEVinterp_z(d2a1primea2alt(:)),[N_d2*n2long,N_a1*N_a2]);
            [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
            Valt(:,z_c,jj)=shiftdim(Vtempii,1);
            d_indalt=rem(maxindexL2alt-1,N_d2)+1;
            allindalt=d_indalt+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
            Policyalt(1,:,z_c,jj)=d_indalt; % d2
            Policyalt(2,:,z_c,jj)=shiftdim(squeeze(midpointalt(allindalt)),-1); % a1prime midpoint
            Policyalt(3,:,z_c,jj)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offsetalt      = ceil(maxindexL2alt/N_d2);
            linidx_loweralt  = d_indalt                   + N_d2*n2long*aind;
            linidx_upperalt  = d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind;
            isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
            inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
            inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
            PolicyL2flagalt(1,:,z_c,jj) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), -1);
            %% Vtilde (beta0*beta)
            entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z; % autofill 3rd dim to N_a1

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_z,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, special_n_z, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(entireEVinterp_z(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            Vtilde(:,z_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
            Policy(1,:,z_c,jj)=d_ind; % d2
            Policy(2,:,z_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,z_c,jj)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offset      = ceil(maxindexL2/N_d2);
            linidx_lower  = d_ind                   + N_d2*n2long*aind;
            linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,z_c,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        end
    end
end




%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy;PolicyL2flag];

adjustalt=(Policyalt(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policyalt(2,:,:,:)=Policyalt(2,:,:,:)-adjustalt; % lower grid point
Policyalt(3,:,:,:)=adjustalt.*Policyalt(3,:,:,:)+(1-adjustalt).*(Policyalt(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policyalt=[Policyalt;PolicyL2flagalt];

% %% For experience asset, just output Policy as single index and then use Case2 to UnKron
% Policy=shiftdim(Policy(1,:,:,:)+N_d2*(Policy(2,:,:,:)-1)+N_d2*N_a1*(Policy(3,:,:,:)-1)+N_d2*N_a1*(n2short+2)*(PolicyL2flag-1),1);


end
