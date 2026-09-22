function [V,Policy]=ValueFnIter_FHorz_AmbAverse_RiskyAsset_GI1_nod1_e_raw(n_ambiguity, n_d2,n_d3,n_a1,n_a2,n_z,n_e,n_u,N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, e_gridvals_J, u_grid, ambiguity_pi_z_J, ambiguity_pi_e_J, ambiguity_pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Ambiguity aversion: multiple priors over ambiguity_pi_u (the risky return distribution is ambiguous, not known risk), pi_z and pi_e;
% the continuation is the worst case at each expectation stage, with the aprime lottery conditional on the prior.
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No d1. e: iid start-of-period.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);
N_u=prod(n_u);

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_z,N_e,N_j,'gpuArray'); % (1)=d2, (2)=d3, (3)=midpoint, (4)=L2ind
Policy(5,:,:,:,:)=2;
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d3_gridvals=CreateGridvals(n_d3,d3_grid,1);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e),'gpuArray');
end
if vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
end

% Setup for GI
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

% Precompute
aind=gpuArray(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);
zindB=shiftdim(gpuArray(0:1:N_z-1),-1);
zeindB=zindB+N_z*shiftdim((0:1:N_e-1),-2);

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_z,n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

        % Grid interpolation layer
        aprimeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_z,n_e, d3_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d3_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d3_ind+N_d3*aind+N_d3*N_a*zeindB;
        Policy(2,:,:,:,N_j)=d3_ind;                                        % d3
        Policy(3,:,:,:,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);     % midpoint
        Policy(4,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);            % L2ind

        % L2flag
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
        linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(5,:,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

        % d2, which was not in ReturnFn
        Policy(1,:,:,:,N_j)=ones(1,N_a,N_z,N_e,'gpuArray'); % d2 (terminal: d2 doesn't matter, only in expectations)
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
            [~,maxindex]=max(ReturnMatrix_e,[],2);
            midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

            % Grid interpolation layer
            aprimeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_z,special_n_e, d3_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d3_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d3_ind+N_d3*aind+N_d3*N_a*zindB;
            Policy(2,:,:,e_c,N_j)=d3_ind;
            Policy(3,:,:,e_c,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);
            Policy(4,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);

            % L2flag
            L2offset      = ceil(maxindexL2/N_d3);
            linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
            linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(5,:,:,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

            % d2, which was not in ReturnFn
            Policy(1,:,:,e_c,N_j)=ones(1,N_a,N_z,'gpuArray'); % d2 (terminal: d2 doesn't matter, only in expectations)
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % Layer 1: full ReturnMatrix max for initial midpoint
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,special_n_z,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                [~,maxindex]=max(ReturnMatrix_ze,[],2);
                midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

                % Grid interpolation layer
                aprimeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,special_n_z,special_n_e, d3_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d3_ind=rem(maxindexL2-1,N_d3)+1;
                allind=d3_ind+N_d3*aind;
                Policy(2,:,z_c,e_c,N_j)=d3_ind;
                Policy(3,:,z_c,e_c,N_j)=midpoint_jj(allind);
                Policy(4,:,z_c,e_c,N_j)=ceil(maxindexL2/N_d3);

                % L2flag
                L2offset      = ceil(maxindexL2/N_d3);
                linidx_lower  = d3_ind                   + N_d3*n2long*aind;
                linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(5,:,z_c,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                % d2, which was not in ReturnFn
                Policy(1,:,z_c,e_c,N_j)=ones(1,N_a,'gpuArray'); % d2 (terminal: d2 doesn't matter, only in expectations)
            end
        end
    end
else % V_Jplus1

    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));
    EVpree=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);
    for amb_ce=1:n_ambiguity(N_j) % evaluate the iid-e expectation under each of the multiple priors (running worst case)
        EVe=sum(EVpree.*shiftdim(ambiguity_pi_e_J(:,N_j+1,amb_ce),-2),3);
        EVe(isnan(EVe))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        if amb_ce==1
            EVnext=EVe;
        else
            EVnext=min(EVnext,EVe);
        end
    end % EVnext is now the worst case over the e-priors; the z expectation is next

    % Build a2primeIndex and a2primeProbs for RisykAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Get EV in terms of next period endogenous states
    ambEVstack=[]; % one slice per z-prior (the aprime lottery below is conditional on the prior)
    for amb_c0=1:n_ambiguity(N_j)
        EV=EVnext.*shiftdim(ambiguity_pi_z_J(:,:,N_j,amb_c0)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        ambEVstack=cat(4,ambEVstack,EV);
    end

    % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
    % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
    for amb_c=1:n_ambiguity(N_j)
        EV=ambEVstack(:,:,:,amb_c);
        a2primeProbsK=a2primeProbs;
        EV=reshape(EV,[N_a,N_z]);

        % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
        skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
        aprimeProbsK=repmat(a2primeProbsK,N_a1,N_z);
        aprimeProbsK(skipinterp)=0;
        aprimeProbsK=reshape(aprimeProbsK,[N_d23*N_a1,N_u,N_z]);
        % Take the expectation over the between period iid u shock
        EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbsK;
        EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbsK);
        EV1(isnan(EV1))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so the term contributes nothing
        EV2(isnan(EV2))=0;
        if amb_c==1
            Mmin=EV1+EV2; EV1sel=EV1; EV2sel=EV2;
        else
            Mk=EV1+EV2;
            newmin=(Mk<Mmin);
            Mmin(newmin)=Mk(newmin);
            EV1sel(newmin)=EV1(newmin);
            EV2sel(newmin)=EV2(newmin);
        end
    end
    % Worst case over the u-priors (the ambiguous risky return distribution)
    EV=sum(EV1sel.*ambiguity_pi_u(:,1)',2)+sum(EV2sel.*ambiguity_pi_u(:,1)',2);
    for amb_cu=2:n_ambiguity(N_j)
        EV=min(EV,sum(EV1sel.*ambiguity_pi_u(:,amb_cu)',2)+sum(EV2sel.*ambiguity_pi_u(:,amb_cu)',2));
    end
    EV=reshape(EV,[N_d23*N_a1,N_z]);

    % Refine d2 out of EV before combining with ReturnFn
    EVres=reshape(EV,[N_d2,N_d3*N_a1,N_z]);
    [EV_onlyd3,d2index]=max(EVres,[],1);
    EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_z]);
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d3,N_a1prime,1,1,N_z]

    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_z,n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        entireRHS=ReturnMatrix+DiscountedEV;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

        % Grid interpolation layer
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_z,n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
        da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d3_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d3_ind+N_d3*aind+N_d3*N_a*zeindB;
        Policy(2,:,:,:,N_j)=d3_ind;
        Policy(3,:,:,:,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);
        Policy(4,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);

        % L2flag
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
        linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(5,:,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

        % Get the d2Policy
        d3opt=d3_ind;
        a1opt_mid=midpoint_jj(allind);
        zlin=shiftdim(gpuArray(0:N_z-1),-1);
        lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
        Policy(1,:,:,:,N_j)=d2index_resh(lin);

    elseif vfoptions.lowmemory>=1
        special_n_e=ones(1,length(n_e));
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
            entireRHS_e=ReturnMatrix_e+DiscountedEV;
            [~,maxindex]=max(entireRHS_e,[],2);
            midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

            % Grid interpolation layer
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_z,special_n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
            da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d3_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d3_ind+N_d3*aind+N_d3*N_a*zindB;
            Policy(2,:,:,e_c,N_j)=d3_ind;
            Policy(3,:,:,e_c,N_j)=shiftdim(squeeze(midpoint_jj(allind)),-1);
            Policy(4,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d3),-1);

            % L2flag
            L2offset      = ceil(maxindexL2/N_d3);
            linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
            linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(5,:,:,e_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

            % Get the d2Policy
            d3opt=d3_ind;
            a1opt_mid=midpoint_jj(allind);
            zlin=shiftdim(gpuArray(0:N_z-1),-1);
            lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
            Policy(1,:,:,e_c,N_j)=d2index_resh(lin);
        end
    end
end

%% Iterate backwards
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj));

    % Build a2primeIndex and a2primeProbs for RisykAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Get EV in terms of next period endogenous states
    EVpree=V(:,:,:,jj+1); % iid-e expectation of V is taken first
    for amb_ce=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors (running worst case)
        EVe=sum(EVpree.*shiftdim(ambiguity_pi_e_J(:,jj+1,amb_ce),-2),3);
        EVe(isnan(EVe))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        if amb_ce==1
            EVnext=EVe;
        else
            EVnext=min(EVnext,EVe);
        end
    end % EVnext is now the worst case over the e-priors; the z expectation is next
    ambEVstack=[]; % one slice per z-prior (the aprime lottery below is conditional on the prior)
    for amb_c0=1:n_ambiguity(jj)
        EV=EVnext.*shiftdim(ambiguity_pi_z_J(:,:,jj,amb_c0)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        ambEVstack=cat(4,ambEVstack,EV);
    end

    % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
    % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
    for amb_c=1:n_ambiguity(jj)
        EV=ambEVstack(:,:,:,amb_c);
        a2primeProbsK=a2primeProbs;
        EV=reshape(EV,[N_a,N_z]);

        % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
        skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
        aprimeProbsK=repmat(a2primeProbsK,N_a1,N_z);
        aprimeProbsK(skipinterp)=0;
        aprimeProbsK=reshape(aprimeProbsK,[N_d23*N_a1,N_u,N_z]);
        % Take the expectation over the between period iid u shock
        EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbsK;
        EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbsK);
        EV1(isnan(EV1))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so the term contributes nothing
        EV2(isnan(EV2))=0;
        if amb_c==1
            Mmin=EV1+EV2; EV1sel=EV1; EV2sel=EV2;
        else
            Mk=EV1+EV2;
            newmin=(Mk<Mmin);
            Mmin(newmin)=Mk(newmin);
            EV1sel(newmin)=EV1(newmin);
            EV2sel(newmin)=EV2(newmin);
        end
    end
    % Worst case over the u-priors (the ambiguous risky return distribution)
    EV=sum(EV1sel.*ambiguity_pi_u(:,1)',2)+sum(EV2sel.*ambiguity_pi_u(:,1)',2);
    for amb_cu=2:n_ambiguity(jj)
        EV=min(EV,sum(EV1sel.*ambiguity_pi_u(:,amb_cu)',2)+sum(EV2sel.*ambiguity_pi_u(:,amb_cu)',2));
    end
    EV=reshape(EV,[N_d23*N_a1,N_z]);

    % Refine d2 out of EV before combining with ReturnFn
    EVres=reshape(EV,[N_d2,N_d3*N_a1,N_z]);
    [EV_onlyd3,d2index]=max(EVres,[],1);
    EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_z]);
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d3,N_a1prime,1,1,N_z]

    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_z,n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
        entireRHS=ReturnMatrix+DiscountedEV;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

        % Grid interpolation layer
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_z,n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
        da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,jj)=shiftdim(Vtempii,1);
        d3_ind=rem(maxindexL2-1,N_d3)+1;
        allind=d3_ind+N_d3*aind+N_d3*N_a*zeindB;
        Policy(2,:,:,:,jj)=d3_ind;
        Policy(3,:,:,:,jj)=shiftdim(squeeze(midpoint_jj(allind)),-1);
        Policy(4,:,:,:,jj)=shiftdim(ceil(maxindexL2/N_d3),-1);

        % L2flag
        L2offset      = ceil(maxindexL2/N_d3);
        linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
        linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zeindB;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(5,:,:,:,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

        % Get the d2Policy
        d3opt=d3_ind;
        a1opt_mid=midpoint_jj(allind);
        zlin=shiftdim(gpuArray(0:N_z-1),-1);
        lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
        Policy(1,:,:,:,jj)=d2index_resh(lin);

    elseif vfoptions.lowmemory>=1
        special_n_e=ones(1,length(n_e));
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,n_a1,n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);
            entireRHS_e=ReturnMatrix_e+DiscountedEV;
            [~,maxindex]=max(entireRHS_e,[],2);
            midpoint_jj=max(min(maxindex,n_a1(1)-1),2);

            % Grid interpolation layer
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n2long,n_a1,n_a2,n_z,special_n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
            da1primez=(1:1:N_d3)'+N_d3*(a1primeindexesfine-1)+N_d3*N_a1prime*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primez(:)),[N_d3*n2long,N_a1*N_a2,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,:,e_c,jj)=shiftdim(Vtempii,1);
            d3_ind=rem(maxindexL2-1,N_d3)+1;
            allind=d3_ind+N_d3*aind+N_d3*N_a*zindB;
            Policy(2,:,:,e_c,jj)=d3_ind;
            Policy(3,:,:,e_c,jj)=shiftdim(squeeze(midpoint_jj(allind)),-1);
            Policy(4,:,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d3),-1);

            % L2flag
            L2offset      = ceil(maxindexL2/N_d3);
            linidx_lower  = d3_ind                   + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
            linidx_upper  = d3_ind + N_d3*(n2long-1) + N_d3*n2long*aind + N_d3*n2long*N_a*zindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(5,:,:,e_c,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

            % Get the d2Policy
            d3opt=d3_ind;
            a1opt_mid=midpoint_jj(allind);
            zlin=shiftdim(gpuArray(0:N_z-1),-1);
            lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*zlin;
            Policy(1,:,:,e_c,jj)=d2index_resh(lin);
        end
    end
end

%% Switch Policy(3,:) from 'midpoint' to 'lower grid index' (using L2ind side)
adjust=(Policy(4,:,:,:,:)<1+n2short+1);                                                  % L2ind strictly < n2short+2
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;                                              % decrement midpoint when chosen-below
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-(n2short+1)*(~adjust);   % rebase L2ind to [1..n2short+2]

end
