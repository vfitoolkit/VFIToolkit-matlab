function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_GI2A_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAsset_GI1_raw.
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
%
% a1: standard endogenous state, this is the one the grid interpolation layer refines
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% Policy is 6-channel: 1=d1, 2=d2, 3=d3, 4=a1prime midpoint, 5=a2prime, 6=a1prime L2.
% A 7th channel PolicyL2flag is concatenated at the end.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);
N_u=prod(n_u);

N_a12=N_a1*N_a2; % the two standard assets, carried forward directly

% For ReturnFn (d1 and d3 only)
n_d13=[n_d1,n_d3];
N_d13=N_d1*N_d3;
d13_grid=[d1_grid;d3_grid];
% For aprimeFn (d2 and d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(6,N_a,N_z,N_j,'gpuArray'); % (1)=d1, (2)=d2, (3)=d3, (4)=a1prime midpoint, (5)=a2prime, (6)=L2ind
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray');
% d2 stored directly into Policy(2,...) via lookup after GI search

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

if vfoptions.lowmemory>=1
    special_n_z=ones(1,length(n_z));
end

% Setup for GI (over a1 only)
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% Precompute
aind=gpuArray(0:1:N_a-1);
zindB=shiftdim(gpuArray(0:1:N_z-1),-1);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No EV; d2 is meaningless. Just do the GI search on Return alone.
    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        % [N_d13, N_a1prime, N_a2prime, N_a1, N_a2, N_a3, N_z]
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint_jj=max(min(maxindex,N_a1-1),2);

        % Grid interpolation layer
        a1primeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        % [N_d13*n2long*N_a2, N_a, N_z]
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind       =rem(maxindexL2-1,N_d13)+1;
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
        d1_ind=rem(d_ind-1,N_d1)+1;
        d3_ind=ceil(d_ind/N_d1);

        allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zindB;
        Policy(1,:,:,N_j)=d1_ind;                       % d1
        Policy(3,:,:,N_j)=d3_ind;                       % d3
        Policy(4,:,:,N_j)=midpoint_jj(allind);          % a1prime midpoint
        Policy(5,:,:,N_j)=maxindexL2a2;                 % a2prime
        Policy(6,:,:,N_j)=maxindexL2a1;                 % L2ind

        % L2flag
        linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        % d2 meaningless at j=N_j (no future)
        Policy(2,:,:,N_j)=ones(1,N_a,N_z,'gpuArray');

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_z,[],2);
            midpoint_jj=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d13)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
            d1_ind=rem(d_ind-1,N_d1)+1;
            d3_ind=ceil(d_ind/N_d1);

            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
            Policy(1,:,z_c,N_j)=d1_ind;                       % d1
            Policy(3,:,z_c,N_j)=d3_ind;                       % d3
            Policy(4,:,z_c,N_j)=midpoint_jj(allind);          % a1prime midpoint
            Policy(5,:,z_c,N_j)=maxindexL2a2;                 % a2prime
            Policy(6,:,z_c,N_j)=maxindexL2a1;                 % L2ind

            % L2flag
            linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
        % d2 meaningless at j=N_j
        Policy(2,:,:,N_j)=ones(1,N_a,N_z,'gpuArray');
    end
else % V_Jplus1

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]);

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states: integrate zprime first
    EV=EVpre.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);
    EV=reshape(EV,[N_a,N_z]);

    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
    aprimeProbs=repmat(a3primeProbs,N_a12,N_z);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_z]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a12,N_u,N_z]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a12,N_u,N_z]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    EV=reshape(EV,[N_d23*N_a12,N_z]);

    % Refine d2 out of EV before combining with ReturnFn
    EVres=reshape(EV,[N_d2,N_d3*N_a12,N_z]);
    [EV_onlyd3,d2index]=max(EVres,[],1);
    EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_z]);

    % DiscountedEV: (d3,a1prime,a2prime,-,-,-,z)
    DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_z]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_z]

    % Broadcast d1 onto DiscountedEV by repelem along the first dim — turns N_d3 into N_d13 with d1 fastest
    DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);             % [N_d13,N_a1,N_a2,1,1,1,N_z]
    DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1); % [N_d13,N_a1fine,N_a2,1,1,1,N_z]

    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        entireRHS=ReturnMatrix+DiscountedEV_d13;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint_jj=max(min(maxindex,N_a1-1),2);

        % Grid interpolation layer
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        % EV does not depend on a3, so the linear index into DiscountedEVinterp_d13 omits an a3 offset
        aprimez=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d13*N_a1fine*N_a2*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprimez),[N_d13*n2long*N_a2,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind       =rem(maxindexL2-1,N_d13)+1;
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
        d1_ind=rem(d_ind-1,N_d1)+1;
        d3_ind=ceil(d_ind/N_d1);

        allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zindB;
        Policy(1,:,:,N_j)=d1_ind;
        Policy(3,:,:,N_j)=d3_ind;
        Policy(4,:,:,N_j)=midpoint_jj(allind);
        Policy(5,:,:,N_j)=maxindexL2a2;
        Policy(6,:,:,N_j)=maxindexL2a1;

        % L2flag
        linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        % d2 lookup: d2index_resh(d3, a1prime_midpoint, a2prime, z)
        a1mid=midpoint_jj(allind); % [1,N_a,N_z]
        zlin=shiftdim(gpuArray(0:N_z-1),-1); % [1,1,N_z]
        lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
        Policy(2,:,:,N_j)=d2index_resh(lin);

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV_d13(:,:,:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp_d13(:,:,:,:,:,:,z_c);
            d2index_z=d2index_resh(:,:,:,z_c); % [N_d3,N_a1,N_a2]

            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            entireRHS_z=ReturnMatrix_z+DiscountedEV_z;
            [~,maxindex]=max(entireRHS_z,[],2);
            midpoint_jj=max(min(maxindex,N_a1-1),2);

            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprime_z=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1);
            entireRHS_ii=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprime_z),[N_d13*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d13)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
            d1_ind=rem(d_ind-1,N_d1)+1;
            d3_ind=ceil(d_ind/N_d1);

            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
            Policy(1,:,z_c,N_j)=d1_ind;
            Policy(3,:,z_c,N_j)=d3_ind;
            Policy(4,:,z_c,N_j)=midpoint_jj(allind);
            Policy(5,:,z_c,N_j)=maxindexL2a2;
            Policy(6,:,z_c,N_j)=maxindexL2a1;

            % L2flag
            linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % d2 lookup
            a1mid=midpoint_jj(allind);
            lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
            Policy(2,:,z_c,N_j)=d2index_z(lin);
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj));

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states: integrate zprime first
    EV=V(:,:,jj+1).*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);
    EV=reshape(EV,[N_a,N_z]);

    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
    aprimeProbs=repmat(a3primeProbs,N_a12,N_z);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_z]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a12,N_u,N_z]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a12,N_u,N_z]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    EV=reshape(EV,[N_d23*N_a12,N_z]);

    % Refine d2 out of EV before combining with ReturnFn
    EVres=reshape(EV,[N_d2,N_d3*N_a12,N_z]);
    [EV_onlyd3,d2index]=max(EVres,[],1);
    EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_z]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_z]

    % Broadcast d1 onto DiscountedEV by repelem along the first dim
    DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
    DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS=ReturnMatrix+DiscountedEV_d13;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint_jj=max(min(maxindex,N_a1-1),2);

        % Grid interpolation layer
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimez=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d13*N_a1fine*N_a2*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprimez),[N_d13*n2long*N_a2,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);

        d_ind       =rem(maxindexL2-1,N_d13)+1;
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
        d1_ind=rem(d_ind-1,N_d1)+1;
        d3_ind=ceil(d_ind/N_d1);

        allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zindB;
        Policy(1,:,:,jj)=d1_ind;
        Policy(3,:,:,jj)=d3_ind;
        Policy(4,:,:,jj)=midpoint_jj(allind);
        Policy(5,:,:,jj)=maxindexL2a2;
        Policy(6,:,:,jj)=maxindexL2a1;

        % L2flag
        linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        % d2 lookup
        a1mid=midpoint_jj(allind); % [1,N_a,N_z]
        zlin=shiftdim(gpuArray(0:N_z-1),-1); % [1,1,N_z]
        lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
        Policy(2,:,:,jj)=d2index_resh(lin);

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV_d13(:,:,:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp_d13(:,:,:,:,:,:,z_c);
            d2index_z=d2index_resh(:,:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            entireRHS_z=ReturnMatrix_z+DiscountedEV_z;
            [~,maxindex]=max(entireRHS_z,[],2);
            midpoint_jj=max(min(maxindex,N_a1-1),2);

            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprime_z=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1);
            entireRHS_ii=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprime_z),[N_d13*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,z_c,jj)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d13)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
            d1_ind=rem(d_ind-1,N_d1)+1;
            d3_ind=ceil(d_ind/N_d1);

            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
            Policy(1,:,z_c,jj)=d1_ind;
            Policy(3,:,z_c,jj)=d3_ind;
            Policy(4,:,z_c,jj)=midpoint_jj(allind);
            Policy(5,:,z_c,jj)=maxindexL2a2;
            Policy(6,:,z_c,jj)=maxindexL2a1;

            % L2flag
            linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % d2 lookup
            a1mid=midpoint_jj(allind);
            lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
            Policy(2,:,z_c,jj)=d2index_z(lin);
        end
    end
end

%% Switch Policy(4,:) from 'midpoint' to 'lower grid index' (using L2ind side)
adjust=(Policy(6,:,:,:)<1+n2short+1);                                            % L2ind strictly < n2short+2
Policy(4,:,:,:)=Policy(4,:,:,:)-adjust;                                          % decrement midpoint when chosen-below
Policy(6,:,:,:)=adjust.*Policy(6,:,:,:)+(1-adjust).*(Policy(6,:,:,:)-n2short-1); % rebase L2ind to [1..n2short+2]

Policy=[Policy; PolicyL2flag];

end
