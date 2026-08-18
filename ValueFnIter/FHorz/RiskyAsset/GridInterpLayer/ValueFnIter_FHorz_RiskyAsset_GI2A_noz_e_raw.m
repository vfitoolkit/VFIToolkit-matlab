function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_GI2A_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, e_gridvals_J, u_grid, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAsset_GI1_noz_e_raw.
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No z, with e: iid start-of-period. e treated as z-slot in Return helper (no z exists).
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
N_e=prod(n_e);
N_u=prod(n_u);

N_a12=N_a1*N_a2; % the two standard assets, carried forward directly

n_d13=[n_d1,n_d3];
N_d13=N_d1*N_d3;
d13_grid=[d1_grid;d3_grid];
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(6,N_a,N_e,N_j,'gpuArray'); % (1)=d1, (2)=d2, (3)=d3, (4)=a1prime midpoint, (5)=a2prime, (6)=L2ind
PolicyL2flag=2*ones(1,N_a,N_e,N_j,'gpuArray');
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e),'gpuArray');
end

% Setup for GI (over a1 only)
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% Precompute
aind=gpuArray(0:1:N_a-1);
eindB=shiftdim((0:1:N_e-1),-1); % [1,1,N_e] (treated like zindB)

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1); % treat e as z (no z exists)
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint_jj=max(min(maxindex,N_a1-1),2);

        % Grid interpolation layer
        a1primeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_e, d13_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind       =rem(maxindexL2-1,N_d13)+1;
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
        d1_ind=rem(d_ind-1,N_d1)+1;
        d3_ind=ceil(d_ind/N_d1);

        allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*eindB;
        Policy(1,:,:,N_j)=d1_ind;                       % d1
        Policy(3,:,:,N_j)=d3_ind;                       % d3
        Policy(4,:,:,N_j)=midpoint_jj(allind);          % a1prime midpoint
        Policy(5,:,:,N_j)=maxindexL2a2;                 % a2prime
        Policy(6,:,:,N_j)=maxindexL2a1;                 % L2ind

        % L2flag
        linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*eindB;
        linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*eindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_e,[],2);
            midpoint_jj=max(min(maxindex,N_a1-1),2);

            % Grid interpolation layer
            a1primeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_e, d13_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d13)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
            d1_ind=rem(d_ind-1,N_d1)+1;
            d3_ind=ceil(d_ind/N_d1);

            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
            Policy(1,:,e_c,N_j)=d1_ind;
            Policy(3,:,e_c,N_j)=d3_ind;
            Policy(4,:,e_c,N_j)=midpoint_jj(allind);
            Policy(5,:,e_c,N_j)=maxindexL2a2;
            Policy(6,:,e_c,N_j)=maxindexL2a1;

            % L2flag
            linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            isInfLower   =(ReturnMatrix_ii_e(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_e(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end

    % d2, which was not in ReturnFn
    Policy(2,:,:,N_j)=ones(1,N_a,N_e,'gpuArray'); % d2 (terminal: d2 doesn't matter, only in expectations)

else % V_Jplus1

    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);
    EV=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j+1),-1),2); % [N_a,1]

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a3primeProbs,N_a12,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a12,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a12,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a12]),[],1);
    EV=reshape(EV,[N_d3*N_a12,1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2]);

    % DiscountedEV: (d3,a1prime,a2prime), broadcast against the (a1,a2,a3,e) dims of ReturnMatrix
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,N_a2]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3]),a1prime_grid),[2,1,3]); % [N_d3,N_a1fine,N_a2]
    % Broadcast d1 onto DiscountedEV by repelem along the first dim
    DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);             % [N_d13,N_a1,N_a2]
    DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1); % [N_d13,N_a1fine,N_a2]

    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1); % e in z-slot
        entireRHS=ReturnMatrix+DiscountedEV_d13;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint_jj=max(min(maxindex,N_a1-1),2);

        % Grid interpolation layer
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_e, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        % EV does not depend on a3 nor e; a1primeindexesfine already carries those dims
        aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprime),[N_d13*n2long*N_a2,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind       =rem(maxindexL2-1,N_d13)+1;
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
        d1_ind=rem(d_ind-1,N_d1)+1;
        d3_ind=ceil(d_ind/N_d1);

        allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*eindB;
        Policy(1,:,:,N_j)=d1_ind;
        Policy(3,:,:,N_j)=d3_ind;
        Policy(4,:,:,N_j)=midpoint_jj(allind);
        Policy(5,:,:,N_j)=maxindexL2a2;
        Policy(6,:,:,N_j)=maxindexL2a1;

        % L2flag
        linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*eindB;
        linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*eindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        % Get the d2Policy
        a1mid=midpoint_jj(allind); % [1,N_a,N_e]
        lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
        Policy(2,:,:,N_j)=reshape(d2index_resh(lin),[1,N_a,N_e]);

    elseif vfoptions.lowmemory>=1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            entireRHS_e=ReturnMatrix_e+DiscountedEV_d13;
            [~,maxindex]=max(entireRHS_e,[],2);
            midpoint_jj=max(min(maxindex,N_a1-1),2);

            % Grid interpolation layer
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_e, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
            aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprime),[N_d13*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d13)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
            d1_ind=rem(d_ind-1,N_d1)+1;
            d3_ind=ceil(d_ind/N_d1);

            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
            Policy(1,:,e_c,N_j)=d1_ind;
            Policy(3,:,e_c,N_j)=d3_ind;
            Policy(4,:,e_c,N_j)=midpoint_jj(allind);
            Policy(5,:,e_c,N_j)=maxindexL2a2;
            Policy(6,:,e_c,N_j)=maxindexL2a1;

            % L2flag
            linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % Get the d2Policy
            a1mid=midpoint_jj(allind);
            lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
            Policy(2,:,e_c,N_j)=d2index_resh(lin);
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

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states
    EV=sum(V(:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-1),2); % [N_a,1]
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a3primeProbs,N_a12,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a12,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a12,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a12]),[],1);
    EV=reshape(EV,[N_d3*N_a12,1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,N_a2]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3]),a1prime_grid),[2,1,3]); % [N_d3,N_a1fine,N_a2]
    DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
    DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS=ReturnMatrix+DiscountedEV_d13;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint_jj=max(min(maxindex,N_a1-1),2);

        % Grid interpolation layer
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_e, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprime),[N_d13*n2long*N_a2,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);

        d_ind       =rem(maxindexL2-1,N_d13)+1;
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
        d1_ind=rem(d_ind-1,N_d1)+1;
        d3_ind=ceil(d_ind/N_d1);

        allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*eindB;
        Policy(1,:,:,jj)=d1_ind;
        Policy(3,:,:,jj)=d3_ind;
        Policy(4,:,:,jj)=midpoint_jj(allind);
        Policy(5,:,:,jj)=maxindexL2a2;
        Policy(6,:,:,jj)=maxindexL2a1;

        % L2flag
        linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*eindB;
        linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*eindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        % Get the d2Policy
        a1mid=midpoint_jj(allind); % [1,N_a,N_e]
        lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
        Policy(2,:,:,jj)=reshape(d2index_resh(lin),[1,N_a,N_e]);

    elseif vfoptions.lowmemory>=1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            entireRHS_e=ReturnMatrix_e+DiscountedEV_d13;
            [~,maxindex]=max(entireRHS_e,[],2);
            midpoint_jj=max(min(maxindex,N_a1-1),2);

            % Grid interpolation layer
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_e, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
            aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprime),[N_d13*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,e_c,jj)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d13)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
            d1_ind=rem(d_ind-1,N_d1)+1;
            d3_ind=ceil(d_ind/N_d1);

            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
            Policy(1,:,e_c,jj)=d1_ind;
            Policy(3,:,e_c,jj)=d3_ind;
            Policy(4,:,e_c,jj)=midpoint_jj(allind);
            Policy(5,:,e_c,jj)=maxindexL2a2;
            Policy(6,:,e_c,jj)=maxindexL2a1;

            % L2flag
            linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,e_c,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % Get the d2Policy
            a1mid=midpoint_jj(allind);
            lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
            Policy(2,:,e_c,jj)=d2index_resh(lin);
        end
    end
end

%% Switch Policy(4,:) from 'midpoint' to 'lower grid index' (using L2ind side)
adjust=(Policy(6,:,:,:)<1+n2short+1);                                            % L2ind strictly < n2short+2
Policy(4,:,:,:)=Policy(4,:,:,:)-adjust;                                          % decrement midpoint when chosen-below
Policy(6,:,:,:)=adjust.*Policy(6,:,:,:)+(1-adjust).*(Policy(6,:,:,:)-n2short-1); % rebase L2ind to [1..n2short+2]

Policy=[Policy; PolicyL2flag];

end
