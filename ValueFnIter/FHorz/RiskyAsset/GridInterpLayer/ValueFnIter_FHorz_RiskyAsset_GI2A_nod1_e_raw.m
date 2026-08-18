function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_GI2A_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_e,n_u,N_j, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, e_gridvals_J, u_grid, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAsset_GI1_nod1_e_raw.
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No d1. e: iid start-of-period.
%
% a1: standard endogenous state, this is the one the grid interpolation layer refines
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% Policy is 5-channel: 1=d2, 2=d3, 3=a1prime midpoint, 4=a2prime, 5=a1prime L2.
% A 6th channel PolicyL2flag is concatenated at the end.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);
N_e=prod(n_e);
N_u=prod(n_u);

N_a12=N_a1*N_a2; % the two standard assets, carried forward directly

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_z,N_e,N_j,'gpuArray'); % (1)=d2, (2)=d3, (3)=a1prime midpoint, (4)=a2prime, (5)=L2ind
PolicyL2flag=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray');
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d3_gridvals=CreateGridvals(n_d3,d3_grid,1);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e),'gpuArray');
end
if vfoptions.lowmemory==2
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
zeindB=zindB+N_z*shiftdim((0:1:N_e-1),-2);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint_jj=max(min(maxindex,N_a1-1),2);

        % Grid interpolation layer
        a1primeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, n_e, d3_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);

        d3_ind      =rem(maxindexL2-1,N_d3)+1;
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

        allind=d3_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zeindB;
        Policy(2,:,:,:,N_j)=d3_ind;                       % d3
        Policy(3,:,:,:,N_j)=midpoint_jj(allind);          % a1prime midpoint
        Policy(4,:,:,:,N_j)=maxindexL2a2;                 % a2prime
        Policy(5,:,:,:,N_j)=maxindexL2a1;                 % L2ind

        % L2flag
        linidx_lower=d3_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zeindB;
        linidx_upper=d3_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zeindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        % d2, which was not in ReturnFn
        Policy(1,:,:,:,N_j)=ones(1,N_a,N_z,N_e,'gpuArray'); % d2 (terminal: d2 doesn't matter, only in expectations)
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, special_n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_e,[],2);
            midpoint_jj=max(min(maxindex,N_a1-1),2);

            % Grid interpolation layer
            a1primeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, special_n_e, d3_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);

            d3_ind      =rem(maxindexL2-1,N_d3)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

            allind=d3_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zindB;
            Policy(2,:,:,e_c,N_j)=d3_ind;
            Policy(3,:,:,e_c,N_j)=midpoint_jj(allind);
            Policy(4,:,:,e_c,N_j)=maxindexL2a2;
            Policy(5,:,:,e_c,N_j)=maxindexL2a1;

            % L2flag
            linidx_lower=d3_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zindB;
            linidx_upper=d3_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zindB;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % d2, which was not in ReturnFn
            Policy(1,:,:,e_c,N_j)=ones(1,N_a,N_z,'gpuArray'); % d2 (terminal: d2 doesn't matter, only in expectations)
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % Layer 1: full ReturnMatrix max for initial midpoint
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_z, special_n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_ze,[],2);
                midpoint_jj=max(min(maxindex,N_a1-1),2);

                % Grid interpolation layer
                a1primeindexes=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_z, special_n_e, d3_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);

                d3_ind      =rem(maxindexL2-1,N_d3)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

                allind=d3_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind;
                Policy(2,:,z_c,e_c,N_j)=d3_ind;
                Policy(3,:,z_c,e_c,N_j)=midpoint_jj(allind);
                Policy(4,:,z_c,e_c,N_j)=maxindexL2a2;
                Policy(5,:,z_c,e_c,N_j)=maxindexL2a1;

                % L2flag
                linidx_lower=d3_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
                linidx_upper=d3_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                % d2, which was not in ReturnFn
                Policy(1,:,z_c,e_c,N_j)=ones(1,N_a,'gpuArray'); % d2 (terminal: d2 doesn't matter, only in expectations)
            end
        end
    end
else % V_Jplus1

    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);
    EVnext=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j+1),-2),3);

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states
    EV=EVnext.*shiftdim(pi_z_J(:,:,N_j)',-1);
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

    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        entireRHS=ReturnMatrix+DiscountedEV;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint_jj=max(min(maxindex,N_a1-1),2);

        % Grid interpolation layer
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        % EV does not depend on a3 nor e; a1primeindexesfine already carries those dims
        aprimez=(1:1:N_d3)' + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d3*N_a1fine*N_a2*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d3*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);

        d3_ind      =rem(maxindexL2-1,N_d3)+1;
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

        allind=d3_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zeindB;
        Policy(2,:,:,:,N_j)=d3_ind;
        Policy(3,:,:,:,N_j)=midpoint_jj(allind);
        Policy(4,:,:,:,N_j)=maxindexL2a2;
        Policy(5,:,:,:,N_j)=maxindexL2a1;

        % L2flag
        linidx_lower=d3_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zeindB;
        linidx_upper=d3_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zeindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        % Get the d2Policy: d2index_resh depends on (d3, a1prime_midpoint, a2prime, z) — not e
        a1mid=midpoint_jj(allind); % [1,N_a,N_z,N_e]
        zlin=shiftdim(gpuArray(0:N_z-1),-1); % [1,1,N_z]
        lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
        Policy(1,:,:,:,N_j)=d2index_resh(lin);

    elseif vfoptions.lowmemory>=1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, special_n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            entireRHS_e=ReturnMatrix_e+DiscountedEV;
            [~,maxindex]=max(entireRHS_e,[],2);
            midpoint_jj=max(min(maxindex,N_a1-1),2);

            % Grid interpolation layer
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, special_n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d3)' + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d3*N_a1fine*N_a2*shiftdim((0:1:N_z-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d3*n2long*N_a2,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);

            d3_ind      =rem(maxindexL2-1,N_d3)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

            allind=d3_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zindB;
            Policy(2,:,:,e_c,N_j)=d3_ind;
            Policy(3,:,:,e_c,N_j)=midpoint_jj(allind);
            Policy(4,:,:,e_c,N_j)=maxindexL2a2;
            Policy(5,:,:,e_c,N_j)=maxindexL2a1;

            % L2flag
            linidx_lower=d3_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zindB;
            linidx_upper=d3_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zindB;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % Get the d2Policy
            a1mid=midpoint_jj(allind); % [1,N_a,N_z]
            zlin=shiftdim(gpuArray(0:N_z-1),-1); % [1,1,N_z]
            lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
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

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states
    EVnext=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);
    EV=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1);
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

    if vfoptions.lowmemory==0
        % Layer 1: full ReturnMatrix max for initial midpoint
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS=ReturnMatrix+DiscountedEV;
        [~,maxindex]=max(entireRHS,[],2);
        midpoint_jj=max(min(maxindex,N_a1-1),2);

        % Grid interpolation layer
        a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimez=(1:1:N_d3)' + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d3*N_a1fine*N_a2*shiftdim((0:1:N_z-1),-5);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d3*n2long*N_a2,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,jj)=shiftdim(Vtempii,1);

        d3_ind      =rem(maxindexL2-1,N_d3)+1;
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

        allind=d3_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zeindB;
        Policy(2,:,:,:,jj)=d3_ind;
        Policy(3,:,:,:,jj)=midpoint_jj(allind);
        Policy(4,:,:,:,jj)=maxindexL2a2;
        Policy(5,:,:,:,jj)=maxindexL2a1;

        % L2flag
        linidx_lower=d3_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zeindB;
        linidx_upper=d3_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zeindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        % Get the d2Policy
        a1mid=midpoint_jj(allind); % [1,N_a,N_z,N_e]
        zlin=shiftdim(gpuArray(0:N_z-1),-1); % [1,1,N_z]
        lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
        Policy(1,:,:,:,jj)=d2index_resh(lin);

    elseif vfoptions.lowmemory>=1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % Layer 1: full ReturnMatrix max for initial midpoint
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, special_n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
            entireRHS_e=ReturnMatrix_e+DiscountedEV;
            [~,maxindex]=max(entireRHS_e,[],2);
            midpoint_jj=max(min(maxindex,N_a1-1),2);

            % Grid interpolation layer
            a1primeindexesfine=(midpoint_jj+(midpoint_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d3, n_a2, n_a3, n_z, special_n_e, d3_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d3)' + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d3*N_a1fine*N_a2*shiftdim((0:1:N_z-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d3*n2long*N_a2,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,:,e_c,jj)=shiftdim(Vtempii,1);

            d3_ind      =rem(maxindexL2-1,N_d3)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

            allind=d3_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zindB;
            Policy(2,:,:,e_c,jj)=d3_ind;
            Policy(3,:,:,e_c,jj)=midpoint_jj(allind);
            Policy(4,:,:,e_c,jj)=maxindexL2a2;
            Policy(5,:,:,e_c,jj)=maxindexL2a1;

            % L2flag
            linidx_lower=d3_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zindB;
            linidx_upper=d3_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zindB;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,:,e_c,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % Get the d2Policy
            a1mid=midpoint_jj(allind); % [1,N_a,N_z]
            zlin=shiftdim(gpuArray(0:N_z-1),-1); % [1,1,N_z]
            lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
            Policy(1,:,:,e_c,jj)=d2index_resh(lin);
        end
    end
end

%% Switch Policy(3,:) from 'midpoint' to 'lower grid index' (using L2ind side)
adjust=(Policy(5,:,:,:,:)<1+n2short+1);                                              % L2ind strictly < n2short+2
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;                                          % decrement midpoint when chosen-below
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1); % rebase L2ind to [1..n2short+2]

Policy=[Policy; PolicyL2flag];

end
