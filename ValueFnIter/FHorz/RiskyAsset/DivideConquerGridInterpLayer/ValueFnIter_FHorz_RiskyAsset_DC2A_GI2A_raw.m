function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAsset_DC1_GI1_raw.
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% With z.
%
% a1: standard endogenous state, this is the one divide-and-conquer (and then the grid interp layer) is applied to
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% The EV pipeline is unchanged from the DC1_GI1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset. DiscountedEV is (d3,a1prime,a2prime,-,-,-,z), no a3 term.

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

n_d13=[n_d1,n_d3];
N_d13=N_d1*N_d3;
d13_grid=[d1_grid; d3_grid];
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(6,N_a,N_z,N_j,'gpuArray'); % (1)=d1, (2)=d2, (3)=d3, (4)=a1prime midpoint, (5)=a2prime, (6)=a1prime L2
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray');
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

% Setup for DC (over a1 only)
if vfoptions.lowmemory==0
    midpoint=zeros(N_d13,1,N_a2,N_a1,N_a2,N_a3,N_z,'gpuArray');
elseif vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
    midpoint=zeros(N_d13,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Setup for GI
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% Precompute
aind=gpuArray(0:1:N_a-1);
zBind=shiftdim(gpuArray(0:1:N_z-1),-1);    % [1,1,N_z]
d3ind=repelem(gpuArray(1:1:N_d3)',N_d1,1); % [N_d13,1]; maps full d13-index to d3-component
a1pcol=reshape(0:1:N_a1-1,[1,N_a1]);       % [1,N_a1prime]
a2pcol=reshape(0:1:N_a2-1,[1,1,N_a2]);     % [1,1,N_a2prime]

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        % [N_d13, N_a1prime, N_a2prime, level1n, N_a2, N_a3, N_z]
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        % Divide-and-conquer layer 2
        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        % Grid interpolation layer
        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind       =rem(maxindexL2-1,N_d13)+1; % d13 index
        d1_ind      =rem(d_ind-1,N_d1)+1;
        d3_ind      =ceil(d_ind/N_d1);
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;

        allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zBind;
        Policy(1,:,:,N_j)=d1_ind;
        Policy(3,:,:,N_j)=d3_ind;
        Policy(4,:,:,N_j)=midpoint(allind);
        Policy(5,:,:,N_j)=maxindexL2a2;
        Policy(6,:,:,N_j)=maxindexL2a1;

        % L2flag
        linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zBind;
        linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zBind;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);

            % Layer 1
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            % Divide-and-conquer layer 2
            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii_z,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            % Grid interpolation layer
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d13)+1;
            d1_ind      =rem(d_ind-1,N_d1)+1;
            d3_ind      =ceil(d_ind/N_d1);
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;

            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
            Policy(1,:,z_c,N_j)=d1_ind;
            Policy(3,:,z_c,N_j)=d3_ind;
            Policy(4,:,z_c,N_j)=midpoint(allind);
            Policy(5,:,z_c,N_j)=maxindexL2a2;
            Policy(6,:,z_c,N_j)=maxindexL2a1;

            % L2flag
            linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            isInfLower=(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end

    % d2, which was not in ReturnFn
    Policy(2,:,:,N_j)=ones(1,N_a,N_z,'gpuArray'); % d2 (terminal: d2 doesn't matter since it's only in the expectations term)

else % V_Jplus1

    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states
    EVnext=reshape(vfoptions.V_Jplus1,[N_a,N_z]);
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
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a12,N_z]),[],1);
    EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_z]);

    % DiscountedEV: (d3,a1prime,a2prime,-,-,-,z); note d1 is not in it, so it has to be indexed out onto d13
    DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_z]);
    % Interpolate EV over a1prime_grid
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_z]

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        d3aprimez=d3ind + N_d3*a1pcol + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4); % [N_d13,N_a1prime,N_a2prime,1,1,1,N_z]
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);

        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d3aprimez=d3ind + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                [~,maxindex_inner]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        % Grid interpolation layer
        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        aprimez=d3ind + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol + N_d3*N_a1fine*N_a2*shiftdim(zBind,-4);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d13*n2long*N_a2,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind       =rem(maxindexL2-1,N_d13)+1;
        d1_ind      =rem(d_ind-1,N_d1)+1;
        d3_ind      =ceil(d_ind/N_d1);
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;

        allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zBind;
        a1mid=midpoint(allind);
        Policy(1,:,:,N_j)=d1_ind;
        Policy(3,:,:,N_j)=d3_ind;
        Policy(4,:,:,N_j)=a1mid;
        Policy(5,:,:,N_j)=maxindexL2a2;
        Policy(6,:,:,N_j)=maxindexL2a1;

        % L2flag
        ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d13*n2long*N_a2,N_a,N_z]);
        linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zBind;
        linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zBind;
        isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        % Get the d2Policy
        lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zBind;
        Policy(2,:,:,N_j)=d2index_resh(lin);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
            d2index_z=d2index_resh(:,:,:,z_c);

            % Layer 1
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            d3aprime_z=d3ind + N_d3*a1pcol + N_d3*N_a1*a2pcol; % [N_d13,N_a1prime,N_a2prime]
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z(d3aprime_z);

            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            % Divide and conquer layer 2
            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    d3aprime_z=d3ind + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
                    entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z(d3aprime_z);
                    [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            % Grid interpolation layer
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprime_z=d3ind + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol;
            entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprime_z),[N_d13*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d13)+1;
            d1_ind      =rem(d_ind-1,N_d1)+1;
            d3_ind      =ceil(d_ind/N_d1);
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;

            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
            a1mid=midpoint(allind);
            Policy(1,:,z_c,N_j)=d1_ind;
            Policy(3,:,z_c,N_j)=d3_ind;
            Policy(4,:,z_c,N_j)=a1mid;
            Policy(5,:,z_c,N_j)=maxindexL2a2;
            Policy(6,:,z_c,N_j)=maxindexL2a1;

            % L2flag
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_z,[N_d13*n2long*N_a2,N_a]);
            linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % Get the d2Policy
            lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
            Policy(2,:,z_c,N_j)=d2index_z(lin);
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
    EVnext=V(:,:,jj+1);
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
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a12,N_z]),[],1);
    EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_z]);
    DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_z]

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        d3aprimez=d3ind + N_d3*a1pcol + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);

        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d3aprimez=d3ind + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                [~,maxindex_inner]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        % Grid interpolation layer
        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimez=d3ind + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol + N_d3*N_a1fine*N_a2*shiftdim(zBind,-4);
        entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d13*n2long*N_a2,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);

        d_ind       =rem(maxindexL2-1,N_d13)+1;
        d1_ind      =rem(d_ind-1,N_d1)+1;
        d3_ind      =ceil(d_ind/N_d1);
        maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
        maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;

        allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zBind;
        a1mid=midpoint(allind);
        Policy(1,:,:,jj)=d1_ind;
        Policy(3,:,:,jj)=d3_ind;
        Policy(4,:,:,jj)=a1mid;
        Policy(5,:,:,jj)=maxindexL2a2;
        Policy(6,:,:,jj)=maxindexL2a1;

        % L2flag
        ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d13*n2long*N_a2,N_a,N_z]);
        linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zBind;
        linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zBind;
        isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        % Get the d2Policy
        lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zBind;
        Policy(2,:,:,jj)=d2index_resh(lin);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
            d2index_z=d2index_resh(:,:,:,z_c);

            % Layer 1
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            d3aprime_z=d3ind + N_d3*a1pcol + N_d3*N_a1*a2pcol;
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z(d3aprime_z);

            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            % Divide and conquer layer 2
            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    d3aprime_z=d3ind + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
                    entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z(d3aprime_z);
                    [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            % Grid interpolation layer
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprime_z=d3ind + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol;
            entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprime_z),[N_d13*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            V(:,z_c,jj)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d13)+1;
            d1_ind      =rem(d_ind-1,N_d1)+1;
            d3_ind      =ceil(d_ind/N_d1);
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;

            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
            a1mid=midpoint(allind);
            Policy(1,:,z_c,jj)=d1_ind;
            Policy(3,:,z_c,jj)=d3_ind;
            Policy(4,:,z_c,jj)=a1mid;
            Policy(5,:,z_c,jj)=maxindexL2a2;
            Policy(6,:,z_c,jj)=maxindexL2a1;

            % L2flag
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_z,[N_d13*n2long*N_a2,N_a]);
            linidx_lower=d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            % Get the d2Policy
            lin=d3_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
            Policy(2,:,z_c,jj)=d2index_z(lin);
        end
    end
end


%% Switch Policy(4,:) from 'midpoint' to 'lower grid index'
adjust=(Policy(6,:,:,:)<1+n2short+1);
Policy(4,:,:,:)=Policy(4,:,:,:)-adjust;
Policy(6,:,:,:)=adjust.*Policy(6,:,:,:)+(1-adjust).*(Policy(6,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

end
