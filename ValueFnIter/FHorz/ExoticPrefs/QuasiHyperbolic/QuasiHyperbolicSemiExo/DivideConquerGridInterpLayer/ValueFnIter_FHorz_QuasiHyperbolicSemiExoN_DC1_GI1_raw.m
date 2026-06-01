function [Vtilde, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_raw(n_d1,n_d2,n_a,n_z,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic + SemiExo + DC + GI raw, with d1, z, no e.
% Dual-V (V for beta and Vtilde for beta0beta) wrapped around d2 outer loop, with DC level1n + GI midpoint+L2 + L2flag.
% Output: (Vtilde=Vtilde, Policy3, Valt=V).

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Policy=zeros(4,N_a,N_semiz*N_z,N_j,'gpuArray'); % first dim: d1,d2,aprime midpoint, aprimeL2ind
PolicyL2flag=2*ones(1,N_a,N_semiz*N_z,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper
Policyalt=zeros(4,N_a,N_semiz*N_z,N_j,'gpuArray'); % exponential discounter optimal [d1; d2; midpoint; aprimeL2ind]
PolicyL2flagalt=2*ones(1,N_a,N_semiz*N_z,N_j,'gpuArray');

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

aind=gpuArray(0:1:N_a-1);
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
bothzind2=shiftdim(gpuArray(0:1:N_bothz-1),-2);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate per-d2 slabs (for Vtilde -- the agent's choice and the Policy)
Valt_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray'); % the V-slab corresponding to the maxindex selected by Vtilde
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray'); % Vtilde slab
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray'); % d1*aprimeL2ind combined
midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
PolicyL2flag_ford2_jj=2*ones(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
midpointV_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
PolicyL2flagV_ford2_jj=2*ones(N_a,N_semiz*N_z,N_d2,'gpuArray');
midpoints_jj=zeros(N_d1,1,N_a,N_semiz*N_z,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);


%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period: V=Vtilde and Policy comes from undiscounted return.
    midpoints_Nj=zeros(N_d,1,N_a,N_semiz*N_z,'gpuArray');

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_bothz, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    midpoints_Nj(:,1,level1ii,:)=maxindex1;
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_bothz, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
            [~,maxindex]=max(ReturnMatrix_ii,[],2);
            midpoints_Nj(:,1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:);
            midpoints_Nj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1);
        end
    end
    midpoints_Nj=max(min(midpoints_Nj,n_a-1),2);
    aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,n_bothz,d_gridvals,aprime_grid(aprimeindexes),a_grid,bothz_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind+N_d*N_a*bothzind;
    Policy(1,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1); % d1
    Policy(2,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1); % d2
    Policy(3,:,:,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1); % midpoint
    Policy(4,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    % L2 flag
    L2offset = ceil(maxindexL2/N_d);
    linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
    linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    Vtilde(:,:,N_j)=V(:,:,N_j);
    % terminal: QH and exponential discounter coincide
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j);
    PolicyL2flagalt(1,:,:,N_j)=PolicyL2flag(1,:,:,N_j);

else
    % Using V_Jplus1 (V for naive)
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_z]);

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2); % sum over z'

        EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

        % n-Monotonicity (shared Return matrix between V and Vtilde branches)
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        %% V (beta) -- time-consistent
        entireRHS_ii=ReturnMatrix_d2ii+beta*shiftdim(EV_d2,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:)=maxindex1;
        maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                aprimez=aprimeindexes+N_a*bothzind2;
                entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[N_d1,(maxgap_V(ii)+1),1,N_bothz]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*bothzind2;
        entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_bothz]);
        [Vtemp,maxindexL2alt]=max(entireRHS_L2,[],1);
        Valt_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
        Policy_V_ford2_jj(:,:,d2_c)=shiftdim(maxindexL2alt,1);
        d1_indalt=rem(maxindexL2alt-1,N_d1)+1;
        allindalt=d1_indalt+N_d1*aind+N_d1*N_a*bothzind;
        midpointV_ford2_jj(:,:,d2_c)=squeeze(midpoints_jj(allindalt));
        L2offsetalt = ceil(maxindexL2alt/N_d1);
        linidx_loweralt = d1_indalt                  + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
        linidx_upperalt = d1_indalt + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
        isInfLoweralt    = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
        isInfUpperalt    = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
        inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
        inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
        PolicyL2flagV_ford2_jj(:,:,d2_c) = squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)); % V-slab for this d2

        %% Vtilde (beta0*beta) -- agent's choice
        entireRHS_ii=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:)=maxindex1;
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                aprimez=aprimeindexes+N_a*bothzind2;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,(maxgap(ii)+1),1,N_bothz]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*bothzind2;
        entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_bothz]);
        [Vtemp,maxindex]=max(entireRHS_L2,[],1);

        V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
        Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

        d1_ind=rem(maxindex-1,N_d1)+1;
        allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind;
        midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoints_jj(allind));

        % L2 flag for this d2 (based on Vtilde -- the agent's choice)
        L2offset_d2 = ceil(maxindex/N_d1);
        linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
        linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
        isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
        inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
        inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
        PolicyL2flag_ford2_jj(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
    end

    % Max over d2 using Vtilde (agent's choice), then gather V slab from same d2
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2 of Vtilde
    Vtilde(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d2
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    % V at exponential discounter optimum (full max over d2, d1, aprime)
    [V_jjalt,maxindexalt_d2]=max(Valt_ford2_jj,[],3);
    V(:,:,N_j)=V_jjalt;
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy(1,:,:,N_j)=shiftdim(rem(d1aprimeL2_ind-1,N_d1)+1,-1); % d1
    Policy(4,:,:,N_j)=shiftdim(ceil(d1aprimeL2_ind/N_d1),-1); % aprimeL2ind
    Policy(3,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    PolicyL2flag(1,:,:,N_j)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policyalt(2,:,:,N_j)=shiftdim(maxindexalt_d2,-1);
    maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z,1]);
    d1aprimeL2_ind_alt=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);
    Policyalt(1,:,:,N_j)=shiftdim(rem(d1aprimeL2_ind_alt-1,N_d1)+1,-1);
    Policyalt(4,:,:,N_j)=shiftdim(ceil(d1aprimeL2_ind_alt/N_d1),-1);
    Policyalt(3,:,:,N_j)=reshape(midpointV_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);
    PolicyL2flagalt(1,:,:,N_j)=reshape(PolicyL2flagV_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EV=reshape(V(:,:,jj+1),[N_a,N_semiz,N_z]); % Naive uses V (time-consistent) for continuation

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj)); % reverse order

        EV_d2=EV.*shiftdim(pi_bothz',-1);
        EV_d2(isnan(EV_d2))=0;
        EV_d2=sum(EV_d2,2);

        EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        %% V (beta)
        entireRHS_ii=ReturnMatrix_d2ii+beta*shiftdim(EV_d2,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:)=maxindex1;
        maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
                aprimez=aprimeindexes+N_a*bothzind2;
                entireRHS_ii=ReturnMatrix_ii+beta*reshape(EV_d2(aprimez),[N_d1,(maxgap_V(ii)+1),1,N_bothz]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*bothzind2;
        entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_bothz]);
        [Vtemp,maxindexL2alt]=max(entireRHS_L2,[],1);
        Valt_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
        Policy_V_ford2_jj(:,:,d2_c)=shiftdim(maxindexL2alt,1);
        d1_indalt=rem(maxindexL2alt-1,N_d1)+1;
        allindalt=d1_indalt+N_d1*aind+N_d1*N_a*bothzind;
        midpointV_ford2_jj(:,:,d2_c)=squeeze(midpoints_jj(allindalt));
        L2offsetalt = ceil(maxindexL2alt/N_d1);
        linidx_loweralt = d1_indalt                  + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
        linidx_upperalt = d1_indalt + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
        isInfLoweralt    = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
        isInfUpperalt    = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
        inLowerStrictalt = (L2offsetalt >= 2)         & (L2offsetalt <= n2short+1);
        inUpperStrictalt = (L2offsetalt >= n2short+3) & (L2offsetalt <= n2long-1);
        PolicyL2flagV_ford2_jj(:,:,d2_c) = squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt));

        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_d2ii+beta0beta*shiftdim(EV_d2,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,level1ii,:)=maxindex1;
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
                aprimez=aprimeindexes+N_a*bothzind2;
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EV_d2(aprimez),[N_d1,(maxgap(ii)+1),1,N_bothz]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*bothzind2;
        entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_bothz]);
        [Vtemp,maxindex]=max(entireRHS_L2,[],1);

        V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
        Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

        d1_ind=rem(maxindex-1,N_d1)+1;
        allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind;
        midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoints_jj(allind));

        L2offset_d2 = ceil(maxindex/N_d1);
        linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
        linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
        isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
        inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
        inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
        PolicyL2flag_ford2_jj(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
    end

    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2 of Vtilde
    Vtilde(:,:,jj)=V_jj;
    Policy(2,:,:,jj)=shiftdim(maxindex,-1); % d2
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    % V at exponential discounter optimum (full max over d2, d1, aprime)
    [V_jjalt,maxindexalt_d2]=max(Valt_ford2_jj,[],3);
    V(:,:,jj)=V_jjalt;
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy(1,:,:,jj)=shiftdim(rem(d1aprimeL2_ind-1,N_d1)+1,-1);
    Policy(4,:,:,jj)=shiftdim(ceil(d1aprimeL2_ind/N_d1),-1);
    Policy(3,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    PolicyL2flag(1,:,:,jj)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policyalt(2,:,:,jj)=shiftdim(maxindexalt_d2,-1);
    maxindexalt_lin=reshape(maxindexalt_d2,[N_a*N_semiz*N_z,1]);
    d1aprimeL2_ind_alt=reshape(Policy_V_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);
    Policyalt(1,:,:,jj)=shiftdim(rem(d1aprimeL2_ind_alt-1,N_d1)+1,-1);
    Policyalt(4,:,:,jj)=shiftdim(ceil(d1aprimeL2_ind_alt/N_d1),-1);
    Policyalt(3,:,:,jj)=reshape(midpointV_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);
    PolicyL2flagalt(1,:,:,jj)=reshape(PolicyL2flagV_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindexalt_lin-1)),[1,N_a,N_semiz*N_z]);
end

%% Post-process Policy: midpoint -> lower grid point, aprimeL2ind shift
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

adjustalt=(Policyalt(4,:,:,:)<1+n2short+1);
Policyalt(3,:,:,:)=Policyalt(3,:,:,:)-adjustalt;
Policyalt(4,:,:,:)=adjustalt.*Policyalt(4,:,:,:)+(1-adjustalt).*(Policyalt(4,:,:,:)-n2short-1);

Policyalt=[Policyalt;PolicyL2flagalt];

%% Outputs
Valt=V;

end
