function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetsemizN_DC2A_GI2A_e_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Naive quasi-hyperbolic discounting + ExperienceAssetsemiz, two standard endogenous states:
% divide-and-conquer on a1 (DC2A) plus the grid interpolation layer on a1 (GI2A); the remaining
% standard endogenous state a2 is folded (choice a2prime); a3 is the experience asset.
% Naive: two fully independent passes over the same candidate set,
%   Valt/Policyalt maximise  F + beta*EV        (the exponential value; drives the backward recursion)
%   Vtilde/Policy  maximise  F + beta0*beta*EV  (the QH-perceived value; reported as V)
% beta0 is received as a trailing input, beta0beta=beta0*beta.
% The two discount factors generally choose different DC brackets, different GI midpoints, different
% folded a2prime and different d3, so each pass re-derives its own maxgap, loweredge, midpoint,
% a1primeindexesfine and level-2 return matrix, and takes its own max over d3. Only the level-1
% return matrix (which does not involve EV) is shared between the two passes.
% Outputs [Vtilde,Policy,Valt,Policyalt].
% ExperienceAssetsemiz: aprimeFn=aprimeFn(d2,a3,semiz), so the a3prime lottery carries a
% current-semiz dimension and the EV contraction over the shock-prime index is done inside the
% d3 loop (it cannot be hoisted out of it). The QH dual attaches to the continuation that comes
% out of that contraction, exactly where it attaches in the SemiExo sibling.
% experienceassetsemiz DC2A_GI2A: a3prime=aprimeFn(d2,a3,semiz), so aprime depends on
% the current semi-exogenous state. Structure follows
% ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_GI2A_e_raw; the EV pipeline differs
% (see the comment at the aprime setup).
% d1 is any other decision, d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is divide-conquered + grid-interp-layer standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% z is exogenous Markov, semiz is semi-exogenous; bothz=(semiz,z) with semiz varying fastest.
% Policy stores (d1, d2, d3, joint(a1prime,a2prime), a1primeL2ind) with the L2 flag as a 6th row; the joint row is a1prime+N_a1*(a2prime-1), a1prime being the lower grid point.
% lowmemory: 3 shocks {z,semiz,e} => levels {0,1,2,3}.
%   =0 vectorise bothz and e; =1 loop e (bothz parallel); =2 outer-loop z / inner-loop e (semiz parallel); =3 joint bothz outer / inner-loop e.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

Vtilde=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray'); % QH-perceived value (beta0*beta)
Valt=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray'); % exponential value (beta); drives the backward recursion
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,d3,joint(a1prime,a2prime),a1primeL2ind seperately
Policy=zeros(6,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy(6,:,:,:,:)=2; % L2 flag: 1=all to lower, 2=usual, 3=all to upper
Policyalt=zeros(6,N_a,N_semiz*N_z,N_e,N_j,'gpuArray'); % exponential-discounter policy
Policyalt(6,:,:,:,:)=2; % L2 flag for the exponential-discounter policy

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]; maps d12-index to d2-component (used inside the d3 loop where d=d12)

if vfoptions.lowmemory==0
    bothzindB=shiftdim(gpuArray(0:1:N_bothz-1),-1);
    eindB=shiftdim(gpuArray(0:1:N_e-1),-2);
    midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_bothz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    bothzindB=shiftdim(gpuArray(0:1:N_bothz-1),-1);
    midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_bothz,'gpuArray');
elseif vfoptions.lowmemory==2
    special_n_semiz=[n_semiz,ones(1,length(n_z))];
    special_n_e=ones(1,length(n_e));
    semizindB=shiftdim(gpuArray(0:1:N_semiz-1),-1);
    midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_e=ones(1,length(n_e));
    midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);

% Preallocate (for the d3 loop)
V_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_alt=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray'); % d1,d2,joint(a1prime,a2prime),a1primeL2ind
Policy4_ford3_tilde=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray'); % d1,d2,joint(a1prime,a2prime),a1primeL2ind
flag_ford3_alt=2*ones(N_a,N_bothz,N_e,N_d3,'gpuArray');
flag_ford3_tilde=2*ones(N_a,N_bothz,N_e,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzindB + N_d12*N_a2*N_a*N_bothz*eindB;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_alt(3,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind
            linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB + N_d12*n2long*N_a2*N_a*N_bothz*eindB;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB + N_d12*n2long*N_a2*N_a*N_bothz*eindB;
            isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(:,:,:,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzindB;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB;
                isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(:,:,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    V_ford3_alt(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizindB;
                    Policy4_ford3_alt(1,:,zind,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_alt(2,:,zind,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_alt(3,:,zind,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_alt(4,:,zind,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
                    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(:,zind,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(:,z_c,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_alt,[],4); % max over d3
    Vtilde(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz,N_e]); % d1
    Policy(2,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz,N_e]); % joint(a1prime,a2prime)
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    Policy(6,:,:,:,N_j)=reshape(flag_ford3_alt((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);


    % Terminal period: the QH and the exponential discounter coincide
    Valt(:,:,:,N_j)=Vtilde(:,:,:,N_j);
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j);
    Policyalt(6,:,:,:,N_j)=Policy(6,:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a3, n_semiz, d2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex, a3primeProbs are [N_d2,N_a3,N_semiz], indexed by the CURRENT semiz
    % aprime depends only on semiz (the FAST index of bothz), so tile over N_z.

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,N_z); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs_full=repmat(a3primeProbs,N_a1*N_a2,1,N_z);
    % aprime depends on the CURRENT semiz, so (unlike the plain-expasset SemiExo version)
    % the interpolation cannot be hoisted out of the d3 loop: EVpre must be contracted over
    % the shock-prime index first (that contraction depends on d3 via pi_semiz), and only
    % then interpolated. See the d3 loops below.
    shock_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzindB + N_d12*N_a2*N_a*N_bothz*eindB;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_alt(3,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB + N_d12*n2long*N_a2*N_a*N_bothz*eindB;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB + N_d12*n2long*N_a2*N_a*N_bothz*eindB;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(:,:,:,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzindB + N_d12*N_a2*N_a*N_bothz*eindB;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_tilde(3,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_tilde(4,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB + N_d12*n2long*N_a2*N_a*N_bothz*eindB;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB + N_d12*n2long*N_a2*N_a*N_bothz*eindB;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_tilde(:,:,:,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);

                %% Valt (beta)
                entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt(aprime),[N_d12*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzindB;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_bothz]);
                linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(:,:,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                %% Vtilde (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde(aprime),[N_d12*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzindB;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_bothz]);
                linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_tilde(:,:,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,:,:,zind);
                DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,:,:,zind);
                DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,:,:,zind);
                DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,:,:,zind);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);

                    %% Valt (beta)
                    entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_alt_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_alt_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt_z(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford3_alt(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizindB;
                    Policy4_ford3_alt(1,:,zind,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_alt(2,:,zind,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_alt(3,:,zind,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_alt(4,:,zind,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(:,zind,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                    %% Vtilde (beta0*beta)
                    entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_tilde_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde_z(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizindB;
                    Policy4_ford3_tilde(1,:,zind,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_tilde(2,:,zind,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_tilde(3,:,zind,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_tilde(4,:,zind,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(:,zind,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,:,:,z_c);
                DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);

                    %% Valt (beta)
                    entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_alt_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_alt_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt_z(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(:,z_c,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                    %% Vtilde (beta0*beta)
                    entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_tilde_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde_z(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(:,z_c,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d3
    Vtilde(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]); % d1
    Policy(2,:,:,:,N_j)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]); % joint(a1prime,a2prime)
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    Policy(6,:,:,:,N_j)=reshape(flag_ford3_tilde((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);

    % Max over d3 for the exponential-discounter (alt) pass
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],4); % max over d3
    Valt(:,:,:,N_j)=V_jjalt;
    Policyalt(3,:,:,:,N_j)=shiftdim(maxindexalt,-1); % d3 is just maxindexalt
    maxindexalt=reshape(maxindexalt,[N_a*N_bothz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    tempalt=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1) -1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+tempalt),[1,N_a,N_bothz,N_e]); % d1
    Policyalt(2,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+tempalt),[1,N_a,N_bothz,N_e]); % d2
    Policyalt(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+tempalt),[1,N_a,N_bothz,N_e]); % joint(a1prime,a2prime)
    Policyalt(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+tempalt),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    Policyalt(6,:,:,:,N_j)=reshape(flag_ford3_alt((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1)),[1,N_a,N_bothz,N_e]);
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
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3)); % [N_a,N_bothz]  -- continuation is Valt (the exponential value)

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a3, n_semiz, d2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex, a3primeProbs are [N_d2,N_a3,N_semiz], indexed by the CURRENT semiz
    % aprime depends only on semiz (the FAST index of bothz), so tile over N_z.

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,N_z); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs_full=repmat(a3primeProbs,N_a1*N_a2,1,N_z);
    % aprime depends on the CURRENT semiz, so (unlike the plain-expasset SemiExo version)
    % the interpolation cannot be hoisted out of the d3 loop: EVpre must be contracted over
    % the shock-prime index first (that contraction depends on d3 via pi_semiz), and only
    % then interpolated. See the d3 loops below.
    shock_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzindB + N_d12*N_a2*N_a*N_bothz*eindB;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_alt(3,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB + N_d12*n2long*N_a2*N_a*N_bothz*eindB;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB + N_d12*n2long*N_a2*N_a*N_bothz*eindB;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(:,:,:,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzindB + N_d12*N_a2*N_a*N_bothz*eindB;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_tilde(3,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_tilde(4,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB + N_d12*n2long*N_a2*N_a*N_bothz*eindB;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB + N_d12*n2long*N_a2*N_a*N_bothz*eindB;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_tilde(:,:,:,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);

                %% Valt (beta)
                entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt(aprime),[N_d12*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzindB;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_bothz]);
                linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(:,:,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                %% Vtilde (beta0*beta)
                entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde(aprime),[N_d12*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzindB;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_bothz]);
                linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzindB;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_tilde(:,:,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,:,:,zind);
                DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,:,:,zind);
                DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,:,:,zind);
                DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,:,:,zind);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);

                    %% Valt (beta)
                    entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_alt_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_alt_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt_z(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford3_alt(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizindB;
                    Policy4_ford3_alt(1,:,zind,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_alt(2,:,zind,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_alt(3,:,zind,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_alt(4,:,zind,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(:,zind,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                    %% Vtilde (beta0*beta)
                    entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_tilde_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde_z(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizindB;
                    Policy4_ford3_tilde(1,:,zind,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_tilde(2,:,zind,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_tilde(3,:,zind,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_tilde(4,:,zind,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(:,zind,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,:,:,z_c);
                DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);

                    %% Valt (beta)
                    entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_alt_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_alt_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt_z(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(:,z_c,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

                    %% Vtilde (beta0*beta)
                    entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_tilde_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde_z(aprime),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind
                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d12*n2long*N_a2,N_a]);
                    linidx_lower=d_ind                + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                    isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(:,z_c,e_c,d3_c)=squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d3
    Vtilde(:,:,:,jj)=V_jj;
    Policy(3,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]); % d1
    Policy(2,:,:,:,jj)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]); % joint(a1prime,a2prime)
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    Policy(6,:,:,:,jj)=reshape(flag_ford3_tilde((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);

    % Max over d3 for the exponential-discounter (alt) pass
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],4); % max over d3
    Valt(:,:,:,jj)=V_jjalt;
    Policyalt(3,:,:,:,jj)=shiftdim(maxindexalt,-1); % d3 is just maxindexalt
    maxindexalt=reshape(maxindexalt,[N_a*N_bothz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    tempalt=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1) -1);
    Policyalt(1,:,:,:,jj)=reshape(Policy4_ford3_alt(1+tempalt),[1,N_a,N_bothz,N_e]); % d1
    Policyalt(2,:,:,:,jj)=reshape(Policy4_ford3_alt(2+tempalt),[1,N_a,N_bothz,N_e]); % d2
    Policyalt(4,:,:,:,jj)=reshape(Policy4_ford3_alt(3+tempalt),[1,N_a,N_bothz,N_e]); % joint(a1prime,a2prime)
    Policyalt(5,:,:,:,jj)=reshape(Policy4_ford3_alt(4+tempalt),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    Policyalt(6,:,:,:,jj)=reshape(flag_ford3_alt((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1)),[1,N_a,N_bothz,N_e]);
end


%% With grid interpolation, switch from midpoint to lower grid index
% Currently Policy(4,:) holds joint(a1prime midpoint,a2prime) and Policy(5,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if we
% switch the a1prime part of the joint to 'lower grid point' and then have Policy(5,:) counting
% 0:nshort+1 up from this.
adjust=(Policy(5,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-adjust; % a1prime part of joint -> lower grid point (a1prime is the low-order part, stays within its a2prime block)
Policy(5,:,:,:,:)=Policy(5,:,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

adjustalt=(Policyalt(5,:,:,:,:)<1+n2short+1); % is the L2 index below midpoint?
Policyalt(4,:,:,:,:)=Policyalt(4,:,:,:,:)-adjustalt; % a1prime part of joint -> lower grid point (a1prime is the low-order part, stays within its a2prime block)
Policyalt(5,:,:,:,:)=Policyalt(5,:,:,:,:)-(n2short+1)*(~adjustalt); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

end
