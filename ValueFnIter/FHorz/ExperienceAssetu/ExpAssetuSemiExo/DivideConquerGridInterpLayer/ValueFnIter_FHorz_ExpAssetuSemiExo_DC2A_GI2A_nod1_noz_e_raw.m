function [V,Policy]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC2A_GI2A_nod1_noz_e_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_semiz, n_e, n_u, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J, e_gridvals_J, u_gridvals, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% DC2A_GI2A (divide-and-conquer + grid interpolation on a1) nod1 (no d1 decision) noz (no Markov z) _e analog.
% Grafts semiz onto ValueFnIter_FHorz_ExpAsset_DC2A_GI2A_nod1_noz_e_raw (which supplies the divide-conquer + grid-interp math), following the semiz structure of the ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_GI2A_nod1_e_raw sibling with the Markov z dropped.
% d2 determines experience asset (a3), d3 determines semi-exog state (semiz). No d1, no Markov z.
% a1 is the divide-conquered+grid-interpolated standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% semiz is the semi-exogenous state (plays the role of bothz as there is no z); e is i.i.d.
% Policy stores (d2, d3, joint(a1prime-midpoint,a2prime), a1prime-L2index) plus the appended L2flag row.
% lowmemory: 2 shocks {semiz,e} => levels {0,1,2}.
%   =0 vectorise semiz and e; =1 loop e (semiz parallel); =2 outer-loop semiz / inner-loop e.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_u=prod(n_u);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,joint(a1prime-midpoint,a2prime),a1primeL2ind
Policy=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

%%
% For the no-V_Jplus1 (period N_j) section we vectorise over the full d=(d2,d3) grid
n_d23=[n_d2,n_d3];
N_d=prod([n_d2,n_d3]);
d_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate (for the d3-loop sections, which loop over d3 and then max over d3)
V_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy3_ford3_jj=zeros(3,N_a,N_semiz,N_e,N_d3,'gpuArray'); % d2, joint(a1prime-midpoint,a2prime), a1primeL2ind
flag_ford3_jj=2*ones(N_a,N_semiz,N_e,N_d3,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1
eBind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1

pi_u=shiftdim(pi_u,-2); % put u into third dimension
%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity to find the midpoint (divide-and-conquer over a1)
        midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,N_semiz,N_e,'gpuArray');
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
            end
        end
        midpoint=max(min(midpoint,N_a1-1),2);

        % Grid interpolation (second layer)
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*semizBind + N_d*N_a2*N_a*N_semiz*eBind;
        Policy(1,:,:,:,N_j)=rem(d_ind-1,N_d2)+1; % d2
        Policy(2,:,:,:,N_j)=ceil(d_ind/N_d2); % d3
        Policy(3,:,:,:,N_j)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
        Policy(4,:,:,:,N_j)=maxindexL2a1; % a1primeL2ind

        linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*semizBind + N_d*n2long*N_a2*N_a*N_semiz*eBind;
        linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*semizBind + N_d*n2long*N_a2*N_a*N_semiz*eBind;
        isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity to find the midpoint (divide-and-conquer over a1)
            midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii_e,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);

            % Grid interpolation (second layer)
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, special_n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

            allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*semizBind;
            Policy(1,:,:,e_c,N_j)=rem(d_ind-1,N_d2)+1; % d2
            Policy(2,:,:,e_c,N_j)=ceil(d_ind/N_d2); % d3
            Policy(3,:,:,e_c,N_j)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy(4,:,:,e_c,N_j)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*semizBind;
            linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*semizBind;
            isInfLower   =(ReturnMatrix_ii_e(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_e(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity to find the midpoint (divide-and-conquer over a1)
                midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                        [~,maxindex_inner]=max(ReturnMatrix_ii_ze,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);

                % Grid interpolation (second layer)
                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

                allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
                Policy(1,:,z_c,e_c,N_j)=rem(d_ind-1,N_d2)+1; % d2
                Policy(2,:,z_c,e_c,N_j)=ceil(d_ind/N_d2); % d3
                Policy(3,:,z_c,e_c,N_j)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy(4,:,z_c,e_c,N_j)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_ze(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_ze(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            end
        end
    end

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3)); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_semiz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz], indexed by semizprime (d3-independent)
    EV_aprime=squeeze(sum((EV_aprime.*pi_u),3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]
    EV_aprime(isnan(EV_aprime))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            % n-Monotonicity to find the midpoint (divide-and-conquer over a1)
            midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_semiz,N_e,'gpuArray');
            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV;
            [~,maxindex1]=max(entireRHS_d3,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_d3=ReturnMatrix_d3+DiscountedEV(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_d3,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);

            % Grid interpolation (second layer)
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind + N_d2*N_a2*N_a*N_semiz*eBind;
            Policy3_ford3_jj(1,:,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_jj(2,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy3_ford3_jj(3,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind + N_d2*n2long*N_a2*N_a*N_semiz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind + N_d2*n2long*N_a2*N_a*N_semiz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity to find the midpoint (divide-and-conquer over a1)
                midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');
                ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_d3=ReturnMatrix_d3+DiscountedEV;
                [~,maxindex1]=max(entireRHS_d3,[],2);
                midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_d3=ReturnMatrix_d3+DiscountedEV(d2aprimez);
                        [~,maxindex_inner]=max(entireRHS_d3,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);

                % Grid interpolation (second layer)
                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                Policy3_ford3_jj(1,:,:,e_c,d3_c)=d_ind; % d2
                Policy3_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy3_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    % n-Monotonicity to find the midpoint (divide-and-conquer over a1)
                    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
                    ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3=ReturnMatrix_d3+DiscountedEV_z;
                    [~,maxindex1]=max(entireRHS_d3,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_d3,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);

                    % Grid interpolation (second layer)
                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp_z(aprimez),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy3_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind; % d2
                    Policy3_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy3_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d3
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_semiz,N_e]); % d2
    Policy(3,:,:,:,N_j)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_semiz,N_e]); % joint(a1prime midpoint,a2prime)
    Policy(4,:,:,:,N_j)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_semiz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_jj((1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3)); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_semiz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz], indexed by semizprime (d3-independent)
    EV_aprime=squeeze(sum((EV_aprime.*pi_u),3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]
    EV_aprime(isnan(EV_aprime))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            % n-Monotonicity to find the midpoint (divide-and-conquer over a1)
            midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_semiz,N_e,'gpuArray');
            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV;
            [~,maxindex1]=max(entireRHS_d3,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_d3=ReturnMatrix_d3+DiscountedEV(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_d3,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end
            midpoint=max(min(midpoint,N_a1-1),2);

            % Grid interpolation (second layer)
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind + N_d2*N_a2*N_a*N_semiz*eBind;
            Policy3_ford3_jj(1,:,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_jj(2,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy3_ford3_jj(3,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind + N_d2*n2long*N_a2*N_a*N_semiz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind + N_d2*n2long*N_a2*N_a*N_semiz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_jj(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                % n-Monotonicity to find the midpoint (divide-and-conquer over a1)
                midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');
                ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_d3=ReturnMatrix_d3+DiscountedEV;
                [~,maxindex1]=max(entireRHS_d3,[],2);
                midpoint(:,1,:,level1ii,:,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_d3=ReturnMatrix_d3+DiscountedEV(d2aprimez);
                        [~,maxindex_inner]=max(entireRHS_d3,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                    end
                end
                midpoint=max(min(midpoint,N_a1-1),2);

                % Grid interpolation (second layer)
                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                Policy3_ford3_jj(1,:,:,e_c,d3_c)=d_ind; % d2
                Policy3_ford3_jj(2,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy3_ford3_jj(3,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_jj(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    % n-Monotonicity to find the midpoint (divide-and-conquer over a1)
                    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
                    ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3=ReturnMatrix_d3+DiscountedEV_z;
                    [~,maxindex1]=max(entireRHS_d3,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_z(d2aprime);
                            [~,maxindex_inner]=max(entireRHS_d3,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end
                    midpoint=max(min(midpoint,N_a1-1),2);

                    % Grid interpolation (second layer)
                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp_z(aprimez),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy3_ford3_jj(1,:,z_c,e_c,d3_c)=d_ind; % d2
                    Policy3_ford3_jj(2,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy3_ford3_jj(3,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d3
    V(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,jj)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_semiz,N_e]); % d2
    Policy(3,:,:,:,jj)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_semiz,N_e]); % joint(a1prime midpoint,a2prime)
    Policy(4,:,:,:,jj)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_semiz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_jj((1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);

end


%% With grid interpolation, switch from midpoint to lower grid index
% Currently Policy(3,:) is joint(a1prime midpoint,a2prime), and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(3,:) to joint('lower grid point',a2prime) and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust; % decrement a1prime component of the joint (midpoint>=2 keeps it within the same a2prime block)
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy; PolicyL2flag];


end
