function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC2A_GI2A_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, u_gridvals, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAssetu_DC2A_GI2A_raw.
% aprime(d2,a3,u) via CreateExperienceAssetuFnMatrix.
% a1=DC'd + grid-interpolated standard endogenous state, a2=folded standard
% endogenous state(s), a3=experience asset, u=i.i.d. shock on the a3 transition.
% GPU only.
%
% Sophisticated: Vhat_j      = max u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EVfine_at_optimal_choice
% EVfine is the (undiscounted) interpolated continuation actually added to the layer-2
% RHS, so the a3 lottery is already baked in and the gather needs no lottery handling.
% lowmemory=0 full; lowmemory=1 loop z.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_u=prod(n_u);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);

Vhat=zeros(N_a,N_z,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_j,'gpuArray'); % exponential value at the QH policy
Policy=zeros(5,N_a,N_z,N_j,'gpuArray');
Policy(5,:,:,:)=2;

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
else
    aind=gpuArray(0:1:N_a-1);
    zindB=shiftdim(gpuArray(0:1:N_z-1),-1);
end

% d2-component of each d slot.
d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d, 1]

% Preallocate midpoint (filled by DC coarse pass)
if vfoptions.lowmemory==0
    midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,N_z,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
end

% n-Monotonicity over a1
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% GI grid
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

pi_u=shiftdim(pi_u,-2); % put u into third dimension

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Vhat(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,N_j)=maxindexL2a1;

        linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        Policy(5,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        aind_z=gpuArray(0:1:N_a-1);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);

            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii_z,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
            Vhat(:,z_c,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

            allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind_z;
            Policy(1,:,z_c,N_j)=d_ind;
            Policy(2,:,z_c,N_j)=midpoint(allind);
            Policy(3,:,z_c,N_j)=maxindexL2a2;
            Policy(4,:,z_c,N_j)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind_z;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind_z;
            isInfLower=(ReturnMatrix_ii_z(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_z(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            Policy(5,:,z_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end

    Vunderbar(:,:,N_j)=Vhat(:,:,N_j); % terminal: no continuation, so Vunderbar equals Vhat

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    EV=squeeze(sum((EV.*pi_u),3)); % integrate out u
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,3));

    entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]); % undiscounted; beta/beta0beta applied at use sites
    entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        % --- Vhat search (beta0*beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*entireEV(d2aprimez);
                [~,maxindex_inner]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
        aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        EVfine=reshape(entireEVinterp(aprimez),[N_d*n2long*N_a2,N_a,N_z]);
        ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d*n2long*N_a2,N_a,N_z]);
        entireRHS_ii=ReturnMatrix_ii_flat+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vhat(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=midpoint(allind);
        Policy(3,:,:,N_j)=maxindexL2a2;
        Policy(4,:,:,N_j)=maxindexL2a1;

        linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        Policy(5,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        % --- Vunderbar: exponential value at the QH-chosen (interpolated) point ---
        linidx=reshape(maxindexL2,[1,N_a*N_z])+size(EVfine,1)*(0:N_a*N_z-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_z]);
        Vunderbar(:,:,N_j)=Vhat(:,:,N_j)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        aind_z=gpuArray(0:1:N_a-1);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            entireEV_z=entireEV(:,:,:,:,:,:,z_c);
            entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);

            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            % --- Vhat search (beta0*beta) ---
            entireRHS_ii_z=ReturnMatrix_ii_z+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    d2aprime_z=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_z=ReturnMatrix_ii_z+beta0beta*entireEV_z(d2aprime_z);
                    [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprime_z=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            EVfine_z=reshape(entireEVinterp_z(aprime_z),[N_d*n2long*N_a2,N_a]);
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_z,[N_d*n2long*N_a2,N_a]);
            entireRHS_ii_z=ReturnMatrix_ii_flat+beta0beta*EVfine_z;
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            Vhat(:,z_c,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

            allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind_z;
            Policy(1,:,z_c,N_j)=d_ind;
            Policy(2,:,z_c,N_j)=midpoint(allind);
            Policy(3,:,z_c,N_j)=maxindexL2a2;
            Policy(4,:,z_c,N_j)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind_z;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind_z;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            Policy(5,:,z_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            % --- Vunderbar: exponential value at the QH-chosen (interpolated) point ---
            linidx_z=reshape(maxindexL2,[1,N_a])+size(EVfine_z,1)*(0:N_a-1);
            EV_at_policy_z=reshape(EVfine_z(linidx_z),[N_a,1]);
            Vunderbar(:,z_c,N_j)=Vhat(:,z_c,N_j)+(beta-beta0beta)*EV_at_policy_z;
        end
    end
end


%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_z);

    Vlower=reshape(Vunderbar(aprimeIndex(:),:,jj+1),    [N_d2*N_a1*N_a2,N_a3,N_u,N_z]);
    Vupper=reshape(Vunderbar(aprimeplus1Index(:),:,jj+1),[N_d2*N_a1*N_a2,N_a3,N_u,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    EV=squeeze(sum((EV.*pi_u),3)); % integrate out u
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,3));

    entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]); % undiscounted; beta/beta0beta applied at use sites
    entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        % --- Vhat search (beta0*beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*entireEV(d2aprimez);
                [~,maxindex_inner]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
        aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
        EVfine=reshape(entireEVinterp(aprimez),[N_d*n2long*N_a2,N_a,N_z]);
        ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d*n2long*N_a2,N_a,N_z]);
        entireRHS_ii=ReturnMatrix_ii_flat+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        Vhat(:,:,jj)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*zindB;
        Policy(1,:,:,jj)=d_ind;
        Policy(2,:,:,jj)=midpoint(allind);
        Policy(3,:,:,jj)=maxindexL2a2;
        Policy(4,:,:,jj)=maxindexL2a1;

        linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*zindB;
        isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        Policy(5,:,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        % --- Vunderbar: exponential value at the QH-chosen (interpolated) point ---
        linidx=reshape(maxindexL2,[1,N_a*N_z])+size(EVfine,1)*(0:N_a*N_z-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_z]);
        Vunderbar(:,:,jj)=Vhat(:,:,jj)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        aind_z=gpuArray(0:1:N_a-1);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            entireEV_z=entireEV(:,:,:,:,:,:,z_c);
            entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);

            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            % --- Vhat search (beta0*beta) ---
            entireRHS_ii_z=ReturnMatrix_ii_z+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    d2aprime_z=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_z=ReturnMatrix_ii_z+beta0beta*entireEV_z(d2aprime_z);
                    [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, special_n_z, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
            aprime_z=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
            EVfine_z=reshape(entireEVinterp_z(aprime_z),[N_d*n2long*N_a2,N_a]);
            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_z,[N_d*n2long*N_a2,N_a]);
            entireRHS_ii_z=ReturnMatrix_ii_flat+beta0beta*EVfine_z;
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            Vhat(:,z_c,jj)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

            allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind_z;
            Policy(1,:,z_c,jj)=d_ind;
            Policy(2,:,z_c,jj)=midpoint(allind);
            Policy(3,:,z_c,jj)=maxindexL2a2;
            Policy(4,:,z_c,jj)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind_z;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind_z;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            Policy(5,:,z_c,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            % --- Vunderbar: exponential value at the QH-chosen (interpolated) point ---
            linidx_z=reshape(maxindexL2,[1,N_a])+size(EVfine_z,1)*(0:N_a-1);
            EV_at_policy_z=reshape(EVfine_z(linidx_z),[N_a,1]);
            Vunderbar(:,z_c,jj)=Vhat(:,z_c,jj)+(beta-beta0beta)*EV_at_policy_z;
        end
    end
end


%% Post-process
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(4,:,:,:)=Policy(4,:,:,:)-(n2short+1)*(~adjust);

end
