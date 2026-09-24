function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetSemiExoN_DC2A_GI2A_nod1_noz_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_semiz, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Naive quasi-hyperbolic discounting + ExperienceAsset + SemiExo, two standard endogenous states:
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
% SemiExo graft of ValueFnIter_FHorz_ExpAsset_DC2A_GI2A_nod1_noz_raw (no d1, no exogenous z, no e; only shock is semiz).
% DC on first standard endo state a1 (divide-conquer), folded standard middle endo state a2, experience asset a3, plus the grid interpolation layer fine pass.
% d2 determines experience asset (a3); d3 determines semi-exog state (semiz).
% bothz=semiz (there is no z). Policy rows: 1=d2, 2=d3, 3=joint(a1prime midpoint,a2prime), 4=a1prime L2; the L2 flag as 5th.
% lowmemory: 1 shock {semiz} => levels {0,1}.
%   =0 vectorise semiz; =1 loop semiz.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);

Vtilde=zeros(N_a,N_semiz,N_j,'gpuArray'); % QH-perceived value (beta0*beta)
Valt=zeros(N_a,N_semiz,N_j,'gpuArray'); % exponential value (beta); drives the backward recursion
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,joint(a1prime,a2prime),a1prime L2 seperately
Policy=zeros(5,N_a,N_semiz,N_j,'gpuArray');
Policy(5,:,:,:)=2; % L2 flag: 1=all to lower, 2=usual, 3=all to upper
Policyalt=zeros(5,N_a,N_semiz,N_j,'gpuArray'); % exponential-discounter policy
Policyalt(5,:,:,:)=2; % L2 flag for the exponential-discounter policy

%%
aind=gpuArray(0:1:N_a-1); % already includes -1
if vfoptions.lowmemory==0
    semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % at dim 3 of [1,N_a,N_semiz]
elseif vfoptions.lowmemory==1
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate midpoint (filled by DC coarse pass, then used for GI fine pass)
if vfoptions.lowmemory==0
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d2,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');
end

% n-Monotonicity over a1 (the DC dim)
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% GI grid
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% Preallocate (for the max-over-d3 assembly, which loops over d3)
V_ford3_alt=zeros(N_a,N_semiz,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_semiz,N_d3,'gpuArray');
d2_ford3_alt=zeros(N_a,N_semiz,N_d3,'gpuArray');
d2_ford3_tilde=zeros(N_a,N_semiz,N_d3,'gpuArray');
mid_ford3_alt=zeros(N_a,N_semiz,N_d3,'gpuArray');
mid_ford3_tilde=zeros(N_a,N_semiz,N_d3,'gpuArray');
L2a1_ford3_alt=zeros(N_a,N_semiz,N_d3,'gpuArray');
L2a1_ford3_tilde=zeros(N_a,N_semiz,N_d3,'gpuArray');
L2flag_ford3_alt=2*ones(N_a,N_semiz,N_d3,'gpuArray');
L2flag_ford3_tilde=2*ones(N_a,N_semiz,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            % --- DC coarse: Level 1 at level1ii nodes ---
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

            % --- DC coarse: Level 2 narrow band, fill midpoint ---
            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end

            % --- GI fine pass ---
            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizind;
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii,1);
            d2_ford3_alt(:,:,d3_c)=shiftdim(d_ind,1);
            mid_ford3_alt(:,:,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
            L2a1_ford3_alt(:,:,d3_c)=shiftdim(maxindexL2a1,1);

            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizind;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizind;
            isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            L2flag_ford3_alt(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        [~,maxindex_inner]=max(ReturnMatrix_ii_z,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d2_ford3_alt(:,z_c,d3_c)=shiftdim(d_ind,1);
                mid_ford3_alt(:,z_c,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_alt(:,z_c,d3_c)=shiftdim(maxindexL2a1,1);

                linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_z(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_z(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_alt(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_alt,[],3); % max over d3
    Vtilde(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    M=N_a*N_semiz;
    maxlin=reshape(maxindex,[M,1]);
    idx=(1:1:M)'+M*(maxlin-1);
    Policy(1,:,:,N_j)=reshape(d2_ford3_alt(idx),[1,N_a,N_semiz]); % d2
    Policy(3,:,:,N_j)=reshape(mid_ford3_alt(idx),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policy(4,:,:,N_j)=reshape(L2a1_ford3_alt(idx),[1,N_a,N_semiz]); % a1prime L2
    Policy(5,:,:,N_j)=reshape(L2flag_ford3_alt(idx),[1,N_a,N_semiz]);


    % Terminal period: the QH and the exponential discounter coincide
    Valt(:,:,N_j)=Vtilde(:,:,N_j);
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j);
    Policyalt(5,:,:,N_j)=Policy(5,:,:,N_j);
else
    % vfoptions.V_Jplus1 provided
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a3, d2_gridvals, a3_grid, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,N_semiz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_semiz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_semiz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz], indexed by semizprime (d3-independent)
    EV_aprime(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV_aprime(aprimeProbs==1)=Vlower(aprimeProbs==1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz_d3',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_alt;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt(aprimez),[N_d2*n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizind;
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii,1);
            d2_ford3_alt(:,:,d3_c)=shiftdim(d_ind,1);
            mid_ford3_alt(:,:,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
            L2a1_ford3_alt(:,:,d3_c)=shiftdim(maxindexL2a1,1);

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_semiz]);
            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizind;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            L2flag_ford3_alt(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde(aprimez),[N_d2*n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizind;
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii,1);
            d2_ford3_tilde(:,:,d3_c)=shiftdim(d_ind,1);
            mid_ford3_tilde(:,:,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
            L2a1_ford3_tilde(:,:,d3_c)=shiftdim(maxindexL2a1,1);

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_semiz]);
            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizind;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            L2flag_ford3_tilde(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz_d3',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,:,:,z_c);
                DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,:,:,z_c);

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);

                %% Valt (beta)
                entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_alt_z;
                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_z=ReturnMatrix_ii_dc+DiscountedEV_alt_z(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_z=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt_z(aprime),[N_d2*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d2_ford3_alt(:,z_c,d3_c)=shiftdim(d_ind,1);
                mid_ford3_alt(:,z_c,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_alt(:,z_c,d3_c)=shiftdim(maxindexL2a1,1);

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a]);
                linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_alt(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                %% Vtilde (beta0*beta)
                entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_tilde_z;
                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_z=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_z=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde_z(aprime),[N_d2*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d2_ford3_tilde(:,z_c,d3_c)=shiftdim(d_ind,1);
                mid_ford3_tilde(:,z_c,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_tilde(:,z_c,d3_c)=shiftdim(maxindexL2a1,1);

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a]);
                linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_tilde(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    M=N_a*N_semiz;
    maxlin=reshape(maxindex,[M,1]);
    idx=(1:1:M)'+M*(maxlin-1);
    Policy(1,:,:,N_j)=reshape(d2_ford3_tilde(idx),[1,N_a,N_semiz]); % d2
    Policy(3,:,:,N_j)=reshape(mid_ford3_tilde(idx),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policy(4,:,:,N_j)=reshape(L2a1_ford3_tilde(idx),[1,N_a,N_semiz]); % a1prime L2
    Policy(5,:,:,N_j)=reshape(L2flag_ford3_tilde(idx),[1,N_a,N_semiz]);

    % Max over d3 for the exponential-discounter (alt) pass
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,N_j)=V_jjalt;
    Policyalt(2,:,:,N_j)=shiftdim(maxindexalt,-1); % d3 is just maxindexalt
    Malt=N_a*N_semiz;
    maxlinalt=reshape(maxindexalt,[Malt,1]);
    idxalt=(1:1:Malt)'+Malt*(maxlinalt-1);
    Policyalt(1,:,:,N_j)=reshape(d2_ford3_alt(idxalt),[1,N_a,N_semiz]); % d2
    Policyalt(3,:,:,N_j)=reshape(mid_ford3_alt(idxalt),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policyalt(4,:,:,N_j)=reshape(L2a1_ford3_alt(idxalt),[1,N_a,N_semiz]); % a1prime L2
    Policyalt(5,:,:,N_j)=reshape(L2flag_ford3_alt(idxalt),[1,N_a,N_semiz]);
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

    EVpre=Valt(:,:,jj+1); % [N_a,N_semiz]  -- continuation is Valt (the exponential value)

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a3, d2_gridvals, a3_grid, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,N_semiz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_semiz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_semiz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz], indexed by semizprime (d3-independent)
    EV_aprime(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV_aprime(aprimeProbs==1)=Vlower(aprimeProbs==1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz_d3',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_alt;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt(aprimez),[N_d2*n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizind;
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii,1);
            d2_ford3_alt(:,:,d3_c)=shiftdim(d_ind,1);
            mid_ford3_alt(:,:,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
            L2a1_ford3_alt(:,:,d3_c)=shiftdim(maxindexL2a1,1);

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_semiz]);
            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizind;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            L2flag_ford3_alt(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde(aprimez),[N_d2*n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizind;
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii,1);
            d2_ford3_tilde(:,:,d3_c)=shiftdim(d_ind,1);
            mid_ford3_tilde(:,:,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
            L2a1_ford3_tilde(:,:,d3_c)=shiftdim(maxindexL2a1,1);

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a,N_semiz]);
            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizind;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            L2flag_ford3_tilde(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz_d3',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % QH-perceived
            DiscountedEVinterp_alt=permute(interp1(a1_grid,permute(DiscountedEV_alt,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            DiscountedEVinterp_tilde=permute(interp1(a1_grid,permute(DiscountedEV_tilde,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,:,:,z_c);
                DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,:,:,z_c);

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);

                %% Valt (beta)
                entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_alt_z;
                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_z=ReturnMatrix_ii_dc+DiscountedEV_alt_z(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_z=reshape(ReturnMatrix_L2+DiscountedEVinterp_alt_z(aprime),[N_d2*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d2_ford3_alt(:,z_c,d3_c)=shiftdim(d_ind,1);
                mid_ford3_alt(:,z_c,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_alt(:,z_c,d3_c)=shiftdim(maxindexL2a1,1);

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a]);
                linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_alt(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                %% Vtilde (beta0*beta)
                entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_tilde_z;
                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_z=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_z=reshape(ReturnMatrix_L2+DiscountedEVinterp_tilde_z(aprime),[N_d2*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d2_ford3_tilde(:,z_c,d3_c)=shiftdim(d_ind,1);
                mid_ford3_tilde(:,z_c,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_tilde(:,z_c,d3_c)=shiftdim(maxindexL2a1,1);

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_L2,[N_d2*n2long*N_a2,N_a]);
                linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_tilde(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,jj)=V_jj;
    Policy(2,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    M=N_a*N_semiz;
    maxlin=reshape(maxindex,[M,1]);
    idx=(1:1:M)'+M*(maxlin-1);
    Policy(1,:,:,jj)=reshape(d2_ford3_tilde(idx),[1,N_a,N_semiz]); % d2
    Policy(3,:,:,jj)=reshape(mid_ford3_tilde(idx),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policy(4,:,:,jj)=reshape(L2a1_ford3_tilde(idx),[1,N_a,N_semiz]); % a1prime L2
    Policy(5,:,:,jj)=reshape(L2flag_ford3_tilde(idx),[1,N_a,N_semiz]);

    % Max over d3 for the exponential-discounter (alt) pass
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,jj)=V_jjalt;
    Policyalt(2,:,:,jj)=shiftdim(maxindexalt,-1); % d3 is just maxindexalt
    Malt=N_a*N_semiz;
    maxlinalt=reshape(maxindexalt,[Malt,1]);
    idxalt=(1:1:Malt)'+Malt*(maxlinalt-1);
    Policyalt(1,:,:,jj)=reshape(d2_ford3_alt(idxalt),[1,N_a,N_semiz]); % d2
    Policyalt(3,:,:,jj)=reshape(mid_ford3_alt(idxalt),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policyalt(4,:,:,jj)=reshape(L2a1_ford3_alt(idxalt),[1,N_a,N_semiz]); % a1prime L2
    Policyalt(5,:,:,jj)=reshape(L2flag_ford3_alt(idxalt),[1,N_a,N_semiz]);

end


%% Post-process: convert "midpoint + L2 offset" into "lower coarse point + L2 ratio"
% Currently Policy(3,:) is joint(a1prime midpoint,a2prime), Policy(4,:) is the L2 index (ranges -n2short-1:1:1+n2short).
% Switch Policy(3,:) to joint(lower a1prime grid point,a2prime), and Policy(4,:) to a 1..(n2short+2) offset.
adjust=(Policy(4,:,:,:)<1+n2short+1); % is the L2 index below midpoint?
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust; % decrement a1prime component of the joint (midpoint>=2 keeps it within the same a2prime block)
Policy(4,:,:,:)=Policy(4,:,:,:)-(n2short+1)*(~adjust);

adjustalt=(Policyalt(4,:,:,:)<1+n2short+1); % is the L2 index below midpoint?
Policyalt(3,:,:,:)=Policyalt(3,:,:,:)-adjustalt; % decrement a1prime component of the joint (midpoint>=2 keeps it within the same a2prime block)
Policyalt(4,:,:,:)=Policyalt(4,:,:,:)-(n2short+1)*(~adjustalt);

end
