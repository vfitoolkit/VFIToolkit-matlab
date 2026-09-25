function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_DC1_GI1_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,n_u,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, u_gridvals, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Naive quasi-hyperbolic discounting + ExperienceAssetu + SemiExo, Divide-and-Conquer (DC1 over a1prime) with grid interpolation layer (GI1) on a1.
% d2 determines the experience asset a2, a1 is the standard endogenous state.
% d3 determines the semi-exogenous state semiz.
% u is an i.i.d. shock on the experience-asset transition (aprimeFn=aprimeFn(d2,a2,u,...)), so
% a2primeIndex/a2primeProbs carry the u dimension; u is integrated out with pi_u inside EV before
% any discounting, so the dual touches only the discount factors, not the lottery.
% Naive: two passes over the same candidate set,
%   Valt/Policyalt maximise  F + beta*EV        (the exponential value)
%   Vtilde/Policy  maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0 is received as a trailing input.
% The two discount factors generally pick different DC brackets AND different GI midpoints, so the beta
% pass uses maxgap_V and the beta0*beta pass uses maxgap, and each pass re-derives its own loweredge,
% midpoint, a1primeindexes(fine) and level-2 return matrix; the level-1 return matrix is shared.
% Each pass also gets its own max over d3.
% Backward EV uses Valt (the exponential continuation value).
% d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% semiz is semi-exog state

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_u=prod(n_u);

Valt=zeros(N_a,N_semiz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,a1prime seperately
Policy=zeros(5,N_a,N_semiz,N_j,'gpuArray');
Policy(5,:,:,:)=2;
Policyalt=zeros(5,N_a,N_semiz,N_j,'gpuArray'); % exponential discounter optimal [d; a1prime midpoint; a1primeL2ind]
Policyalt(5,:,:,:)=2; % L2 flag for the exponential discounter policy

pi_u=shiftdim(pi_u,-2); % put it into third dimension

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory==1
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate
if vfoptions.lowmemory==0
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d2,1,N_a1,N_a2,'gpuArray');
end

% Preallocate
V_ford3_alt=zeros(N_a,N_semiz,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_semiz,N_d3,'gpuArray');
Policy3_ford3_alt=zeros(3,N_a,N_semiz,N_d3,'gpuArray');
Policy3_ford3_tilde=zeros(3,N_a,N_semiz,N_d3,'gpuArray');
flag_ford3_alt=2*ones(1,N_a,N_semiz,N_d3,'gpuArray');
flag_ford3_tilde=2*ones(1,N_a,N_semiz,N_d3,'gpuArray');


% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3); % already includes -1
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    if vfoptions.lowmemory==0

        % Period N_j could be done without looping over d3, but then it needs much more memory than the rest, and since looping for the other periods the runtime cost of looping here is negligible.
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d2-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d2-by-maxgap(ii)+1-by-1-by-n_a2-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=3, Refine=0
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_semiz
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_semiz
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_semiz, d23_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*semizBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_semiz
            Policy3_ford3_alt(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_alt(2,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy3_ford3_alt(3,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset=ceil(maxindexL2/N_d2);
            linidx_lower=d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
            isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
            flag_ford3_alt(1,:,:,d3_c)=shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

        end

    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoint(:,1,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d2-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d2-by-maxgap(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        [~,maxindex]=max(ReturnMatrix_ii_z,[],2);
                        midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                % Turn this into the 'midpoint'
                midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
                Policy3_ford3_alt(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_alt(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3_ford3_alt(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset=ceil(maxindexL2/N_d2);
                linidx_lower=d_ind                   + N_d2*n2long*aind;
                linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
                flag_ford3_alt(1,:,z_c,d3_c)=shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_alt,[],3); % max over d3
    Vtilde(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1) -1);
    Policy(1,:,:,N_j)=reshape(Policy3_ford3_alt(1+temp),[1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(Policy3_ford3_alt(2+temp),[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(Policy3_ford3_alt(3+temp),[1,N_a,N_semiz]);
    flat_idx=(1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1);
    Policy(5,:,:,N_j)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_semiz]);
    % terminal: QH and exponential discounter coincide
    Valt(:,:,N_j)=Vtilde(:,:,N_j);
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j);
    Policyalt(5,:,:,N_j)=Policy(5,:,:,N_j);

else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a2, n_u, d2_gridvals, a2_grid, u_gridvals, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_semiz);  % [N_d2*N_a1,N_a2,N_u,N_semiz]

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,u,semiz), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,u,semiz), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs; % fresh per d3: skipinterp varies with d3_c, so the zeroing must not accumulate
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % QH-perceived
            % Interpolate EV over aprime_grid
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            % n-Monotonicity
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

            %% Valt (beta)

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_alt;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap_V=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                    % loweredge is n_d2-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    % aprime possibilities are n_d2-by-maxgap_V(ii)+1-by-1-by-n_a2-by-n_semiz
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,n_semiz, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=3, Refine=0
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind; % [N_d2,maxgap_V+1,1,N_a2,N_semiz]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2,N_semiz]
                    entireRHS_ii_d3=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end


            % Turn this into the 'midpoint'
            midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_semiz
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_semiz
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
            d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
            entireRHS_ii_d3=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
            [Vtempii,maxindexL2alt]=max(entireRHS_ii_d3,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii,1);
            d_indalt=rem(maxindexL2alt-1,N_d2)+1;
            allindalt=d_indalt+N_d2*aind+N_d2*N_a*semizBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_semiz
            Policy3_ford3_alt(1,:,:,d3_c)=d_indalt; % d2
            Policy3_ford3_alt(2,:,:,d3_c)=shiftdim(squeeze(midpoint(allindalt)),-1); % a1prime midpoint
            Policy3_ford3_alt(3,:,:,d3_c)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offsetalt=ceil(maxindexL2alt/N_d2);
            linidx_loweralt=d_indalt                   + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
            linidx_upperalt=d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
            isInfLoweralt=(ReturnMatrix_L2(linidx_loweralt)==-Inf);
            isInfUpperalt=(ReturnMatrix_L2(linidx_upperalt)==-Inf);
            inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
            inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
            flag_ford3_alt(1,:,:,d3_c)=shiftdim(squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)),-1);

            %% Vtilde (beta0*beta)

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_tilde;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d2-by-1-by-n_a2-by-1-by-n_a2-by-n_semiz
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d2-by-maxgap(ii)+1-by-1-by-n_a2-by-n_semiz
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=3, Refine=0
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind; % [N_d2,maxgap+1,1,N_a2,N_semiz]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2,N_semiz]
                    entireRHS_ii_d3=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end


            % Turn this into the 'midpoint'
            midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_semiz
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_semiz
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
            d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
            entireRHS_ii_d3=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*semizBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_semiz
            Policy3_ford3_tilde(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_tilde(2,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy3_ford3_tilde(3,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset=ceil(maxindexL2/N_d2);
            linidx_lower=d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
            isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
            flag_ford3_tilde(1,:,:,d3_c)=shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,u,semiz), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,u,semiz), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs; % fresh per d3: skipinterp varies with d3_c, so the zeroing must not accumulate
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % QH-perceived
            % Interpolate EV over aprime_grid
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);


            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,z_c);
                DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,z_c);
                DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,z_c);
                DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,z_c);

                % n-Monotonicity
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                %% Valt (beta)

                entireRHS_ii_d3z=ReturnMatrix_ii_z+DiscountedEV_alt_z;

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_d3z,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoint(:,1,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                        % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind; % [N_d2,maxgap_V+1,1,N_a2]; linear index into DiscountedEV_alt_z [N_d2,N_a1,1,N_a2]
                        entireRHS_ii_d3z=ReturnMatrix_ii_dc+DiscountedEV_alt_z(d2aprime);
                        [~,maxindex]=max(entireRHS_ii_d3z,[],2);
                        midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end


                % Turn this into the 'midpoint'
                midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
                d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt_z(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2alt]=max(entireRHS_ii_d3z,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_indalt=rem(maxindexL2alt-1,N_d2)+1;
                allindalt=d_indalt+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
                Policy3_ford3_alt(1,:,z_c,d3_c)=d_indalt; % d2
                Policy3_ford3_alt(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint(allindalt)),-1); % a1prime midpoint
                Policy3_ford3_alt(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offsetalt=ceil(maxindexL2alt/N_d2);
                linidx_loweralt=d_indalt                   + N_d2*n2long*aind;
                linidx_upperalt=d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind;
                isInfLoweralt=(ReturnMatrix_L2(linidx_loweralt)==-Inf);
                isInfUpperalt=(ReturnMatrix_L2(linidx_upperalt)==-Inf);
                inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
                inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
                flag_ford3_alt(1,:,z_c,d3_c)=shiftdim(squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)),-1);

                %% Vtilde (beta0*beta)

                entireRHS_ii_d3z=ReturnMatrix_ii_z+DiscountedEV_tilde_z;

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_d3z,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoint(:,1,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind; % [N_d2,maxgap+1,1,N_a2]; linear index into DiscountedEV_tilde_z [N_d2,N_a1,1,N_a2]
                        entireRHS_ii_d3z=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(d2aprime);
                        [~,maxindex]=max(entireRHS_ii_d3z,[],2);
                        midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end


                % Turn this into the 'midpoint'
                midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
                d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde_z(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3z,[],1);
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
                Policy3_ford3_tilde(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_tilde(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3_ford3_tilde(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset=ceil(maxindexL2/N_d2);
                linidx_lower=d_ind                   + N_d2*n2long*aind;
                linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
                flag_ford3_tilde(1,:,z_c,d3_c)=shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1) -1);
    Policy(1,:,:,N_j)=reshape(Policy3_ford3_tilde(1+temp),[1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(Policy3_ford3_tilde(2+temp),[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(Policy3_ford3_tilde(3+temp),[1,N_a,N_semiz]);
    flat_idx=(1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1);
    Policy(5,:,:,N_j)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_semiz]);

    % Max over d3 for alt (the exponential discounter) -> Valt/Policyalt
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,N_j)=V_jjalt;
    Policyalt(2,:,:,N_j)=shiftdim(maxindexalt,-1); % d3 is just maxindexalt
    maxindexalt=reshape(maxindexalt,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    tempalt=3*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt-1) -1);
    Policyalt(1,:,:,N_j)=reshape(Policy3_ford3_alt(1+tempalt),[1,N_a,N_semiz]);
    Policyalt(3,:,:,N_j)=reshape(Policy3_ford3_alt(2+tempalt),[1,N_a,N_semiz]);
    Policyalt(4,:,:,N_j)=reshape(Policy3_ford3_alt(3+tempalt),[1,N_a,N_semiz]);
    flat_idxalt=(1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt-1);
    Policyalt(5,:,:,N_j)=reshape(flag_ford3_alt(flat_idxalt),[1,N_a,N_semiz]);

end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a2, n_u, d2_gridvals, a2_grid, u_gridvals, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_semiz);  % [N_d2*N_a1,N_a2,N_u,N_semiz]

    EVpre=Valt(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,u,semiz), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,u,semiz), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs; % fresh per d3: skipinterp varies with d3_c, so the zeroing must not accumulate
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % QH-perceived
            % Interpolate EV over aprime_grid
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

            % n-Monotonicity
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % Level=1, Refine=0

            %% Valt (beta)

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_alt;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap_V=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,n_semiz, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level=3, Refine=0
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind; % [N_d2,maxgap_V+1,1,N_a2,N_semiz]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2,N_semiz]
                    entireRHS_ii_d3=ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end


            % Turn this into the 'midpoint'
            midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_semiz
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_semiz
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
            d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
            entireRHS_ii_d3=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
            [Vtempii,maxindexL2alt]=max(entireRHS_ii_d3,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii,1);
            d_indalt=rem(maxindexL2alt-1,N_d2)+1;
            allindalt=d_indalt+N_d2*aind+N_d2*N_a*semizBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_semiz
            Policy3_ford3_alt(1,:,:,d3_c)=d_indalt; % d2
            Policy3_ford3_alt(2,:,:,d3_c)=shiftdim(squeeze(midpoint(allindalt)),-1); % a1prime midpoint
            Policy3_ford3_alt(3,:,:,d3_c)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offsetalt=ceil(maxindexL2alt/N_d2);
            linidx_loweralt=d_indalt                   + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
            linidx_upperalt=d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
            isInfLoweralt=(ReturnMatrix_L2(linidx_loweralt)==-Inf);
            isInfUpperalt=(ReturnMatrix_L2(linidx_upperalt)==-Inf);
            inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
            inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
            flag_ford3_alt(1,:,:,d3_c)=shiftdim(squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)),-1);

            %% Vtilde (beta0*beta)

            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_tilde;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level=3, Refine=0
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind; % [N_d2,maxgap+1,1,N_a2,N_semiz]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2,N_semiz]
                    entireRHS_ii_d3=ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end


            % Turn this into the 'midpoint'
            midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_semiz
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_semiz
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
            d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
            entireRHS_ii_d3=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*semizBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_semiz
            Policy3_ford3_tilde(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_tilde(2,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy3_ford3_tilde(3,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset=ceil(maxindexL2/N_d2);
            linidx_lower=d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
            linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
            isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
            flag_ford3_tilde(1,:,:,d3_c)=shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,u,semiz), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,u,semiz), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs; % fresh per d3: skipinterp varies with d3_c, so the zeroing must not accumulate
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_semiz]); % QH-perceived
            % Interpolate EV over aprime_grid
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);


            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,z_c);
                DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,z_c);
                DiscountedEVinterp_alt_z=DiscountedEVinterp_alt(:,:,:,:,z_c);
                DiscountedEVinterp_tilde_z=DiscountedEVinterp_tilde(:,:,:,:,z_c);

                % n-Monotonicity
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                %% Valt (beta)

                entireRHS_ii_d3z=ReturnMatrix_ii_z+DiscountedEV_alt_z;

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_d3z,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoint(:,1,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                        % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind; % [N_d2,maxgap_V+1,1,N_a2]; linear index into DiscountedEV_alt_z [N_d2,N_a1,1,N_a2]
                        entireRHS_ii_d3z=ReturnMatrix_ii_dc+DiscountedEV_alt_z(d2aprime);
                        [~,maxindex]=max(entireRHS_ii_d3z,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end


                % Turn this into the 'midpoint'
                midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
                d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z=ReturnMatrix_L2+reshape(DiscountedEVinterp_alt_z(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2alt]=max(entireRHS_ii_d3z,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_indalt=rem(maxindexL2alt-1,N_d2)+1;
                allindalt=d_indalt+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
                Policy3_ford3_alt(1,:,z_c,d3_c)=d_indalt; % d2
                Policy3_ford3_alt(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint(allindalt)),-1); % a1prime midpoint
                Policy3_ford3_alt(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2alt/N_d2),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offsetalt=ceil(maxindexL2alt/N_d2);
                linidx_loweralt=d_indalt                   + N_d2*n2long*aind;
                linidx_upperalt=d_indalt + N_d2*(n2long-1) + N_d2*n2long*aind;
                isInfLoweralt=(ReturnMatrix_L2(linidx_loweralt)==-Inf);
                isInfUpperalt=(ReturnMatrix_L2(linidx_upperalt)==-Inf);
                inLowerStrictalt=(L2offsetalt>=2)         & (L2offsetalt<=n2short+1);
                inUpperStrictalt=(L2offsetalt>=n2short+3) & (L2offsetalt<=n2long-1);
                flag_ford3_alt(1,:,z_c,d3_c)=shiftdim(squeeze(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt)),-1);

                %% Vtilde (beta0*beta)

                entireRHS_ii_d3z=ReturnMatrix_ii_z+DiscountedEV_tilde_z;

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_d3z,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoint(:,1,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind; % [N_d2,maxgap+1,1,N_a2]; linear index into DiscountedEV_tilde_z [N_d2,N_a1,1,N_a2]
                        entireRHS_ii_d3z=ReturnMatrix_ii_dc+DiscountedEV_tilde_z(d2aprime);
                        [~,maxindex]=max(entireRHS_ii_d3z,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end


                % Turn this into the 'midpoint'
                midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
                d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde_z(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3z,[],1);
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
                Policy3_ford3_tilde(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_tilde(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3_ford3_tilde(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset=ceil(maxindexL2/N_d2);
                linidx_lower=d_ind                   + N_d2*n2long*aind;
                linidx_upper=d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                isInfLower=(ReturnMatrix_L2(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_L2(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)         & (L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3) & (L2offset<=n2long-1);
                flag_ford3_tilde(1,:,z_c,d3_c)=shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,jj)=V_jj;
    Policy(2,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1) -1);
    Policy(1,:,:,jj)=reshape(Policy3_ford3_tilde(1+temp),[1,N_a,N_semiz]);
    Policy(3,:,:,jj)=reshape(Policy3_ford3_tilde(2+temp),[1,N_a,N_semiz]);
    Policy(4,:,:,jj)=reshape(Policy3_ford3_tilde(3+temp),[1,N_a,N_semiz]);
    flat_idx=(1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1);
    Policy(5,:,:,jj)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_semiz]);

    % Max over d3 for alt (the exponential discounter) -> Valt/Policyalt
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,jj)=V_jjalt;
    Policyalt(2,:,:,jj)=shiftdim(maxindexalt,-1); % d3 is just maxindexalt
    maxindexalt=reshape(maxindexalt,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    tempalt=3*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt-1) -1);
    Policyalt(1,:,:,jj)=reshape(Policy3_ford3_alt(1+tempalt),[1,N_a,N_semiz]);
    Policyalt(3,:,:,jj)=reshape(Policy3_ford3_alt(2+tempalt),[1,N_a,N_semiz]);
    Policyalt(4,:,:,jj)=reshape(Policy3_ford3_alt(3+tempalt),[1,N_a,N_semiz]);
    flat_idxalt=(1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt-1);
    Policyalt(5,:,:,jj)=reshape(flag_ford3_alt(flat_idxalt),[1,N_a,N_semiz]);

end



%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust; % lower grid point
Policy(4,:,:,:)=Policy(4,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

adjustalt=(Policyalt(4,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policyalt(3,:,:,:)=Policyalt(3,:,:,:)-adjustalt; % lower grid point
Policyalt(4,:,:,:)=Policyalt(4,:,:,:)-(n2short+1)*(~adjustalt); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

end
