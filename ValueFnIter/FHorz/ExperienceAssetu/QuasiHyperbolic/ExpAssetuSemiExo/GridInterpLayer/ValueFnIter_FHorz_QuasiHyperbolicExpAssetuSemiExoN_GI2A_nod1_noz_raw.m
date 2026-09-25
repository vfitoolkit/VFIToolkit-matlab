function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_GI2A_nod1_noz_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_semiz, n_u, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J, u_gridvals, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAssetuSemiExo_GI2A_nod1_noz_raw.
% ExperienceAsset + semi-exogenous shocks, with the grid interpolation layer on a1. GPU only.
%
% Naive:  Valt_j   = max_{d,a1'} u + beta*E[Valt_{j+1}]         (exponential discounter)
%         Vtilde_j = max_{d,a1'} u + beta_0*beta*E[Valt_{j+1}]  (agent's perceived choice)
% The two discount factors generally pick different GI midpoints, so each pass re-derives its
% own midpoint and its own layer-2 return matrix, and each keeps its own d3 choice, its own
% (midpoint, L2 index) policy pair and its own L2 flag.
% beta0 is received as a trailing input.
% SemiExo graft of ValueFnIter_FHorz_ExpAsset_GI2A_nod1_noz_raw (no d1, no exogenous z, no e; only shock is semiz).
% d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is grid-interpolated standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% bothz=semiz (there is no z). Policy stores (d2,d3,joint(a1prime,a2prime)); the 3rd row is a1prime_lower+N_a1*(a2prime-1).
% Two extra rows are appended for the grid-interp layer: a1prime L2 index and L2flag.
% lowmemory: 1 shock {semiz} => levels {0,1}.
%   =0 vectorise semiz; =1 loop semiz.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_u=prod(n_u);
N_semiz=prod(n_semiz);

Vtilde=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,joint(a1prime,a2prime) seperately
Policy=zeros(5,N_a,N_semiz,N_j,'gpuArray'); % 1=d2, 2=d3, 3=joint(a1prime,a2prime), 4=a1prime L2
Policy(5,:,:,:)=2; % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
Valt=zeros(N_a,N_semiz,N_j,'gpuArray');
Policyalt=zeros(5,N_a,N_semiz,N_j,'gpuArray'); % exponential discounter optimal choice
Policyalt(5,:,:,:)=2;

%%
if vfoptions.lowmemory==1
    special_n_semiz=ones(1,length(n_semiz));
end

% GI setup
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1
a2ind=gpuArray(0:N_a2-1)'; % column, for the experience-asset EV enumeration over a2prime
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1

% Preallocate (for the max over d3)
V_ford3_tilde=zeros(N_a,N_semiz,N_d3,'gpuArray');
Policy3_ford3_tilde=zeros(3,N_a,N_semiz,N_d3,'gpuArray'); % 1=d2, 2=joint(a1prime midpoint,a2prime), 3=a1prime L2
flag_ford3_tilde=2*ones(N_a,N_semiz,N_d3,'gpuArray');
V_ford3_alt=zeros(N_a,N_semiz,N_d3,'gpuArray');
Policy3_ford3_alt=zeros(3,N_a,N_semiz,N_d3,'gpuArray'); % 1=d2, 2=joint(a1prime midpoint,a2prime), 3=a1prime L2
flag_ford3_alt=2*ones(N_a,N_semiz,N_d3,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put u into third dimension
%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Period N_j could be done without looping over d3, but then it needs much more memory than the rest, and since looping for the other periods the runtime cost of looping here is negligible.
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            % [N_d2*n2long*N_a2, N_a, N_semiz]
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
            Policy3_ford3_tilde(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_tilde(2,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy3_ford3_tilde(3,:,:,d3_c)=maxindexL2a1; % a1prime L2
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d2)
            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_tilde(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);

                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_d3z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                % [N_d2*n2long*N_a2, N_a]
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3z,[],1);
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy3_ford3_tilde(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_tilde(2,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy3_ford3_tilde(3,:,z_c,d3_c)=maxindexL2a1; % a1prime L2
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d2)
                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_d3z(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3z(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_tilde(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1) -1);
    Policy(1,:,:,N_j)=reshape(Policy3_ford3_tilde(1+temp),[1,N_a,N_semiz]); % d2
    Policy(3,:,:,N_j)=reshape(Policy3_ford3_tilde(2+temp),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policy(4,:,:,N_j)=reshape(Policy3_ford3_tilde(3+temp),[1,N_a,N_semiz]); % a1prime L2
    Policy(5,:,:,N_j)=reshape(flag_ford3_tilde((1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    % Terminal period: no continuation, so the exponential and the QH-perceived problems coincide
    Valt(:,:,N_j)=Vtilde(:,:,N_j);
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j);
    Policyalt(5,:,:,N_j)=Policy(5,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_semiz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz], indexed by semizprime (d3-independent)
    EV_aprime(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV_aprime(aprimeProbs==1)=Vlower(aprimeProbs==1);
    EV_aprime=EV_aprime.*pi_u; EV_aprime(isnan(EV_aprime))=0; % a zero pi_u against an infinite node gives 0*(-Inf)=NaN, so zero the term BEFORE summing
    EV_aprime=squeeze(sum(EV_aprime,3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz_d3',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3)); % [N_d2*N_a1*N_a2,N_a3,N_semiz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d2,N_a1fine,N_a2,1,1,N_a3,N_semiz]

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            % alt pass (exponential discounter): F + beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta*entireEV;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
            Policy3_ford3_alt(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_alt(2,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy3_ford3_alt(3,:,:,d3_c)=maxindexL2a1; % a1prime L2
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d2)
            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
            isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

            % tilde pass (QH-perceived): F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta0beta*entireEV;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
            Policy3_ford3_tilde(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_tilde(2,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy3_ford3_tilde(3,:,:,d3_c)=maxindexL2a1; % a1prime L2
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d2)
            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
            isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_tilde(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz_d3',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                entireEV_z=entireEV(:,:,:,:,:,:,z_c);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);

                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                % alt pass (exponential discounter): F + beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta*entireEV_z;
                [~,maxindex]=max(entireRHS_d3z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp_z(aprime),[N_d2*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy3_ford3_alt(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_alt(2,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy3_ford3_alt(3,:,z_c,d3_c)=maxindexL2a1; % a1prime L2
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d2)
                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                % tilde pass (QH-perceived): F + beta0*beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta0beta*entireEV_z;
                [~,maxindex]=max(entireRHS_d3z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp_z(aprime),[N_d2*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy3_ford3_tilde(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_tilde(2,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy3_ford3_tilde(3,:,z_c,d3_c)=maxindexL2a1; % a1prime L2
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d2)
                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_tilde(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,N_j)=V_jjalt;
    Policyalt(2,:,:,N_j)=shiftdim(maxindexalt,-1); % d3 is just maxindexalt
    maxindexalt=reshape(maxindexalt,[N_a*N_semiz,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    tempalt=3*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt-1) -1);
    Policyalt(1,:,:,N_j)=reshape(Policy3_ford3_alt(1+tempalt),[1,N_a,N_semiz]); % d2
    Policyalt(3,:,:,N_j)=reshape(Policy3_ford3_alt(2+tempalt),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policyalt(4,:,:,N_j)=reshape(Policy3_ford3_alt(3+tempalt),[1,N_a,N_semiz]); % a1prime L2
    Policyalt(5,:,:,N_j)=reshape(flag_ford3_alt((1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt-1)),[1,N_a,N_semiz]);

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1) -1);
    Policy(1,:,:,N_j)=reshape(Policy3_ford3_tilde(1+temp),[1,N_a,N_semiz]); % d2
    Policy(3,:,:,N_j)=reshape(Policy3_ford3_tilde(2+temp),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policy(4,:,:,N_j)=reshape(Policy3_ford3_tilde(3+temp),[1,N_a,N_semiz]); % a1prime L2
    Policy(5,:,:,N_j)=reshape(flag_ford3_tilde((1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
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

    EVpre=Valt(:,:,jj+1); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_semiz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz], indexed by semizprime (d3-independent)
    EV_aprime(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV_aprime(aprimeProbs==1)=Vlower(aprimeProbs==1);
    EV_aprime=EV_aprime.*pi_u; EV_aprime(isnan(EV_aprime))=0; % a zero pi_u against an infinite node gives 0*(-Inf)=NaN, so zero the term BEFORE summing
    EV_aprime=squeeze(sum(EV_aprime,3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz_d3',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            % alt pass (exponential discounter): F + beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta*entireEV;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
            Policy3_ford3_alt(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_alt(2,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy3_ford3_alt(3,:,:,d3_c)=maxindexL2a1; % a1prime L2
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d2)
            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
            isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

            % tilde pass (QH-perceived): F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta0beta*entireEV;
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
            Policy3_ford3_tilde(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_tilde(2,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy3_ford3_tilde(3,:,:,d3_c)=maxindexL2a1; % a1prime L2
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d2)
            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
            isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_tilde(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz_d3',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                entireEV_z=entireEV(:,:,:,:,:,:,z_c);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);

                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                % alt pass (exponential discounter): F + beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta*entireEV_z;
                [~,maxindex]=max(entireRHS_d3z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp_z(aprime),[N_d2*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy3_ford3_alt(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_alt(2,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy3_ford3_alt(3,:,z_c,d3_c)=maxindexL2a1; % a1prime L2
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d2)
                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                % tilde pass (QH-perceived): F + beta0*beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta0beta*entireEV_z;
                [~,maxindex]=max(entireRHS_d3z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp_z(aprime),[N_d2*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;
                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy3_ford3_tilde(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_tilde(2,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy3_ford3_tilde(3,:,z_c,d3_c)=maxindexL2a1; % a1prime L2
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d2)
                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_tilde(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,jj)=V_jjalt;
    Policyalt(2,:,:,jj)=shiftdim(maxindexalt,-1); % d3 is just maxindexalt
    maxindexalt=reshape(maxindexalt,[N_a*N_semiz,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    tempalt=3*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt-1) -1);
    Policyalt(1,:,:,jj)=reshape(Policy3_ford3_alt(1+tempalt),[1,N_a,N_semiz]); % d2
    Policyalt(3,:,:,jj)=reshape(Policy3_ford3_alt(2+tempalt),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policyalt(4,:,:,jj)=reshape(Policy3_ford3_alt(3+tempalt),[1,N_a,N_semiz]); % a1prime L2
    Policyalt(5,:,:,jj)=reshape(flag_ford3_alt((1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindexalt-1)),[1,N_a,N_semiz]);

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,jj)=V_jj;
    Policy(2,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1) -1);
    Policy(1,:,:,jj)=reshape(Policy3_ford3_tilde(1+temp),[1,N_a,N_semiz]); % d2
    Policy(3,:,:,jj)=reshape(Policy3_ford3_tilde(2+temp),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policy(4,:,:,jj)=reshape(Policy3_ford3_tilde(3+temp),[1,N_a,N_semiz]); % a1prime L2
    Policy(5,:,:,jj)=reshape(flag_ford3_tilde((1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);

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
