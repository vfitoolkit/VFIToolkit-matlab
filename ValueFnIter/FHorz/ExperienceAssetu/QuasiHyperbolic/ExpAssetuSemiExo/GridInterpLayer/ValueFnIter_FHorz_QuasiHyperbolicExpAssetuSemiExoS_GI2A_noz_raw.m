function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_GI2A_noz_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_semiz, n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J, u_gridvals, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAssetuSemiExo_GI2A_noz_raw.
% ExperienceAsset + semi-exogenous shocks, with the grid interpolation layer on a1. GPU only.
%
% Sophisticated: Vhat_j      = max_{d,a1'} u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EVfine_at_optimal_choice
% EVfine is the (undiscounted) interpolated continuation actually added to the layer-2 RHS, so
% the a3 lottery is already baked in and the gather needs no lottery handling. The gather is
% taken at the FINAL layer-2 argmax (maxindexL2), and then at the chosen d3.
% beta0 is received as a trailing input.
% SemiExo _noz graft of ValueFnIter_FHorz_ExpAsset_GI2A_noz_raw, mirroring the semiz mechanics of ValueFnIter_FHorz_ExpAssetSemiExo_GI1_noz_raw.
% d1 is any other decision, d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is the grid-interpolated standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% NO z, NO e: the only shock is semiz (so bothz=semiz throughout).
% Policy stores (d1, d2, d3, joint(a1prime,a2prime), a1primeL2ind); the 4th row is a1prime+N_a1*(a2prime-1), a1prime being the lower grid point. The L2 flag is the final row.
% lowmemory: 1 shock {semiz} => levels {0,1}.
%   =0 vectorise semiz; =1 loop semiz.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_u=prod(n_u);
N_semiz=prod(n_semiz);

Vhat=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,d3,joint(a1prime,a2prime),a1primeL2ind seperately
Policy=zeros(6,N_a,N_semiz,N_j,'gpuArray');
Policy(6,:,:,:)=2; % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
Vunderbar=zeros(N_a,N_semiz,N_j,'gpuArray');

%%
d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]; maps d12-index to d2-component (used inside the d3 loop where d=d12)

if vfoptions.lowmemory==0
    semizindB=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1
elseif vfoptions.lowmemory==1
    special_n_semiz=ones(1,length(n_semiz));
end

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1

% Preallocate (for the max over d3)
V_ford3_hat=zeros(N_a,N_semiz,N_d3,'gpuArray');
Policy4_ford3_hat=zeros(4,N_a,N_semiz,N_d3,'gpuArray'); % d1,d2,joint(a1prime,a2prime),a1primeL2ind
flag_ford3_hat=2*ones(N_a,N_semiz,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_semiz,N_d3,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put u into third dimension
%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Period N_j could be done without looping over d3, but then it needs much more memory than the rest, and since looping for the other periods the runtime cost of looping here is negligible.
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizindB;
            Policy4_ford3_hat(1,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_hat(2,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_hat(3,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_hat(4,:,:,d3_c)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                  + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);

                ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                Policy4_ford3_hat(1,:,z_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_hat(2,:,z_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_hat(3,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_hat(4,:,z_c,d3_c)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                  + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_hat,[],3); % max over d3
    Vhat(:,:,N_j)=V_jj;
    Policy(3,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=4*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1) -1);
    Policy(1,:,:,N_j)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_semiz]); % d1
    Policy(2,:,:,N_j)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_semiz]); % d2
    Policy(4,:,:,N_j)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_semiz]); % joint(a1prime,a2prime)
    Policy(5,:,:,N_j)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_semiz]); % a1primeL2ind
    Policy(6,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col     =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col     =repelem((0:N_a2-1)',N_d2*N_a1,1);
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
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz_d3',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3)); % [N_d2*N_a1*N_a2,N_a3,N_semiz], now indexed by current semiz
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS=ReturnMatrix+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            EVfine=reshape(entireEVinterp(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
            entireRHS_ii=reshape(ReturnMatrix_ii,[N_d12*n2long*N_a2,N_a,N_semiz])+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizindB;
            Policy4_ford3_hat(1,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_hat(2,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_hat(3,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_hat(4,:,:,d3_c)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                  + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            % Vunderbar: the beta-discounted RHS gathered at the FINAL (layer-2) Vhat argmax -- not a second max
            linidx=reshape(maxindexL2,[1,N_a*N_semiz])+size(EVfine,1)*(0:N_a*N_semiz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
            V_ford3_under(:,:,d3_c)=V_ford3_hat(:,:,d3_c)+(beta-beta0beta)*EV_at_policy;
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
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

                ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_z=ReturnMatrix_z+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS_z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                EVfine=reshape(entireEVinterp_z(aprime),[N_d12*n2long*N_a2,N_a]);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z,[N_d12*n2long*N_a2,N_a])+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                Policy4_ford3_hat(1,:,z_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_hat(2,:,z_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_hat(3,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_hat(4,:,z_c,d3_c)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                  + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                % Vunderbar: the beta-discounted RHS gathered at the FINAL (layer-2) Vhat argmax -- not a second max
                linidx=reshape(maxindexL2,[1,N_a])+size(EVfine,1)*(0:N_a-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
                V_ford3_under(:,z_c,d3_c)=V_ford3_hat(:,z_c,d3_c)+(beta-beta0beta)*EV_at_policy;
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_hat,[],3); % max over d3
    Vhat(:,:,N_j)=V_jj;
    Policy(3,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=4*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1) -1);
    Policy(1,:,:,N_j)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_semiz]); % d1
    Policy(2,:,:,N_j)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_semiz]); % d2
    Policy(4,:,:,N_j)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_semiz]); % joint(a1prime,a2prime)
    Policy(5,:,:,N_j)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_semiz]); % a1primeL2ind
    Policy(6,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    % Vunderbar at the d3 that Vhat chose
    Vunderbar(:,:,N_j)=reshape(V_ford3_under((1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[N_a,N_semiz]);
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

    EVpre=Vunderbar(:,:,jj+1); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col     =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col     =repelem((0:N_a2-1)',N_d2*N_a1,1);
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
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz_d3',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS=ReturnMatrix+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            EVfine=reshape(entireEVinterp(aprime),[N_d12*n2long*N_a2,N_a,N_semiz]);
            entireRHS_ii=reshape(ReturnMatrix_ii,[N_d12*n2long*N_a2,N_a,N_semiz])+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizindB;
            Policy4_ford3_hat(1,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_hat(2,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_hat(3,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_hat(4,:,:,d3_c)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                  + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizindB;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            % Vunderbar: the beta-discounted RHS gathered at the FINAL (layer-2) Vhat argmax -- not a second max
            linidx=reshape(maxindexL2,[1,N_a*N_semiz])+size(EVfine,1)*(0:N_a*N_semiz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
            V_ford3_under(:,:,d3_c)=V_ford3_hat(:,:,d3_c)+(beta-beta0beta)*EV_at_policy;
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
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

                ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_z=ReturnMatrix_z+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS_z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                EVfine=reshape(entireEVinterp_z(aprime),[N_d12*n2long*N_a2,N_a]);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z,[N_d12*n2long*N_a2,N_a])+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                Policy4_ford3_hat(1,:,z_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_hat(2,:,z_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_hat(3,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_hat(4,:,z_c,d3_c)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                  + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                % Vunderbar: the beta-discounted RHS gathered at the FINAL (layer-2) Vhat argmax -- not a second max
                linidx=reshape(maxindexL2,[1,N_a])+size(EVfine,1)*(0:N_a-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
                V_ford3_under(:,z_c,d3_c)=V_ford3_hat(:,z_c,d3_c)+(beta-beta0beta)*EV_at_policy;
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_hat,[],3); % max over d3
    Vhat(:,:,jj)=V_jj;
    Policy(3,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=4*( (1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1) -1);
    Policy(1,:,:,jj)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_semiz]); % d1
    Policy(2,:,:,jj)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_semiz]); % d2
    Policy(4,:,:,jj)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_semiz]); % joint(a1prime,a2prime)
    Policy(5,:,:,jj)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_semiz]); % a1primeL2ind
    Policy(6,:,:,jj)=reshape(flag_ford3_hat((1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    % Vunderbar at the d3 that Vhat chose
    Vunderbar(:,:,jj)=reshape(V_ford3_under((1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[N_a,N_semiz]);

end


%% With grid interpolation, switch from midpoint to lower grid index
% Currently Policy(4,:) holds joint(a1prime midpoint,a2prime) and Policy(5,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch the a1prime part of the joint to 'lower grid point' and then have Policy(5,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(5,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(4,:,:,:)=Policy(4,:,:,:)-adjust; % a1prime part of joint -> lower grid point
Policy(5,:,:,:)=Policy(5,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)


end
