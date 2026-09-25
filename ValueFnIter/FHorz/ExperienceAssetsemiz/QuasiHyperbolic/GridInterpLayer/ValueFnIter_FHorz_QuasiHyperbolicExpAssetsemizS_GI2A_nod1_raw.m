function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetsemizS_GI2A_nod1_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAssetsemiz_GI2A_nod1_raw.
% ExperienceAssetsemiz (a3prime=aprimeFn(d2,a3,semiz)) + semi-exogenous shocks, with the grid interpolation layer on a1. GPU only.
%
% Sophisticated: Vhat_j      = max_{d,a1'} u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EVfine_at_optimal_choice
% EVfine is the (undiscounted) interpolated continuation actually added to the layer-2 RHS, so
% the a3 lottery is already baked in and the gather needs no lottery handling. The gather is
% taken at the FINAL layer-2 argmax (maxindexL2), and then at the chosen d3.
% beta0 is received as a trailing input.
% experienceassetsemiz GI2A: a3prime=aprimeFn(d2,a3,semiz), so aprime depends on
% the current semi-exogenous state. Structure follows
% ValueFnIter_FHorz_ExpAssetSemiExo_GI2A_nod1_raw; the EV pipeline differs
% (see the comment at the aprime setup).
% _nod1: only d2 (which determines experience asset a3) and d3 (which determines semi-exog state semiz).
% a1 is the grid-interpolated first standard asset (scalar); a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% z is exogenous Markov, semiz is semi-exogenous; bothz=(semiz,z) with semiz varying fastest.
% Policy stores (d2, d3, joint(a1prime,a2prime)); the 3rd row is a1prime_lower+N_a1*(a2prime-1).
% Two extra rows are appended for the grid-interp layer: a1prime L2 index and L2flag.
% lowmemory: 2 shocks {z,semiz} => levels {0,1,2}.
%   =0 vectorise bothz; =1 outer-loop z / inner semiz parallel; =2 joint bothz outer loop.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

Vhat=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,joint(a1prime,a2prime) seperately
Policy=zeros(5,N_a,N_semiz*N_z,N_j,'gpuArray'); % 1=d2, 2=d3, 3=joint(a1prime,a2prime), 4=a1prime L2
Policy(5,:,:,:)=2; % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
Vunderbar=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory==1
    special_n_semiz=[n_semiz,ones(1,length(n_z))];
elseif vfoptions.lowmemory==2
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

%% GI setup
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);                   % flat a-axis
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % at dim 3 of [1,N_a,N_bothz]
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % at dim 3 of [1,N_a,N_semiz]

% Preallocate (for the d3-loop sections, which loop over d3)
V_ford3_hat=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');
Policy3_ford3_hat=zeros(3,N_a,N_semiz*N_z,N_d3,'gpuArray'); % 1=d2, 2=joint(a1prime midpoint,a2prime), 3=a1prime L2
flag_ford3_hat=2*ones(N_a,N_semiz*N_z,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            % --- Coarse pass ---
            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            % --- Fine pass ---
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
            Policy3_ford3_hat(1,:,:,d3_c)=d_ind;
            Policy3_ford3_hat(2,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy3_ford3_hat(3,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);

                ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
                V_ford3_hat(:,zind,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                Policy3_ford3_hat(1,:,zind,d3_c)=d_ind;
                Policy3_ford3_hat(2,:,zind,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy3_ford3_hat(3,:,zind,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,zind,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy3_ford3_hat(1,:,z_c,d3_c)=d_ind;
                Policy3_ford3_hat(2,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy3_ford3_hat(3,:,z_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
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
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policy(1,:,:,N_j)=reshape(Policy3_ford3_hat(1+temp),[1,N_a,N_bothz]); % d2
    Policy(3,:,:,N_j)=reshape(Policy3_ford3_hat(2+temp),[1,N_a,N_bothz]); % joint(a1prime midpoint,a2prime)
    Policy(4,:,:,N_j)=reshape(Policy3_ford3_hat(3+temp),[1,N_a,N_bothz]); % a1prime L2
    Policy(5,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);

else
    % vfoptions.V_Jplus1 provided
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a3, n_semiz, d2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex, a3primeProbs are [N_d2,N_a3,N_semiz], indexed by the CURRENT semiz
    % aprime depends only on semiz (the FAST index of bothz), so tile over N_z.

    a1_col     =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col     =repelem((0:N_a2-1)',N_d2*N_a1,1);
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
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            % --- Coarse pass ---
            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS=ReturnMatrix+beta0beta*entireEV;
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            % --- Fine pass ---
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            EVfine=reshape(entireEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz]);
            entireRHS_ii=reshape(ReturnMatrix_ii,[N_d2*n2long*N_a2,N_a,N_bothz])+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
            Policy3_ford3_hat(1,:,:,d3_c)=d_ind;
            Policy3_ford3_hat(2,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
            Policy3_ford3_hat(3,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            % Vunderbar: the beta-discounted RHS gathered at the FINAL (layer-2) Vhat argmax -- not a second max
            linidx=reshape(maxindexL2,[1,N_a*N_bothz])+size(EVfine,1)*(0:N_a*N_bothz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_bothz]);
            V_ford3_under(:,:,d3_c)=V_ford3_hat(:,:,d3_c)+(beta-beta0beta)*EV_at_policy;
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                entireEV_z=entireEV(:,:,:,:,:,:,zind);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,zind);

                ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z;
                [~,maxindex]=max(entireRHS_z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime_z=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                EVfine=reshape(entireEVinterp_z(aprime_z),[N_d2*n2long*N_a2,N_a,N_semiz]);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z,[N_d2*n2long*N_a2,N_a,N_semiz])+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
                V_ford3_hat(:,zind,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                Policy3_ford3_hat(1,:,zind,d3_c)=d_ind;
                Policy3_ford3_hat(2,:,zind,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
                Policy3_ford3_hat(3,:,zind,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,zind,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                % Vunderbar: the beta-discounted RHS gathered at the FINAL (layer-2) Vhat argmax -- not a second max
                linidx=reshape(maxindexL2,[1,N_a*N_semiz])+size(EVfine,1)*(0:N_a*N_semiz-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
                V_ford3_under(:,zind,d3_c)=V_ford3_hat(:,zind,d3_c)+(beta-beta0beta)*EV_at_policy;
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                entireEV_z=entireEV(:,:,:,:,:,:,z_c);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);

                ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z;
                [~,maxindex]=max(entireRHS_z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime_z=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                EVfine=reshape(entireEVinterp_z(aprime_z),[N_d2*n2long*N_a2,N_a]);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z,[N_d2*n2long*N_a2,N_a])+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy3_ford3_hat(1,:,z_c,d3_c)=d_ind;
                Policy3_ford3_hat(2,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
                Policy3_ford3_hat(3,:,z_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
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
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=3*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policy(1,:,:,N_j)=reshape(Policy3_ford3_hat(1+temp),[1,N_a,N_bothz]);
    Policy(3,:,:,N_j)=reshape(Policy3_ford3_hat(2+temp),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(Policy3_ford3_hat(3+temp),[1,N_a,N_bothz]);
    Policy(5,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    % Vunderbar at the d3 that Vhat chose
    Vunderbar(:,:,N_j)=reshape(V_ford3_under((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[N_a,N_bothz]);
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

    EVpre=Vunderbar(:,:,jj+1); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a3, n_semiz, d2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex, a3primeProbs are [N_d2,N_a3,N_semiz], indexed by the CURRENT semiz
    % aprime depends only on semiz (the FAST index of bothz), so tile over N_z.

    a1_col     =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col     =repelem((0:N_a2-1)',N_d2*N_a1,1);
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
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            % --- Coarse pass ---
            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS=ReturnMatrix+beta0beta*entireEV;
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            % --- Fine pass ---
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            EVfine=reshape(entireEVinterp(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz]);
            entireRHS_ii=reshape(ReturnMatrix_ii,[N_d2*n2long*N_a2,N_a,N_bothz])+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
            Policy3_ford3_hat(1,:,:,d3_c)=d_ind;
            Policy3_ford3_hat(2,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
            Policy3_ford3_hat(3,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            % Vunderbar: the beta-discounted RHS gathered at the FINAL (layer-2) Vhat argmax -- not a second max
            linidx=reshape(maxindexL2,[1,N_a*N_bothz])+size(EVfine,1)*(0:N_a*N_bothz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_bothz]);
            V_ford3_under(:,:,d3_c)=V_ford3_hat(:,:,d3_c)+(beta-beta0beta)*EV_at_policy;
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                entireEV_z=entireEV(:,:,:,:,:,:,zind);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,zind);

                ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z;
                [~,maxindex]=max(entireRHS_z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime_z=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                EVfine=reshape(entireEVinterp_z(aprime_z),[N_d2*n2long*N_a2,N_a,N_semiz]);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z,[N_d2*n2long*N_a2,N_a,N_semiz])+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
                V_ford3_hat(:,zind,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                Policy3_ford3_hat(1,:,zind,d3_c)=d_ind;
                Policy3_ford3_hat(2,:,zind,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
                Policy3_ford3_hat(3,:,zind,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                isInfLower   =(ReturnMatrix_ii_z(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_z(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,zind,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                % Vunderbar: the beta-discounted RHS gathered at the FINAL (layer-2) Vhat argmax -- not a second max
                linidx=reshape(maxindexL2,[1,N_a*N_semiz])+size(EVfine,1)*(0:N_a*N_semiz-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
                V_ford3_under(:,zind,d3_c)=V_ford3_hat(:,zind,d3_c)+(beta-beta0beta)*EV_at_policy;
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EVc=EVpre.*shiftdim(pi_bothz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_bothz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                entireEV_z=entireEV(:,:,:,:,:,:,z_c);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);

                ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z;
                [~,maxindex]=max(entireRHS_z,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime_z=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                EVfine=reshape(entireEVinterp_z(aprime_z),[N_d2*n2long*N_a2,N_a]);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z,[N_d2*n2long*N_a2,N_a])+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                Policy3_ford3_hat(1,:,z_c,d3_c)=d_ind;
                Policy3_ford3_hat(2,:,z_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1);
                Policy3_ford3_hat(3,:,z_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                  + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
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
    Policy(2,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=3*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policy(1,:,:,jj)=reshape(Policy3_ford3_hat(1+temp),[1,N_a,N_bothz]);
    Policy(3,:,:,jj)=reshape(Policy3_ford3_hat(2+temp),[1,N_a,N_bothz]);
    Policy(4,:,:,jj)=reshape(Policy3_ford3_hat(3+temp),[1,N_a,N_bothz]);
    Policy(5,:,:,jj)=reshape(flag_ford3_hat((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    % Vunderbar at the d3 that Vhat chose
    Vunderbar(:,:,jj)=reshape(V_ford3_under((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[N_a,N_bothz]);

end


%% Post-process: convert "midpoint + L2 offset" into "lower coarse point + L2 ratio"
% Currently Policy(3,:) is joint(a1prime midpoint,a2prime), Policy(4,:) is the L2 index (ranges -n2short-1:1:1+n2short).
% Switch Policy(3,:) to joint(lower a1prime grid point,a2prime), and Policy(4,:) to a 1..(n2short+2) offset.
adjust=(Policy(4,:,:,:)<1+n2short+1); % is the L2 index below midpoint?
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust; % decrement a1prime component of the joint (midpoint>=2 keeps it within the same a2prime block)
Policy(4,:,:,:)=Policy(4,:,:,:)-(n2short+1)*(~adjust);

end
