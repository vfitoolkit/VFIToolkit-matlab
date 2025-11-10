function [V,Policy]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u,N_j, d2_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, u_grid, pi_z_J, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% z is exogenous state, semiz is semi-exog state

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_u=prod(n_u);

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,a1prime seperately
Policy4=zeros(4,N_a,N_semiz*N_z,N_j,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put it into third dimension

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory==0
    % precompute
    bothzind=shiftdim((0:1:N_bothz-1),-3); % already includes -1
    bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
elseif vfoptions.lowmemory==1 % loop over z
    special_n_semiz=[n_semiz,ones(1,length(n_z))];
    % precompute
    semizind=shiftdim((0:1:N_semiz-1),-3); % already includes -1
    semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1
elseif vfoptions.lowmemory==2 % loop over semiz & z
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end


% Preallocate
V_ford3_jj=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');
Policy3_ford3_jj=zeros(3,N_a,N_semiz*N_z,N_d3,'gpuArray');

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=0:1:N_a-1; % already includes -1

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        % Period N_j could be done without looping over d3, but then it needs much more memory than the rest, and since looping for the other periods the runtime cost of looping here is negligible.
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1); % [N_d2,N_d3,N_a1,N_a1,N_a2,N_bothz]

            % Calc the max and it's index
            [~,maxindex]=max(ReturnMatrix,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_bothz
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_bothz
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d23_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2); % [N_d2,N_d3,N_a1prime,N_a1,N_a2,N_bothz]
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_bothz
            Policy3_ford3_jj(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_jj(2,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy3_ford3_jj(3,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

        end
    elseif vfoptions.lowmemory==1

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);

                ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1); % [N_d2,N_a1,N_a1,N_a2,N_semiz]

                % Calc the max and it's index
                [~,maxindex]=max(ReturnMatrix_z,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_semiz
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_semiz
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2); % [N_d12,N_a1prime,N_a1,N_a2,N_semiz]
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford3_jj(:,zind,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*semizBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_semiz
                Policy3_ford3_jj(1,:,zind,d3_c)=d_ind; % d2
                Policy3_ford3_jj(2,:,zind,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3_ford3_jj(3,:,zind,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            end
        end
    elseif vfoptions.lowmemory==2

        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1); % [N_d2,N_a1,N_a1,N_a2]

                % Calc the max and it's index
                [~,maxindex]=max(ReturnMatrix_z,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2); % [N_d12,N_a1prime,N_a1,N_a2]
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford3_jj(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_semiz
                Policy3_ford3_jj(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_jj(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3_ford3_jj(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d3
    V(:,:,N_j)=V_jj;
    Policy4(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policy4(1,:,:,N_j)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_bothz]);
    Policy4(3,:,:,N_j)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_bothz]);
    Policy4(4,:,:,N_j)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_bothz]);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_bothz);  % [N_d2*N_a1,N_a2,N_u,N_bothz]
    
    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,both)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_bothz]);
            % Interpolate EV over aprime_grid
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]

            ReturnMatrix_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            % (d,aprime,a,z)

            entireRHS_d3=ReturnMatrix_d3+DiscountedEV;

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_d3,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_bothz
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_bothz
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2); % [N_d2,N_a1prime,N_a1,N_a2,N_bothz]
            d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_bothz
            Policy3_ford3_jj(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_jj(2,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy3_ford3_jj(3,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

        end
        
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,both)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_bothz]);
            % Interpolate EV over aprime_grid
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,:,:,zind);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,zind);

                ReturnMatrix_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1);

                entireRHS_d3z=ReturnMatrix_d3z+DiscountedEV_z;

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS_d3z,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_semiz
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_semiz
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]
                d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_z(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3z,[],1);
                V_ford3_jj(:,zind,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*semizBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_semiz
                Policy3_ford3_jj(1,:,zind,d3_c)=d_ind; % d2
                Policy3_ford3_jj(2,:,zind,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3_ford3_jj(3,:,zind,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,both)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_bothz]);
            % Interpolate EV over aprime_grid
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]
            
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);

                ReturnMatrix_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1);

                entireRHS_d3z=ReturnMatrix_d3z+DiscountedEV_z;

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS_d3z,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]
                d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_z(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
                Policy3_ford3_jj(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_jj(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3_ford3_jj(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d3
    V(:,:,N_j)=V_jj;
    Policy4(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policy4(1,:,:,N_j)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_bothz]);
    Policy4(3,:,:,N_j)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_bothz]);
    Policy4(4,:,:,N_j)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_bothz]);
    
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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_bothz);  % [N_d2*N_a1,N_a2,N_u,N_bothz]
    
    EVpre=V(:,:,jj+1);


    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,both)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_bothz]);
            % Interpolate EV over aprime_grid
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]

            ReturnMatrix_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
            % (d,aprime,a,z)

            entireRHS_d3=ReturnMatrix_d3+DiscountedEV;

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_d3,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_bothz
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_bothz
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2); % [N_d2,N_a1prime,N_a1,N_a2,N_bothz]
            d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_bothz
            Policy3_ford3_jj(1,:,:,d3_c)=d_ind; % d2
            Policy3_ford3_jj(2,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy3_ford3_jj(3,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

        end
        
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,both)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_bothz]);
            % Interpolate EV over aprime_grid
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,:,:,zind);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,zind);

                ReturnMatrix_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1);

                entireRHS_d3z=ReturnMatrix_d3z+DiscountedEV_z;

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS_d3z,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_semiz
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_semiz
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_semiz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]
                d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_z(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3z,[],1);
                V_ford3_jj(:,zind,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*semizBind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_semiz
                Policy3_ford3_jj(1,:,zind,d3_c)=d_ind; % d2
                Policy3_ford3_jj(2,:,zind,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3_ford3_jj(3,:,zind,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_bothz]); % (d2,a1prime,a2,u,bothz), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,both)

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_bothz]);
            % Interpolate EV over aprime_grid
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]
            
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);

                ReturnMatrix_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1);

                entireRHS_d3z=ReturnMatrix_d3z+DiscountedEV_z;

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS_d3z,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2); % [N_d2,N_a1prime,N_a1,N_a2,N_semiz]
                d2a1primea2semiz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_z(d2a1primea2semiz(:)),[N_d2*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
                Policy3_ford3_jj(1,:,z_c,d3_c)=d_ind; % d2
                Policy3_ford3_jj(2,:,z_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3_ford3_jj(3,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d3
    V(:,:,jj)=V_jj;
    Policy4(2,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    temp=3*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policy4(1,:,:,jj)=reshape(Policy3_ford3_jj(1+temp),[1,N_a,N_bothz]);
    Policy4(3,:,:,jj)=reshape(Policy3_ford3_jj(2+temp),[1,N_a,N_bothz]);
    Policy4(4,:,:,jj)=reshape(Policy3_ford3_jj(3+temp),[1,N_a,N_bothz]);
    

end



%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy4(4,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy4(3,:,:,:)=Policy4(3,:,:,:)-adjust; % lower grid point
Policy4(4,:,:,:)=adjust.*Policy4(4,:,:,:)+(1-adjust).*(Policy4(4,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%% For experience asset, just output Policy as single index and then use Case2 to UnKron
Policy=shiftdim(Policy4(1,:,:,:)+N_d2*(Policy4(2,:,:,:)-1)+N_d2*N_d3*(Policy4(3,:,:,:)-1)+N_d2*N_d3*N_a1*(Policy4(4,:,:,:)-1),1);


end
