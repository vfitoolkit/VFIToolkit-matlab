function [V,Policy]=ValueFnIter_FHorz_ExpAsset_GI_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
% n_a1prime=n_a1;
% a1prime_gridvals=a1_gridvals;
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e),'gpuArray');
end
if vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
else
    zind=shiftdim((0:1:N_z-1),-3); % already includes -1
end

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=0:1:N_a-1; % already includes -1
zindB=shiftdim((0:1:N_z-1),-1); % already includes -1
zeindB=zindB+N_z*shiftdim((0:1:N_e-1),-2); % already includes -1

a2ind=shiftdim((0:1:N_a2-1),-2); % already includes -1

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % Calc the max and it's index
        [~,maxindex]=max(ReturnMatrix,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,n_e, d_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2,N_e]
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policy3(1,:,:,:,N_j)=d_ind; % d2
        Policy3(2,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy3(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            % Calc the max and it's index
            [~,maxindex]=max(ReturnMatrix_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2]
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
            Policy3(1,:,:,e_c,N_j)=d_ind; % d2
            Policy3(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy3(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1);
                % Calc the max and it's index
                [~,maxindex]=max(ReturnMatrix_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2]
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
                Policy3(1,:,z_c,e_c,N_j)=d_ind; % d2
                Policy3(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
            end
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    Vnext=sum(shiftdim(pi_e_J(:,N_j),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3); % First, switch V_Jplus1 into Kron form

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z); % [N_d2*N_a1,N_a2,N_z]

    Vlower=reshape(Vnext(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(Vnext(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]);
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_z]
    DiscountedEV=repelem(DiscountedEV,N_d1,1);% [N_d1*N_d2,N_a1,1,N_a2,N_z]
    DiscountedEVinterp=repelem(DiscountedEVinterp,N_d1,1); % [N_d1*N_d2,N_a1prime,1,N_a2,N_z]

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1); % [N_d,N_a1prime,N_a1,N_a2,N_z,N_e]

        entireRHS=ReturnMatrix+DiscountedEV; % autofill 3rd dim to N_a1

        % Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2,N_z,N_e]
        da1primea2z=(1:1:N_d)'+N_d*(a1primeindexesfine-1)+N_d*N_a1prime*a2ind+N_d*N_a1prime*N_a2*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primea2z(:)),[N_d*n2long,N_a1*N_a2,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policy3(1,:,:,:,N_j)=d_ind; % d2
        Policy3(2,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy3(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1); % [N_d,N_a1prime,N_a1,N_a2,N_z]

            entireRHS_e=ReturnMatrix_e+DiscountedEV; % autofill 3rd dim to N_a1

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2,N_z]
            da1primea2z=(1:1:N_d)'+N_d*(a1primeindexesfine-1)+N_d*N_a1prime*a2ind+N_d*N_a1prime*N_a2*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primea2z(:)),[N_d*n2long,N_a1*N_a2,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
            Policy3(1,:,:,e_c,N_j)=d_ind; % d2
            Policy3(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy3(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1); % [N_d,N_a1prime,N_a1,N_a2]

                entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z; % autofill 3rd dim to N_a1

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2]
                da1primea2=(1:1:N_d)'+N_d*(a1primeindexesfine-1)+N_d*N_a1prime*a2ind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(da1primea2(:)),[N_d*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
                Policy3(1,:,z_c,e_c,N_j)=d_ind; % d2
                Policy3(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
            end
        end
    end
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z); % [N_d2*N_a1,N_a2,N_z]

    Vnext=sum(shiftdim(pi_e_J(:,N_j),-2).*V(:,:,:,jj+1),3); % First, switch V_Jplus1 into Kron form

    Vlower=reshape(Vnext(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(Vnext(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation
   
    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]);
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_z]
    DiscountedEV=repelem(DiscountedEV,N_d1,1);% [N_d1*N_d2,N_a1,1,N_a2,N_z]
    DiscountedEVinterp=repelem(DiscountedEVinterp,N_d1,1); % [N_d1*N_d2,N_a1prime,1,N_a2,N_z]

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1); % [N_d,N_a1prime,N_a1,N_a2,N_z,N_e]

        entireRHS=ReturnMatrix+DiscountedEV; % autofill 3rd dim to N_a1

        % Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2,N_z,N_e]
        da1primea2z=(1:1:N_d)'+N_d*(a1primeindexesfine-1)+N_d*N_a1prime*a2ind+N_d*N_a1prime*N_a2*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primea2z(:)),[N_d*n2long,N_a1*N_a2,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policy3(1,:,:,:,jj)=d_ind; % d2
        Policy3(2,:,:,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy3(3,:,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            ReturnMatrix_e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1); % [N_d,N_a1prime,N_a1,N_a2,N_z]

            entireRHS_e=ReturnMatrix_e+DiscountedEV; % autofill 3rd dim to N_a1

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2,N_z]
            da1primea2z=(1:1:N_d)'+N_d*(a1primeindexesfine-1)+N_d*N_a1prime*a2ind+N_d*N_a1prime*N_a2*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primea2z(:)),[N_d*n2long,N_a1*N_a2,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,:,e_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
            Policy3(1,:,:,e_c,jj)=d_ind; % d2
            Policy3(2,:,:,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy3(3,:,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
            DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1); % [N_d,N_a1prime,N_a1,N_a2]

                entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z; % autofill 3rd dim to N_a1

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2]
                da1primea2=(1:1:N_d)'+N_d*(a1primeindexesfine-1)+N_d*N_a1prime*a2ind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(da1primea2(:)),[N_d*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
                Policy3(1,:,z_c,e_c,jj)=d_ind; % d2
                Policy3(2,:,z_c,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy3(3,:,z_c,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
            end
        end
    end
end



%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy3(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy3(2,:,:,:,:)=Policy3(2,:,:,:,:)-adjust; % lower grid point
Policy3(3,:,:,:,:)=adjust.*Policy3(3,:,:,:,:)+(1-adjust).*(Policy3(3,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%% For experience asset, just output Policy as single index and then use Case2 to UnKron
Policy=shiftdim(Policy3(1,:,:,:,:)+N_d*(Policy3(2,:,:,:,:)-1)+N_d*N_a1*(Policy3(3,:,:,:,:)-1),1);


end
