function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_DC1_GI_nod1_noz_e_raw(V,n_d2,n_a1,n_a2,n_e,N_j, d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J,pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,e), rather than standard (a,e,j)
% V is (a,j)-by-e
% pi_e_J is (j,e') for fastOLG
% e_gridvals_J is (j,N_e,l_e) for fastOLG

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_e=prod(n_e);

e_gridvals_J=shiftdim(e_gridvals_J,-4); % [1,1,1,1,N_j,N_e,l_e]

Policy3=zeros(3,N_a,N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%

% Preallocate
if vfoptions.lowmemory==0
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_j,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_j,'gpuArray');
end

% n-Monotonicity
% vfoptions.level1n=21;
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

d2ind=gpuArray(1:1:N_d2)';
aind=shiftdim(gpuArray(0:1:N_a-1),-2);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
jind=shiftdim(gpuArray(0:1:N_j-1),-3);
eind=shiftdim(gpuArray(0:1:N_e-1),-4);


%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-4);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_J(aprimeFn, n_d2, n_a2, N_j, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_j], whereas aprimeProbs is [N_d2,N_a2,N_j]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat((a2primeIndex-1),N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_j]

    EVpre=[sum(V(N_a+1:end,:).*replem(pi_e_J,N_a-1,1),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations

    % Need to add the indexes for j to the aprimeIndex, remember fastOLG so V is (a,j)-by-e
    Vlower=reshape(EVpre(aprimeIndex+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    Vupper=reshape(EVpre(aprimeplus1Index+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,N_j,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=reshape(EV,[N_d2*N_a1,N_a2,N_j]); % (aprime,1,j), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_J(aprimeFn, n_d2, n_a2, N_j, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_j], whereas aprimeProbs is [N_d2,N_a2,N_j]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat((a2primeIndex-1),N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_j]

    EVpre=sum(V.*replem(pi_e_J,N_a,1),2);

    % Need to add the indexes for j to the aprimeIndex, remember fastOLG so V is (a,j)-by-e
    Vlower=reshape(EVpre(aprimeIndex+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    Vupper=reshape(EVpre(aprimeplus1Index+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,N_j,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=reshape(EV,[N_d2*N_a1,N_a2,N_j]); % (aprime,1,j), 2nd dim will be autofilled with a
end

DiscountedEV=DiscountFactorParamsVec.*reshape(EV,[N_d2,N_a1,1,N_a2,N_j]);
% Interpolate EV over aprime_grid
DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);   % [N_d2,N_a1prime,1,N_a2,N_j]

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_fastOLG_ExpAsset_Disc_Par2(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n,n_a2, n_e,N_j, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,1,0);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want a1prime conditional on (d,1,a)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoint(:,1,level1ii,:,:,:)=maxindex1;

    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:,:),n_a1-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are N_d2-by-maxgap(ii)+1-by-1-by-N_j-by-N_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_fastOLG_ExpAsset_Disc_Par2(ReturnFn, 0, n_d2, maxgap(ii)+1, level1iidiff(ii),n_a2, n_e,N_j, d2_gridvals, a1_gridvals(aprimeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,3,0);
            d2aprimej=d2ind+N_d2*(aprimeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*jind; % with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(d2aprimej),[N_d2,(maxgap(ii)+1),1,N_a2,N_j,N_e]);
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:,:,:);
            midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d2-1-by-n_a1-by-n_a2-by-N_j-by-n_e
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
    % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-N_j-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_fastOLG_ExpAsset_Disc_Par2(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, n_e,N_j, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_grid, e_gridvals_J, ReturnFnParamsAgeMatrix,2,0);
    d2aprimej=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*jind; % the current aprimeii(ii):aprimeii(ii+1)
    entireRHS_ii=ReturnMatrix_ii+DiscountedEVinterp(reshape(d2aprimej,[N_d2*n2long,N_a1*N_a2,N_j,N_e]));
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V=shiftdim(Vtempii,1);
    dind=rem(maxindexL2-1,N_d2)+1;
    allind=reshape(dind,[1,1,1,N_a1*N_a2,N_j,N_e])+N_d2*aind+N_d2*N_a*jind+N_d2*N_a*N_j*eind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-N_j-by-n_e
    allind=reshape(allind,[1,N_a1*N_a2,N_j,N_e]);
    Policy3(1,:,:,:)=dind; % d2
    Policy3(2,:,:,:)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
    Policy3(3,:,:,:)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

elseif vfoptions.lowmemory==1
    V=zeros(N_a,N_j,N_e,'gpuArray');
    special_n_e=ones(1,length(n_e),'gpuArray');

    for e_c=1:N_e
        e_val=e_gridvals_J(1,1,1,1,:,e_c,:);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_fastOLG_ExpAsset_Disc_Par2(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n,n_a2, special_n_e,N_j, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_grid, e_val, ReturnFnParamsAgeMatrix,1,0);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoint(:,1,level1ii,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a1-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are N_d2-by-maxgap(ii)+1-by-1-by-N_j
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_fastOLG_ExpAsset_Disc_Par2(ReturnFn, 0, n_d2, maxgap(ii)+1, level1iidiff(ii),n_a2, special_n_e,N_j, d2_gridvals, a1_gridvals(aprimeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_val, ReturnFnParamsAgeMatrix,3,0);
                d2aprimej=d2ind+N_d2*(aprimeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*jind; % with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(d2aprimej),[N_d2,(maxgap(ii)+1),1,N_a2,N_j]);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d2-1-by-n_a1-by-n_a2-by-N_j
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
        % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-N_j
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_fastOLG_ExpAsset_Disc_Par2(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, special_n_e,N_j, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_grid, e_val, ReturnFnParamsAgeMatrix,2,0);
        d2aprimej=d2ind+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*jind; % the current aprimeii(ii):aprimeii(ii+1)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEVinterp(reshape(d2aprimej,[N_d2*n2long,N_a1*N_a2,N_j]));
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,e_c)=shiftdim(Vtempii,1);
        dind=rem(maxindexL2-1,N_d2)+1;
        allind=reshape(dind,[1,1,1,N_a1*N_a2,N_j])+N_d2*aind+N_d2*N_a*jind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-N_j
        allind=reshape(allind,[1,N_a1*N_a2,N_j]);
        Policy3(1,:,:,e_c)=dind; % d2
        Policy3(2,:,:,e_c)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy3(3,:,:,e_c)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

    end
end


%% fastOLG with z, so need to output to take certain shapes
V=reshape(V,[N_a*N_j,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_e]);


%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy3(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy3(2,:,:,:)=Policy3(2,:,:,:)-adjust; % lower grid point
Policy3(3,:,:,:)=adjust.*Policy3(3,:,:,:)+(1-adjust).*(Policy3(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%% For experience asset, just output Policy as single index and then use Case2 to UnKron
Policy=Policy3(1,:,:,:)+N_d2*(Policy3(2,:,:,:)-1)+N_d2*N_a1*(Policy3(3,:,:,:)-1);
% Output shape for policy, first dim is just one point


end
