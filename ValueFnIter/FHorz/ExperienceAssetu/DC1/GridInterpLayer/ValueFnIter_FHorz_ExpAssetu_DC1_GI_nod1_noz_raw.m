function [V,Policy]=ValueFnIter_FHorz_ExpAssetu_DC1_GI_nod1_noz_raw(n_d2,n_a1,n_a2,n_u,N_j, d2_gridvals, d2_grid, a1_gridvals, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_u=prod(n_u);

V=zeros(N_a,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

pi_u=shiftdim(pi_u,-2); % put it into third dimension

% Preallocate
midpoint=zeros(N_d2,1,N_a1,N_a2,'gpuArray');

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

aind=0:1:N_a-1; % already includes -1
a2ind=shiftdim((0:1:N_a2-1),-2); % already includes -1

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n, n_a2, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, ReturnFnParamsVec,1);

    % First, we want a1prime conditional on (d,1,a)
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

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
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, 0,n_d2, maxgap(ii)+1, level1iidiff(ii), n_a2, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, ReturnFnParamsVec,3); % Level 3 as DC1+GI
            [~,maxindex]=max(ReturnMatrix_ii,[],2);
            midpoint(:,1,curraindex,:)=maxindex+N_d2*(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:);
            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d2-1-by-n_a1-by-n_a2
    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
    % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(a1primeindexes), a1_gridvals, a2_gridvals, ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2]
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d2)+1;
    allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
    Policy3(1,:,N_j)=d_ind; % d2
    Policy3(2,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
    Policy3(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_u]

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);

    Vlower=reshape(EVpre(aprimeIndex(:)),[N_d2*N_a1,N_a2,N_u]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1,N_a2,N_u]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u)
    % Already applied the probabilities from interpolating onto grid
    EV=sum((EV.*pi_u),3); % (d2,a1prime,a2)
    
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2]);
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]);   % [N_d2,N_a1prime,1,N_a2] 

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n, n_a2, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want a1prime conditional on (d,1,a)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Just keep the 'midpoint' vesion of maxindex1 [as GI]
    midpoint(:,1,level1ii,:)=maxindex1;

    % Attempt for improved version
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-1-by-n_a2
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, 0, n_d2, maxgap(ii)+1, level1iidiff(ii), n_a2, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, ReturnFnParamsVec,3);  % Level 3 as DC1+GI
            daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,1,level1iidiff(ii),1,1)+N_d2*N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(daprime,[N_d2,(maxgap(ii)+1),level1iidiff(ii),N_a2]));
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:);
            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d2-1-by-n_a1-by-n_a2
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
    % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2]
    daprime=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
    entireRHS_ii=ReturnMatrix_ii+DiscountedEVinterp(reshape(daprime,[N_d2*n2long,N_a1*N_a2]));
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d2)+1;
    allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
    Policy3(1,:,N_j)=d_ind; % d2
    Policy3(2,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
    Policy3(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
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
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_u]

    Vlower=reshape(V(aprimeIndex(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
    Vupper=reshape(V(aprimeplus1Index(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u)
    % Already applied the probabilities from interpolating onto grid
    EV=sum((EV.*pi_u),3); % (d2,a1prime,a2)

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2]);
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]); % [N_d2,N_a1prime,1,N_a2]   
    
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n, n_a2, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want a1prime conditional on (d,1,a1,a2)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Just keep the 'midpoint' vesion of maxindex1 [as GI]
    midpoint(:,1,level1ii,:)=maxindex1;
    
    % Attempt for improved version
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d2-by-1-by-1-by-n_a2
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d2-by-maxgap(ii)+1-by-1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, 0, n_d2, maxgap(ii)+1, level1iidiff(ii), n_a2, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, ReturnFnParamsVec,3);  % Level 3 as DC1+GI
            daprime=(1:1:N_d2)'+N_d2*repelem(a1primeindexes-1,1,1,level1iidiff(ii),1)+N_d2*N_a1*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(daprime,[N_d2,(maxgap(ii)+1),level1iidiff(ii),N_a2]));
            [~,maxindex]=max(entireRHS_ii,[],2); % just a1prime
            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:);
            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
        end
    end
    
    % Turn this into the 'midpoint'
    midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d2-1-by-n_a1-by-n_a2
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
    % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2_noz(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, ReturnFnParamsVec,2); % [N_d,N_a1prime,N_a1,N_a2]
    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind; % the current aprimeii(ii):aprimeii(ii+1)
    entireRHS_ii=ReturnMatrix_ii+DiscountedEVinterp(reshape(d2a1primea2,[N_d2*n2long,N_a1*N_a2]));
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,jj)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d2)+1;
    allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
    Policy3(1,:,jj)=d_ind; % d2
    Policy3(2,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
    Policy3(3,:,jj)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

end


%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy3(3,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy3(2,:,:)=Policy3(2,:,:)-adjust; % lower grid point
Policy3(3,:,:)=adjust.*Policy3(3,:,:)+(1-adjust).*(Policy3(3,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%% For experience asset, just output Policy as single index and then use Case2 to UnKron
Policy=shiftdim(Policy3(1,:,:)+N_d2*(Policy3(2,:,:)-1)+N_d2*N_a1*(Policy3(3,:,:)-1),1);



end
