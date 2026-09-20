function [V,Policy]=ValueFnIter_FHorz_GulPesendorferExpAsset_GI1_nod1_noz_raw(n_d2,n_a1,n_a2,N_j, d2_gridvals, a1_gridvals, a2_grid, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions)
% Gul-Pesendorfer with an experience asset and the grid interpolation layer on a1prime:
% V_j = max [u+v+beta*E V_{j+1}] - max v, both maxes over the same joint (d,a1prime) choice set.
% The tempted objective goes through the standard ExpAsset GI machinery (the continuation is at
% a2prime=aprimeFn(d2,a2), interpolated onto the a2 grid; a1prime is chosen on the fine grid).
% The choice set is the fine a1prime grid, so the most-tempting term is the max of v over the
% FINE grid, found by the same two-stage scheme as the main max but around v's OWN coarse argmax
% (otherwise the chosen fine point could be more tempting than the coarse max of v, making the
% self-control cost negative). The L2 -Inf flag is based on u+v. The most-tempting term is
% subtracted after the max (v never touches the continuation, so the temptation side needs none
% of the aprime-probs machinery).

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(3,N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

% n_a1prime=n_a;
% a1prime_gridvals=a1_gridvals;
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=1, Refine=0
    TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, TemptationFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=1, Refine=0
    % Calc the max and it's index
    [~,maxindex]=max(ReturnMatrix+TemptationMatrix,[],2);

    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
    [~,maxindexT]=max(TemptationMatrix,[],2);
    midpointT=max(min(maxindexT,n_a1(1)-1),2);
    a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
    TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
    MostTempting=max(TemptationMatrix_Tii,[],1);
    MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-1-by-n_a1-by-n_a2
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
    TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
    Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
    [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii-MostTempting,1);
    d_ind=rem(maxindexL2-1,N_d2)+1;
    allind=d_ind+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
    Policy(1,:,N_j)=d_ind; % d2
    Policy(2,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
    Policy(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
    L2offset      = ceil(maxindexL2/N_d2);
    linidx_lower  = d_ind                   + N_d2*n2long*aind;
    linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
    isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
    isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2]

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);

    Vlower=reshape(EV(aprimeIndex(:)),[N_d2*N_a1,N_a2]);
    Vupper=reshape(EV(aprimeplus1Index(:)),[N_d2*N_a1,N_a2]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    % Already applied the probabilities from interpolating onto grid

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2]);
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]); % [N_d2,N_a1prime,1,N_a2]

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=1, Refine=0

    TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, TemptationFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=1, Refine=0
    entireRHS=ReturnMatrix+TemptationMatrix+DiscountedEV; % autofill 3rd dim to N_a1

    % Calc the max and it's index
    [~,maxindex]=max(entireRHS,[],2);

    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
    [~,maxindexT]=max(TemptationMatrix,[],2);
    midpointT=max(min(maxindexT,n_a1(1)-1),2);
    a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
    TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
    MostTempting=max(TemptationMatrix_Tii,[],1);
    MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-1-by-n_a1-by-n_a2
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
    TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
    Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
    entireRHS_ii=Ftemp_ii+reshape(DiscountedEVinterp(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii-MostTempting,1);
    d_ind=rem(maxindexL2-1,N_d2)+1;
    allind=d_ind+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
    Policy(1,:,N_j)=d_ind; % d2
    Policy(2,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
    Policy(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
    L2offset      = ceil(maxindexL2/N_d2);
    linidx_lower  = d_ind                   + N_d2*n2long*aind;
    linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
    isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
    isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2]

    Vlower=reshape(V(aprimeIndex(:),jj+1),[N_d2*N_a1,N_a2]);
    Vupper=reshape(V(aprimeplus1Index(:),jj+1),[N_d2*N_a1,N_a2]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    % Already applied the probabilities from interpolating onto grid

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2]);
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]); % [N_d2,N_a1prime,1,N_a2]

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=1, Refine=0

    TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, TemptationFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=1, Refine=0
    entireRHS=ReturnMatrix+TemptationMatrix+DiscountedEV; % autofill 3rd dim to N_a1

    % Calc the max and it's index
    [~,maxindex]=max(entireRHS,[],2);

    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
    [~,maxindexT]=max(TemptationMatrix,[],2);
    midpointT=max(min(maxindexT,n_a1(1)-1),2);
    a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
    TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
    MostTempting=max(TemptationMatrix_Tii,[],1);
    MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

    % Turn this into the 'midpoint'
    midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-1-by-n_a1-by-n_a2
    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
    TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n2long, n_a1,n_a2, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
    Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
    entireRHS_ii=Ftemp_ii+reshape(DiscountedEVinterp(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V(:,jj)=shiftdim(Vtempii-MostTempting,1);
    d_ind=rem(maxindexL2-1,N_d2)+1;
    allind=d_ind+N_d2*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
    Policy(1,:,jj)=d_ind; % d2
    Policy(2,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
    Policy(3,:,jj)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind
    % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
    L2offset      = ceil(maxindexL2/N_d2);
    linidx_lower  = d_ind                   + N_d2*n2long*aind;
    linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
    isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
    isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

end



%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:)=Policy(2,:,:)-adjust; % lower grid point
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)


Policy=[Policy;PolicyL2flag];


end
