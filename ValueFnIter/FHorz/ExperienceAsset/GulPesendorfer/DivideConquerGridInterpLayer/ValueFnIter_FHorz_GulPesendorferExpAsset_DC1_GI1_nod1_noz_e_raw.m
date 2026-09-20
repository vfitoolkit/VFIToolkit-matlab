function [V,Policy]=ValueFnIter_FHorz_GulPesendorferExpAsset_DC1_GI1_nod1_noz_e_raw(n_d2,n_a1,n_a2,n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, e_gridvals_J, pi_e_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions)
% Gul-Pesendorfer with an experience asset, divide-and-conquer plus the grid interpolation
% layer (both on a1). The DC pass runs on the tempted objective u+v+beta*EV and produces the
% midpoints for the GI refinement; the tempted continuation goes through the standard ExpAsset
% machinery (a2prime=aprimeFn(d2,a2), interpolated onto the a2 grid). The most-tempting term is
% the max of v over the FINE (d,a1prime) choice set, found by its own two-stage scheme: v's
% coarse argmax is computed EXACTLY over the full a1prime grid (full-column temptation matrices
% one a1-slab at a time -- never a window max, and no monotonicity assumption on v), then
% refined on the fine points around it. The most-tempting term is subtracted from V after the
% max (v never touches the continuation, so the temptation side needs none of the aprime-probs
% machinery).

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z
PolicyL2flag=2*ones(1,N_a,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
    % Preallocate
    midpoint=zeros(N_d2,1,N_a1,N_a2,'gpuArray');
    midpointT=zeros(N_d2,1,N_a1,N_a2,'gpuArray'); % v's own coarse argmax (for the most-tempting term)
else
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_e,'gpuArray');
    midpointT=zeros(N_d2,1,N_a1,N_a2,N_e,'gpuArray'); % v's own coarse argmax (for the most-tempting term)
end

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
eind=shiftdim(gpuArray(0:1:N_e-1),-1); % already includes -1
eindB=shiftdim((0:1:N_e-1),-1); % already includes -1

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0
        TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,1,0); % Level=1, Refine=0

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(ReturnMatrix_ii+TemptationMatrix_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoint(:,1,level1ii,:,:)=maxindex1;

        % v's own coarse argmax at the level-1 stations (full a1prime grid)
        [~,maxindexT1]=max(TemptationMatrix_ii,[],2);
        midpointT(:,1,level1ii,:,:)=maxindexT1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
            % v's coarse argmax over the FULL a1prime grid for these a1 (never just the window)
            TemptationMatrix_full=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,length(curraindex),n_a2,n_e, d2_gridvals, a1_gridvals, a1_gridvals(curraindex), a2_gridvals, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,1,0);
            [~,maxindexT]=max(TemptationMatrix_full,[],2);
            midpointT(:,1,curraindex,:,:)=maxindexT;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                [~,maxindex]=max(ReturnMatrix_ii+TemptationMatrix_ii,[],2);
                midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_e
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
        % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n2long,n_a1,n_a2,n_e, d2_gridvals, a1prime_grid(a1primeindexes), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
        TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,n_e, d2_gridvals, a1prime_grid(a1primeindexes), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
        Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
        [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);
        % Most-tempting term: fine refinement of v around v's own coarse argmax
        midpointT=max(min(midpointT,n_a1(1)-1),2);
        a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,n_e, d2_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,2,0);
        MostTempting=max(TemptationMatrix_Tii,[],1);
        MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        V(:,:,N_j)=shiftdim(Vtempii-MostTempting,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*eindB; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_e
        Policy(1,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset = ceil(maxindexL2/N_d2);
        linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*eindB;
        linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*eindB;
        isInfLower = (Ftemp_ii(linidx_lower) == -Inf);
        isInfUpper = (Ftemp_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
            TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, TemptationFnParamsVec,1,0); % Level=1, Refine=0

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii+TemptationMatrix_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:)=maxindex1;

            % v's own coarse argmax at the level-1 stations (full a1prime grid)
            [~,maxindexT1]=max(TemptationMatrix_ii,[],2);
            midpointT(:,1,level1ii,:)=maxindexT1;

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                % v's coarse argmax over the FULL a1prime grid for these a1 (never just the window)
                TemptationMatrix_full=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,length(curraindex),n_a2,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals(curraindex), a2_gridvals, e_val, TemptationFnParamsVec,1,0);
                [~,maxindexT]=max(TemptationMatrix_full,[],2);
                midpointT(:,1,curraindex,:)=maxindexT;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                    TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, TemptationFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                    [~,maxindex]=max(ReturnMatrix_ii+TemptationMatrix_ii,[],2);
                    midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
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
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n2long,n_a1,n_a2,special_n_e, d2_gridvals, a1prime_grid(a1primeindexes), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
            TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,special_n_e, d2_gridvals, a1prime_grid(a1primeindexes), a1_gridvals, a2_gridvals, e_val, TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
            Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
            [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);
            % Most-tempting term: fine refinement of v around v's own coarse argmax
            midpointT=max(min(midpointT,n_a1(1)-1),2);
            a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, e_val, TemptationFnParamsVec,2,0);
            MostTempting=max(TemptationMatrix_Tii,[],1);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
            V(:,e_c,N_j)=shiftdim(Vtempii-MostTempting,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
            Policy(1,:,e_c,N_j)=d_ind; % d2
            Policy(2,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                   + N_d2*n2long*aind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
            isInfLower = (Ftemp_ii(linidx_lower) == -Inf);
            isInfUpper = (Ftemp_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2]

    EV=sum(pi_e_J(:,N_j+1)'.*reshape(vfoptions.V_Jplus1,[N_a,N_e]),2);    % Expectations over e

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

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2]); % (d,a1prime,1,a2)
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]);   % [N_d2,N_a1prime,1,N_a2]

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0
        TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,1,0); % Level=1, Refine=0

        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountedEV; % autofill e for DiscountedentireEV

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoint(:,1,level1ii,:,:)=maxindex1;

        % v's own coarse argmax at the level-1 stations (full a1prime grid)
        [~,maxindexT1]=max(TemptationMatrix_ii,[],2);
        midpointT(:,1,level1ii,:,:)=maxindexT1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
            % v's coarse argmax over the FULL a1prime grid for these a1 (never just the window)
            TemptationMatrix_full=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,length(curraindex),n_a2,n_e, d2_gridvals, a1_gridvals, a1_gridvals(curraindex), a2_gridvals, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,1,0);
            [~,maxindexT]=max(TemptationMatrix_full,[],2);
            midpointT(:,1,curraindex,:,:)=maxindexT;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind; % [N_d2,maxgap+1,1,N_a2,N_e]; linear index into DiscountedEV [N_d2,N_a1,1,N_a2]
                entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountedEV(d2aprime);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_e
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
        % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n2long,n_a1,n_a2,n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
        TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
        Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
        da1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind; % [N_d2,n2long,N_a1,N_a2,N_e]; linear index into DiscountedEVinterp [N_d2,N_a1prime,1,N_a2]
        entireRHS_ii=Ftemp_ii+reshape(DiscountedEVinterp(da1primea2),[N_d2*n2long,N_a1*N_a2,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        % Most-tempting term: fine refinement of v around v's own coarse argmax
        midpointT=max(min(midpointT,n_a1(1)-1),2);
        a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,n_e, d2_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,2,0);
        MostTempting=max(TemptationMatrix_Tii,[],1);
        MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        V(:,:,N_j)=shiftdim(Vtempii-MostTempting,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*eindB; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_e
        Policy(1,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset = ceil(maxindexL2/N_d2);
        linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*eindB;
        linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*eindB;
        isInfLower = (Ftemp_ii(linidx_lower) == -Inf);
        isInfUpper = (Ftemp_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
            TemptationMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, TemptationFnParamsVec,1,0); % Level=1, Refine=0

            entireRHS_ii_e=ReturnMatrix_ii_e+TemptationMatrix_ii_e+DiscountedEV;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_e,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:)=maxindex1;

            % v's own coarse argmax at the level-1 stations (full a1prime grid)
            [~,maxindexT1]=max(TemptationMatrix_ii_e,[],2);
            midpointT(:,1,level1ii,:)=maxindexT1;

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                % v's coarse argmax over the FULL a1prime grid for these a1 (never just the window)
                TemptationMatrix_full=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,length(curraindex),n_a2,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals(curraindex), a2_gridvals, e_val, TemptationFnParamsVec,1,0);
                [~,maxindexT]=max(TemptationMatrix_full,[],2);
                midpointT(:,1,curraindex,:)=maxindexT;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                    TemptationMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, TemptationFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                    d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind; % [N_d2,maxgap+1,1,N_a2]; linear index into DiscountedEV [N_d2,N_a1,1,N_a2]
                    entireRHS_ii_e=ReturnMatrix_ii_e+TemptationMatrix_ii_e+DiscountedEV(d2aprime);
                    [~,maxindex]=max(entireRHS_ii_e,[],2);
                    midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n2long,n_a1,n_a2,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_val, TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
            da1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind; % [N_d2,n2long,N_a1,N_a2]; linear index into DiscountedEVinterp [N_d2,N_a1prime,1,N_a2]
            entireRHS_ii=Ftemp_ii+reshape(DiscountedEVinterp(da1primea2),[N_d2*n2long,N_a1*N_a2]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            % Most-tempting term: fine refinement of v around v's own coarse argmax
            midpointT=max(min(midpointT,n_a1(1)-1),2);
            a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, e_val, TemptationFnParamsVec,2,0);
            MostTempting=max(TemptationMatrix_Tii,[],1);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
            V(:,e_c,N_j)=shiftdim(Vtempii-MostTempting,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
            Policy(1,:,e_c,N_j)=d_ind; % d2
            Policy(2,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                   + N_d2*n2long*aind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
            isInfLower = (Ftemp_ii(linidx_lower) == -Inf);
            isInfUpper = (Ftemp_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

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
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2]

    EV=sum(pi_e_J(:,jj+1)'.*V(:,:,jj+1),2); % Expectations over e

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

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2]); % (d,a1prime,1,a2)
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4]),a1prime_grid),[2,1,3,4]);   % [N_d2,N_a1prime,1,N_a2]

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % Level=1, Refine=0
        TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,jj), TemptationFnParamsVec,1,0); % Level=1, Refine=0

        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountedEV; % autofill e for DiscountedentireEV

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoint(:,1,level1ii,:,:)=maxindex1;

        % v's own coarse argmax at the level-1 stations (full a1prime grid)
        [~,maxindexT1]=max(TemptationMatrix_ii,[],2);
        midpointT(:,1,level1ii,:,:)=maxindexT1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
            % v's coarse argmax over the FULL a1prime grid for these a1 (never just the window)
            TemptationMatrix_full=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,length(curraindex),n_a2,n_e, d2_gridvals, a1_gridvals, a1_gridvals(curraindex), a2_gridvals, e_gridvals_J(:,:,jj), TemptationFnParamsVec,1,0);
            [~,maxindexT]=max(TemptationMatrix_full,[],2);
            midpointT(:,1,curraindex,:,:)=maxindexT;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,jj), TemptationFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind; % [N_d2,maxgap+1,1,N_a2,N_e]; linear index into DiscountedEV [N_d2,N_a1,1,N_a2]
                entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountedEV(d2aprime);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_e
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
        % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n2long,n_a1,n_a2,n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
        TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,jj), TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
        Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
        da1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind; % [N_d2,n2long,N_a1,N_a2,N_e]; linear index into DiscountedEVinterp [N_d2,N_a1prime,1,N_a2]
        entireRHS_ii=Ftemp_ii+reshape(DiscountedEVinterp(da1primea2),[N_d2*n2long,N_a1*N_a2,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        % Most-tempting term: fine refinement of v around v's own coarse argmax
        midpointT=max(min(midpointT,n_a1(1)-1),2);
        a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,n_e, d2_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, e_gridvals_J(:,:,jj), TemptationFnParamsVec,2,0);
        MostTempting=max(TemptationMatrix_Tii,[],1);
        MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        V(:,:,jj)=shiftdim(Vtempii-MostTempting,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*eindB; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_e
        Policy(1,:,:,jj)=d_ind; % d2
        Policy(2,:,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,jj)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset = ceil(maxindexL2/N_d2);
        linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*eindB;
        linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*eindB;
        isInfLower = (Ftemp_ii(linidx_lower) == -Inf);
        isInfUpper = (Ftemp_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,jj) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
            TemptationMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, TemptationFnParamsVec,1,0); % Level=1, Refine=0

            entireRHS_ii_e=ReturnMatrix_ii_e+TemptationMatrix_ii_e+DiscountedEV;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_e,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:)=maxindex1;

            % v's own coarse argmax at the level-1 stations (full a1prime grid)
            [~,maxindexT1]=max(TemptationMatrix_ii_e,[],2);
            midpointT(:,1,level1ii,:)=maxindexT1;

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                % v's coarse argmax over the FULL a1prime grid for these a1 (never just the window)
                TemptationMatrix_full=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n_a1,length(curraindex),n_a2,special_n_e, d2_gridvals, a1_gridvals, a1_gridvals(curraindex), a2_gridvals, e_val, TemptationFnParamsVec,1,0);
                [~,maxindexT]=max(TemptationMatrix_full,[],2);
                midpointT(:,1,curraindex,:)=maxindexT;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                    TemptationMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, TemptationFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                    d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind; % [N_d2,maxgap+1,1,N_a2]; linear index into DiscountedEV [N_d2,N_a1,1,N_a2]
                    entireRHS_ii_e=ReturnMatrix_ii_e+TemptationMatrix_ii_e+DiscountedEV(d2aprime);
                    [~,maxindex]=max(entireRHS_ii_e,[],2);
                    midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d2,n2long,n_a1,n_a2,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            TemptationMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, e_val, TemptationFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
            da1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind; % [N_d2,n2long,N_a1,N_a2]; linear index into DiscountedEVinterp [N_d2,N_a1prime,1,N_a2]
            entireRHS_ii=Ftemp_ii+reshape(DiscountedEVinterp(da1primea2),[N_d2*n2long,N_a1*N_a2]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            % Most-tempting term: fine refinement of v around v's own coarse argmax
            midpointT=max(min(midpointT,n_a1(1)-1),2);
            a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, 0,n_d2,n2long,n_a1,n_a2,special_n_e, d2_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, e_val, TemptationFnParamsVec,2,0);
            MostTempting=max(TemptationMatrix_Tii,[],1);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
            V(:,e_c,jj)=shiftdim(Vtempii-MostTempting,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
            Policy(1,:,e_c,jj)=d_ind; % d2
            Policy(2,:,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d2),-1); % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                   + N_d2*n2long*aind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
            isInfLower = (Ftemp_ii(linidx_lower) == -Inf);
            isInfUpper = (Ftemp_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,e_c,jj) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

        end
    end
end




%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy;PolicyL2flag];

% %% For experience asset, just output Policy as single index and then use Case2 to UnKron
% Policy=shiftdim(Policy(1,:,:,:)+N_d2*(Policy(2,:,:,:)-1)+N_d2*N_a1*(Policy(3,:,:,:)-1)+N_d2*N_a1*(n2short+2)*(PolicyL2flag-1),1);


end