function [V,Policy]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_GI1_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, n_e,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin version of ValueFnIter_FHorz_SemiExo_DC1_GI1_e_raw.
% The certainty-equivalent is taken over the JOINT distribution of
% (semizprime,zprime,eprime), which depends on the chosen d2: V' is
% transformed by ^ezc5 once per age (d2-independent, pointwise), the
% eprime-expectation (also d2-independent) is taken on the transformed
% object, then the d2-dependent (semizprime,zprime)-expectation is taken, and
% the interpolation acts on EV=E[(ezc4*V')^ezc5] (the expectations object,
% exactly where the vNM GI layer interpolates its EV); the
% certainty-equivalent power ^ezc6 and the rest of the transform chain are
% applied pointwise AFTER the interpolation. The return transform and the
% final ^ezc7 wrap each entireRHS before its max (a monotone transform, so
% the divide-and-conquer monotonicity logic is unaffected). The warm-glow fn
% is evaluated exactly on the fine grid (no interpolation needed); the coarse
% passes use its strided subset. The final max over d2 compares
% fully-transformed values, so it is unaffected.

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod(n_d); % Needed for N_j when converting to form of Policy3
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy=zeros(5,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy(5,:,:,:,:)=2; % L2 flag: 1=all to lower, 2=usual, 3=all to upper
% First dimension: d1, d2, aprime, aprime2

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]); % version to use when looping over d2

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_e=ones(1,length(n_e));
end
aind=gpuArray(0:1:N_a-1); % already includes -1
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
bothzind2=shiftdim(gpuArray(0:1:N_bothz-1),-2); % already includes -1
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1 (for lowmemory==2 split: outer z, inner e, vectorize semiz)
semizind2=shiftdim(gpuArray(0:1:N_semiz-1),-2); % already includes -1 (for lowmemory==2 split)

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
PolicyL2flag_ford2_jj=2*ones(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d1,1,N_a,N_semiz*N_z,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(N_d1,1,N_a,N_semiz*N_z,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoints_jj=zeros(N_d1,1,N_a,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==3
    midpoints_jj=zeros(N_d1,1,N_a,'gpuArray');
end

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
if length(n_a)>1
    error('can only do gridinterplayer with one endo state (you have length(n_a)>1)')
end
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
elseif vfoptions.EZoneminusbeta==2
    ezc1=1-sj(N_j)*DiscountFactorParamsVec;
end

% If there is a warm-glow at end of the final period, evaluate the warmglowfn
% (evaluated exactly on the fine grid; coarse passes use the strided subset)
if warmglow==1
    WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,N_j);
    WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
    WGmatrixfine=WGmatrixfineraw;
    WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(N_j);
    WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrixfine==0);
        % The warm-glow enters INSIDE the ^ezc7 root at the terminal age, so build the
        % temp4-analogue of the main loop (with no EV term):
        WGmatrixfine(isfinite(WGmatrixfine))=((1-sj(N_j))*WGmatrixfine(isfinite(WGmatrixfine)).^ezc8(N_j)).^ezc6(N_j);
        WGmatrixfine(becareful)=0;
    end
    WGmatrix=WGmatrixfine(1:(n2short+1):end); % coarse-grid subset
    WGmatrix=WGmatrix(:); % column over the coarse aprime grid
else
    WGmatrixfine=zeros(n2aprime,1,'gpuArray');
    WGmatrix=zeros(N_a,1,'gpuArray');
end

if ~isfield(vfoptions,'V_Jplus1')

    if vfoptions.lowmemory==0
        midpoints_Nj=zeros(N_d,1,N_a,N_semiz*N_z,N_e,'gpuArray');

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        waszero=(ReturnMatrix_ii==0);
        ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
        ReturnMatrix_ii(waszero)=-Inf;
        % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
        ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*shiftdim(WGmatrix,-1); % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
        temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
        ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;

        % First, we want aprime conditional on (d,1,a,z,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_Nj(:,1,level1ii,:,:)=maxindex1;

        % Second level based on monotonicity
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,6);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                waszero=(ReturnMatrix_ii==0);
                ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                ReturnMatrix_ii(waszero)=-Inf;
                % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix(aprimeindexes); % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
                ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                [~,maxindex]=max(ReturnMatrix_ii,[],2); % ,2) as GI
                midpoints_Nj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_Nj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
            end
        end

        % Midpoints
        midpoints_Nj=max(min(midpoints_Nj,n_a-1),2);

        % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, [n_d1,n_d2], n_bothz, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        waszero=(ReturnMatrix_ii==0);
        wasnegInf=(ReturnMatrix_ii==-Inf); % save raw -Inf pattern for the L2 flag (the in-place transform below overwrites ReturnMatrix_ii)
        ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
        ReturnMatrix_ii(waszero)=-Inf;
        % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
        ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrixfine(aprimeindexes),[N_d*n2long,N_a,N_bothz,N_e]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
        temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
        ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*bothzind+N_d*N_a*N_bothz*eind; % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        Policy(1,:,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy(2,:,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy(3,:,:,:,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1); % midpoint
        Policy(4,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); %aprimeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset = ceil(maxindexL2/N_d);
        linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*bothzind + N_d*n2long*N_a*N_bothz*eind;
        linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*bothzind + N_d*n2long*N_a*N_bothz*eind;
        isInfLower = wasnegInf(linidx_lower);
        isInfUpper = wasnegInf(linidx_upper);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(5,:,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    elseif vfoptions.lowmemory==1

        midpoints_Nj=zeros(N_d,1,N_a,N_semiz*N_z,'gpuArray');

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_bothz, special_n_e, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            waszero=(ReturnMatrix_ii==0);
            ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
            ReturnMatrix_ii(waszero)=-Inf;
            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
            ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*shiftdim(WGmatrix,-1); % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
            temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
            ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_Nj(:,1,level1ii,:)=maxindex1;

            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_bothz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_bothz, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,6);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    waszero=(ReturnMatrix_ii==0);
                    ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                    ReturnMatrix_ii(waszero)=-Inf;
                    % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                    ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix(aprimeindexes); % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                    temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
                    ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                    ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                    [~,maxindex]=max(ReturnMatrix_ii,[],2); % ,2) as GI
                    midpoints_Nj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_Nj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
                end
            end

            % Midpoints
            midpoints_Nj=max(min(midpoints_Nj,n_a-1),2);

            % midpoint is n_d-by-1-by-n_a-by-n_bothz
            aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, [n_d1,n_d2], n_bothz, special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            waszero=(ReturnMatrix_ii==0);
            wasnegInf=(ReturnMatrix_ii==-Inf); % save raw -Inf pattern for the L2 flag (the in-place transform below overwrites ReturnMatrix_ii)
            ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
            ReturnMatrix_ii(waszero)=-Inf;
            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
            ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrixfine(aprimeindexes),[N_d*n2long,N_a,N_bothz]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
            temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
            ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*bothzind; % midpoint is n_d-by-1-by-n_a-by-n_bothz
            Policy(1,:,:,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy(2,:,:,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy(3,:,:,e_c,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1); % midpoint
            Policy(4,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); %aprimeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d);
            linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
            linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
            isInfLower = wasnegInf(linidx_lower);
            isInfUpper = wasnegInf(linidx_upper);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(5,:,:,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz

        midpoints_Nj=zeros(N_d,1,N_a,N_semiz,'gpuArray');

        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, [n_semiz,special_n_z], special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_valblock, e_val, ReturnFnParamsVec,1);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                waszero=(ReturnMatrix_ii==0);
                ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                ReturnMatrix_ii(waszero)=-Inf;
                % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*shiftdim(WGmatrix,-1); % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
                ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;

                % First, we want aprime conditional on (d,1,a,z)
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_Nj(:,1,level1ii,:)=maxindex1;

                % Second level based on monotonicity
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_semiz
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, [n_semiz,special_n_z], special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,6);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        waszero=(ReturnMatrix_ii==0);
                        ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                        ReturnMatrix_ii(waszero)=-Inf;
                        % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                        ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix(aprimeindexes); % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                        temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
                        ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                        [~,maxindex]=max(ReturnMatrix_ii,[],2); % ,2) as GI
                        midpoints_Nj(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoints_Nj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
                    end
                end

                % Midpoints
                midpoints_Nj=max(min(midpoints_Nj,n_a-1),2);

                % midpoint is n_d-by-1-by-n_a-by-n_semiz
                aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, [n_d1,n_d2], [n_semiz,special_n_z], special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, e_val, ReturnFnParamsVec,2);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                waszero=(ReturnMatrix_ii==0);
                wasnegInf=(ReturnMatrix_ii==-Inf); % save raw -Inf pattern for the L2 flag (the in-place transform below overwrites ReturnMatrix_ii)
                ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                ReturnMatrix_ii(waszero)=-Inf;
                % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrixfine(aprimeindexes),[N_d*n2long,N_a,N_semiz]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
                ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V(:,semizblock,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind+N_d*N_a*semizind; % midpoint is n_d-by-1-by-n_a-by-n_semiz
                Policy(1,:,semizblock,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
                Policy(2,:,semizblock,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
                Policy(3,:,semizblock,e_c,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1); % midpoint
                Policy(4,:,semizblock,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); %aprimeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d);
                linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*semizind;
                linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*semizind;
                isInfLower = wasnegInf(linidx_lower);
                isInfUpper = wasnegInf(linidx_upper);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(5,:,semizblock,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end

    elseif vfoptions.lowmemory>=3 % joint bothz, inner e

        midpoints_Nj=zeros(N_d,1,N_a,'gpuArray');

        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_bothz, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                waszero=(ReturnMatrix_ii==0);
                ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                ReturnMatrix_ii(waszero)=-Inf;
                % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*shiftdim(WGmatrix,-1); % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
                ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;

                % First, we want aprime conditional on (d,1,a,z)
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_Nj(:,1,level1ii)=maxindex1;

                % Second level based on monotonicity
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_bothz, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,6);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        waszero=(ReturnMatrix_ii==0);
                        ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                        ReturnMatrix_ii(waszero)=-Inf;
                        % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                        ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix(aprimeindexes); % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                        temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
                        ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                        [~,maxindex]=max(ReturnMatrix_ii,[],2); % ,2) as GI
                        midpoints_Nj(:,1,curraindex)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        midpoints_Nj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex)); % unfortunately doesn't autofill
                    end
                end

                % Midpoints
                midpoints_Nj=max(min(midpoints_Nj,n_a-1),2);

                % midpoint is n_d-by-1-by-n_a
                aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, [n_d1,n_d2], special_n_bothz, special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, e_val, ReturnFnParamsVec,2);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                waszero=(ReturnMatrix_ii==0);
                wasnegInf=(ReturnMatrix_ii==-Inf); % save raw -Inf pattern for the L2 flag (the in-place transform below overwrites ReturnMatrix_ii)
                ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                ReturnMatrix_ii(waszero)=-Inf;
                % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrixfine(aprimeindexes),[N_d*n2long,N_a]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
                ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
                Policy(1,:,z_c,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
                Policy(2,:,z_c,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
                Policy(3,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1); % midpoint
                Policy(4,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); %aprimeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d);
                linidx_lower = d_ind                  + N_d*n2long*aind;
                linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
                isInfLower = wasnegInf(linidx_lower);
                isInfUpper = wasnegInf(linidx_upper);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(5,:,z_c,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]); % First, switch V_Jplus1 into Kron form

    % Part of Epstein-Zin is before taking expectation (d2-independent, so done once)
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0;

    % Take expectation over e
    temp=sum(temp.*pi_e_J(1,1,:,N_j+1),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            EV_d2=temp.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,bothz)
            temp4=EV_d2;
            temp4interp=EVinterp_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
                WGmatrixfinebig=WGmatrixfine.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixfinebig)); % both are finite
                temp4interp(becareful)=(sj(N_j)*temp4interp(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixfinebig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4interp((EVinterp_d2==0)&(WGmatrixfinebig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV_d2==0)=0;
                temp4interp(isfinite(temp4interp))=(sj(N_j)*temp4interp(isfinite(temp4interp)).^ezc8(N_j)).^ezc6(N_j);
                temp4interp(EVinterp_d2==0)=0;
            end

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz,n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_d2ii;
            temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            % First, we want aprime conditional on (d,1,a,z,e)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(:,1,level1ii,:,:)=maxindex1;

            % Second level based on monotonicity
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,6);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    aprimez=aprimeindexes+N_a*bothzind2;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1,(maxgap(ii)+1),1,N_bothz,N_e]); % autoexpand level1iidiff(ii) in 3rd-dim
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii));
                end
            end

            % Now for the interpolation layer

            % Turn maxindex into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;
            aprimez=aprimeindexes+n2aprime*bothzind2; % the current aprime
            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimez),[N_d1*n2long,N_a,N_bothz,N_e]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind+N_d1*N_a*N_semiz*N_z*eind; % loweredge is n_d-by-1-by-n_a-by-n_semiz-by-n_e
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoints_jj(allind));

            % L2 flag for this d2
            L2offset_d2 = ceil(maxindex/N_d1);
            linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
            inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
            PolicyL2flag_ford2_jj(:,:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,N_j)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]); % d1
        Policy(4,:,:,:,N_j)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,N_j)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);


    elseif vfoptions.lowmemory==1
         for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            EV_d2=temp.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,bothz)
            temp4=EV_d2;
            temp4interp=EVinterp_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
                WGmatrixfinebig=WGmatrixfine.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixfinebig)); % both are finite
                temp4interp(becareful)=(sj(N_j)*temp4interp(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixfinebig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4interp((EVinterp_d2==0)&(WGmatrixfinebig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV_d2==0)=0;
                temp4interp(isfinite(temp4interp))=(sj(N_j)*temp4interp(isfinite(temp4interp)).^ezc8(N_j)).^ezc6(N_j);
                temp4interp(EVinterp_d2==0)=0;
            end

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
                becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_d2ii;
                temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                % First, we want aprime conditional on (d,1,a,z)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,level1ii,:)=maxindex1;

                % Second level based on monotonicity
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d1-by-1-by-1-by-n_bothz
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,6);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        aprimez=aprimeindexes+N_a*bothzind2;
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1,(maxgap(ii)+1),1,N_bothz]); % autoexpand level1iidiff(ii) in 3rd-dim
                        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                        entireRHS_ii(entireRHS_ii==0)=-Inf;
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii));
                    end
                end

                % Now for the interpolation layer

                % Turn maxindex into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                % midpoint is n_d-by-1-by-n_a-by-n_bothz
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                aprimez=aprimeindexes+n2aprime*bothzind2; % the current aprime
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimez),[N_d1*n2long,N_a,N_bothz]);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind; % loweredge is n_d-by-1-by-n_a-by-n_semiz
                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoints_jj(allind));

                % L2 flag for this (d2, e_c)
                L2offset_d2 = ceil(maxindex/N_d1);
                linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                PolicyL2flag_ford2_jj(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,N_j)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]); % d1
        Policy(4,:,:,:,N_j)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,N_j)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

                EV_d2z=temp.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
                EVinterp_d2z=interp1(a_grid,EV_d2z,aprime_grid);

                % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,semiz)
                temp4=EV_d2z;
                temp4interp=EVinterp_d2z;
                if warmglow==1
                    WGmatrixbig=WGmatrix.*ones(1,1,N_semiz);
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_d2z==0)&(WGmatrixbig==0))=0; % Is actually zero
                    WGmatrixfinebig=WGmatrixfine.*ones(1,1,N_semiz);
                    becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixfinebig)); % both are finite
                    temp4interp(becareful)=(sj(N_j)*temp4interp(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixfinebig(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4interp((EVinterp_d2z==0)&(WGmatrixfinebig==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_d2z==0)=0;
                    temp4interp(isfinite(temp4interp))=(sj(N_j)*temp4interp(isfinite(temp4interp)).^ezc8(N_j)).^ezc6(N_j);
                    temp4interp(EVinterp_d2z==0)=0;
                end

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    % n-Monotonicity
                    ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), z_valblock, e_val, ReturnFnParamsVec,1);
                    becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_d2ii;
                    temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    % First, we want aprime conditional on (d,1,a,z)
                    [~,maxindex1]=max(entireRHS_ii,[],2);

                    % Just keep the 'midpoint' version of maxindex1 [as GI]
                    midpoints_jj(:,1,level1ii,:)=maxindex1;

                    % Second level based on monotonicity
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            % loweredge is n_d1-by-1-by-1-by-n_semiz
                            aprimeindexes=loweredge+(0:1:maxgap(ii));
                            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,6);
                            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                            temp2_ii=ReturnMatrix_ii;
                            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                            temp2_ii(ReturnMatrix_ii==0)=-Inf;
                            aprimez=aprimeindexes+N_a*semizind2;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1,(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 3rd-dim
                            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                            entireRHS_ii(entireRHS_ii==0)=-Inf;
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii));
                        end
                    end

                    % Now for the interpolation layer

                    % Turn maxindex into the 'midpoint'
                    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                    % midpoint is n_d-by-1-by-n_a-by-n_semiz
                    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                    % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, e_val, ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    aprimez=aprimeindexes+n2aprime*semizind2; % the current aprime
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimez),[N_d1*n2long,N_a,N_semiz]);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                    V_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(maxindex,1);

                    d1_ind=rem(maxindex-1,N_d1)+1;
                    allind=d1_ind+N_d1*aind+N_d1*N_a*semizind; % loweredge is n_d-by-1-by-n_a-by-n_semiz
                    midpoint_ford2_jj(:,semizblock,e_c,d2_c)=squeeze(midpoints_jj(allind));

                    % L2 flag for this (d2, z, e_c)
                    L2offset_d2 = ceil(maxindex/N_d1);
                    linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                    linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                    inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                    PolicyL2flag_ford2_jj(:,semizblock,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,N_j)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]); % d1
        Policy(4,:,:,:,N_j)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,N_j)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory>=3 % joint bothz, inner e
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                EV_d2z=temp.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2);

                % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
                EVinterp_d2z=interp1(a_grid,EV_d2z,aprime_grid'); % (column, to match WGmatrixfine)

                % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
                temp4=EV_d2z;
                temp4interp=EVinterp_d2z;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_d2z==0)&(WGmatrix==0))=0; % Is actually zero
                    becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixfine)); % both are finite
                    temp4interp(becareful)=(sj(N_j)*temp4interp(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixfine(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4interp((EVinterp_d2z==0)&(WGmatrixfine==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_d2z==0)=0;
                    temp4interp(isfinite(temp4interp))=(sj(N_j)*temp4interp(isfinite(temp4interp)).^ezc8(N_j)).^ezc6(N_j);
                    temp4interp(EVinterp_d2z==0)=0;
                end

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    % n-Monotonicity
                    ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                    becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_d2ii;
                    temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    % First, we want aprime conditional on (d,1,a,z)
                    [~,maxindex1]=max(entireRHS_ii,[],2);

                    % Just keep the 'midpoint' version of maxindex1 [as GI]
                    midpoints_jj(:,1,level1ii)=maxindex1;

                    % Second level based on monotonicity
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            aprimeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,6);
                            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                            temp2_ii=ReturnMatrix_ii;
                            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                            temp2_ii(ReturnMatrix_ii==0)=-Inf;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimeindexes),[N_d1,(maxgap(ii)+1)]); % autoexpand level1iidiff(ii) in 3rd-dim
                            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                            entireRHS_ii(entireRHS_ii==0)=-Inf;
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii);
                            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,level1iidiff(ii));
                        end
                    end

                    % Now for the interpolation layer

                    % Turn maxindex into the 'midpoint'
                    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, e_val, ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimeindexes),[N_d1*n2long,N_a]);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                    V_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(maxindex,1);

                    d1_ind=rem(maxindex-1,N_d1)+1;
                    allind=d1_ind+N_d1*aind; % loweredge is n_d-by-1-by-n_a
                    midpoint_ford2_jj(:,z_c,e_c,d2_c)=squeeze(midpoints_jj(allind));

                    % L2 flag for this (d2, z, e_c)
                    L2offset_d2 = ceil(maxindex/N_d1);
                    linidx_lower = d1_ind                  + N_d1*n2long*aind;
                    linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                    inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                    PolicyL2flag_ford2_jj(:,z_c,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,N_j)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]); % d1
        Policy(4,:,:,:,N_j)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,N_j)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

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
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % If there is a warm-glow, evaluate the warmglowfn (exactly on the fine grid; coarse passes use the strided subset)
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
        WGmatrixfine=WGmatrixfineraw;
        WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(jj);
        WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
        WGmatrix=WGmatrixfine(1:(n2short+1):end); % coarse-grid subset
        WGmatrix=WGmatrix(:); % column over the coarse aprime grid
    end

    EVpre=V(:,:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation (d2-independent, so done once)
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    % Take expectations over e
    temp=sum(temp.*pi_e_J(1,1,:,jj+1),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj)); % reverse order

            EV_d2=temp.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,bothz)
            temp4=EV_d2;
            temp4interp=EVinterp_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
                WGmatrixfinebig=WGmatrixfine.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixfinebig)); % both are finite
                temp4interp(becareful)=(sj(jj)*temp4interp(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixfinebig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4interp((EVinterp_d2==0)&(WGmatrixfinebig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV_d2==0)=0;
                temp4interp(isfinite(temp4interp))=(sj(jj)*temp4interp(isfinite(temp4interp)).^ezc8(jj)).^ezc6(jj);
                temp4interp(EVinterp_d2==0)=0;
            end

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
            becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_d2ii;
            temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            % First, we want aprime conditional on (d,1,a,z,e)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(:,1,level1ii,:,:)=maxindex1;

            % Second level based on monotonicity
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d1-by-1-by-1-by-n_bothz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,6);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    aprimez=aprimeindexes+N_a*bothzind2;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1,(maxgap(ii)+1),1,N_bothz,N_e]); % autoexpand level1iidiff(ii) in 3rd-dim
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii));
                end
            end

            % Now for the interpolation layer

            % Turn maxindex into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;
            aprimez=aprimeindexes+n2aprime*bothzind2; % the current aprime
            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimez),[N_d1*n2long,N_a,N_bothz,N_e]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind+N_d1*N_a*N_semiz*N_z*eind; % loweredge is n_d-by-1-by-n_a-by-n_semiz-by-n_e
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoints_jj(allind));

            % L2 flag for this d2
            L2offset_d2 = ceil(maxindex/N_d1);
            linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
            inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
            PolicyL2flag_ford2_jj(:,:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]); % d1
        Policy(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,jj)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);


elseif vfoptions.lowmemory==1

        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj)); % reverse order

            EV_d2=temp.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,bothz)
            temp4=EV_d2;
            temp4interp=EVinterp_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
                WGmatrixfinebig=WGmatrixfine.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixfinebig)); % both are finite
                temp4interp(becareful)=(sj(jj)*temp4interp(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixfinebig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4interp((EVinterp_d2==0)&(WGmatrixfinebig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV_d2==0)=0;
                temp4interp(isfinite(temp4interp))=(sj(jj)*temp4interp(isfinite(temp4interp)).^ezc8(jj)).^ezc6(jj);
                temp4interp(EVinterp_d2==0)=0;
            end

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);
                becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_d2ii;
                temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                % First, we want aprime conditional on (d,1,a,z)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,level1ii,:)=maxindex1;

                % Second level based on monotonicity
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d1-by-1-by-1-by-n_bothz
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,6);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        aprimez=aprimeindexes+N_a*bothzind2;
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1,(maxgap(ii)+1),1,N_bothz]); % autoexpand level1iidiff(ii) in 3rd-dim
                        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                        entireRHS_ii(entireRHS_ii==0)=-Inf;
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii));
                    end
                end

                % Now for the interpolation layer

                % Turn maxindex into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                % midpoint is n_d-by-1-by-n_a-by-n_bothz
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                aprimez=aprimeindexes+n2aprime*bothzind2; % the current aprime
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimez),[N_d1*n2long,N_a,N_bothz]);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind; % loweredge is n_d-by-1-by-n_a-by-n_semiz
                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoints_jj(allind));

                % L2 flag for this (d2, e_c)
                L2offset_d2 = ceil(maxindex/N_d1);
                linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                PolicyL2flag_ford2_jj(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]); % d1
        Policy(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,jj)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj)); % reverse order

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

                EV_d2z=temp.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
                EVinterp_d2z=interp1(a_grid,EV_d2z,aprime_grid);

                % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,semiz)
                temp4=EV_d2z;
                temp4interp=EVinterp_d2z;
                if warmglow==1
                    WGmatrixbig=WGmatrix.*ones(1,1,N_semiz);
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_d2z==0)&(WGmatrixbig==0))=0; % Is actually zero
                    WGmatrixfinebig=WGmatrixfine.*ones(1,1,N_semiz);
                    becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixfinebig)); % both are finite
                    temp4interp(becareful)=(sj(jj)*temp4interp(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixfinebig(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4interp((EVinterp_d2z==0)&(WGmatrixfinebig==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_d2z==0)=0;
                    temp4interp(isfinite(temp4interp))=(sj(jj)*temp4interp(isfinite(temp4interp)).^ezc8(jj)).^ezc6(jj);
                    temp4interp(EVinterp_d2z==0)=0;
                end

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    % n-Monotonicity
                    ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), z_valblock, e_val, ReturnFnParamsVec,1);
                    becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_d2ii;
                    temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    % First, we want aprime conditional on (d,1,a,z)
                    [~,maxindex1]=max(entireRHS_ii,[],2);

                    % Just keep the 'midpoint' version of maxindex1 [as GI]
                    midpoints_jj(:,1,level1ii,:)=maxindex1;

                    % Second level based on monotonicity
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            % loweredge is n_d1-by-1-by-1-by-n_semiz
                            aprimeindexes=loweredge+(0:1:maxgap(ii));
                            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,6);
                            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                            temp2_ii=ReturnMatrix_ii;
                            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                            temp2_ii(ReturnMatrix_ii==0)=-Inf;
                            aprimez=aprimeindexes+N_a*semizind2;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1,(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 3rd-dim
                            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                            entireRHS_ii(entireRHS_ii==0)=-Inf;
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii));
                        end
                    end

                    % Now for the interpolation layer

                    % Turn maxindex into the 'midpoint'
                    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                    % midpoint is n_d-by-1-by-n_a-by-n_semiz
                    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                    % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, e_val, ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    aprimez=aprimeindexes+n2aprime*semizind2; % the current aprime
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimez),[N_d1*n2long,N_a,N_semiz]);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                    V_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(maxindex,1);

                    d1_ind=rem(maxindex-1,N_d1)+1;
                    allind=d1_ind+N_d1*aind+N_d1*N_a*semizind; % loweredge is n_d-by-1-by-n_a-by-n_semiz
                    midpoint_ford2_jj(:,semizblock,e_c,d2_c)=squeeze(midpoints_jj(allind));

                    % L2 flag for this (d2, z, e_c)
                    L2offset_d2 = ceil(maxindex/N_d1);
                    linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                    linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                    inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                    PolicyL2flag_ford2_jj(:,semizblock,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]); % d1
        Policy(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,jj)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory>=3 % joint bothz, inner e
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj)); % reverse order

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

                EV_d2z=temp.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2);

                % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
                EVinterp_d2z=interp1(a_grid,EV_d2z,aprime_grid'); % (column, to match WGmatrixfine)

                % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
                temp4=EV_d2z;
                temp4interp=EVinterp_d2z;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_d2z==0)&(WGmatrix==0))=0; % Is actually zero
                    becareful=logical(isfinite(temp4interp).*isfinite(WGmatrixfine)); % both are finite
                    temp4interp(becareful)=(sj(jj)*temp4interp(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixfine(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4interp((EVinterp_d2z==0)&(WGmatrixfine==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_d2z==0)=0;
                    temp4interp(isfinite(temp4interp))=(sj(jj)*temp4interp(isfinite(temp4interp)).^ezc8(jj)).^ezc6(jj);
                    temp4interp(EVinterp_d2z==0)=0;
                end

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    % n-Monotonicity
                    ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
                    becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_d2ii;
                    temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    % First, we want aprime conditional on (d,1,a,z)
                    [~,maxindex1]=max(entireRHS_ii,[],2);

                    % Just keep the 'midpoint' version of maxindex1 [as GI]
                    midpoints_jj(:,1,level1ii)=maxindex1;

                    % Second level based on monotonicity
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            aprimeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,6);
                            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                            temp2_ii=ReturnMatrix_ii;
                            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                            temp2_ii(ReturnMatrix_ii==0)=-Inf;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimeindexes),[N_d1,(maxgap(ii)+1)]); % autoexpand level1iidiff(ii) in 3rd-dim
                            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                            entireRHS_ii(entireRHS_ii==0)=-Inf;
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii);
                            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,level1iidiff(ii));
                        end
                    end

                    % Now for the interpolation layer

                    % Turn maxindex into the 'midpoint'
                    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
                    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, e_val, ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimeindexes),[N_d1*n2long,N_a]);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                    V_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(maxindex,1);

                    d1_ind=rem(maxindex-1,N_d1)+1;
                    allind=d1_ind+N_d1*aind; % loweredge is n_d-by-1-by-n_a
                    midpoint_ford2_jj(:,z_c,e_c,d2_c)=squeeze(midpoints_jj(allind));

                    % L2 flag for this (d2, z, e_c)
                    L2offset_d2 = ceil(maxindex/N_d1);
                    linidx_lower = d1_ind                  + N_d1*n2long*aind;
                    linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                    inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                    PolicyL2flag_ford2_jj(:,z_c,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]); % d1
        Policy(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,jj)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    end
end

%% Currently Policy(3,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(3,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust; % lower grid point
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% Policy=squeeze(Policy(1,:,:,:,:)+N_d1*(Policy(2,:,:,:,:)-1)+N_d*(Policy(3,:,:,:,:)-1)+N_d*N_a*(Policy(4,:,:,:,:)-1)+N_d*N_a*(n2short+2)*(Policy(5,:,:,:,:)-1));



end
