function [V,Policy]=ValueFnIter_FHorz_EpsteinZin_SemiExo_GI1_nod1_raw(n_d2,n_a,n_z,n_semiz,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Grid-interpolation-layer version of ValueFnIter_FHorz_EpsteinZin_SemiExo_nod1_raw.
% Grafts the Epstein-Zin transforms onto ValueFnIter_FHorz_SemiExo_GI1_nod1_raw.
% The interpolation acts on EV=E[(ezc4*V')^ezc5] (the d2-dependent expectations
% object, exactly where the vNM GI layer interpolates its EV); the
% certainty-equivalent power ^ezc6 and the rest of the transform chain are
% applied pointwise AFTER the interpolation. (Linear interpolation of EV
% commutes with affine maps, so at gamma=1/phi this reproduces the vNM GI layer
% exactly.) The warm-glow fn is evaluated exactly on the fine grid (no
% interpolation needed); the coarse pass uses its strided subset. V' is
% transformed by ^ezc5 once per age (d2-independent, pointwise) before the d2
% loop; the final max over d2 compares fully-transformed values, so it is
% unaffected.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,aprime seperately
Policy=zeros(4,N_a,N_semiz*N_z,N_j,'gpuArray'); % first dim indexes the optimal choice for d2, aprime and aprime2 (in GI layer)
Policy(4,:,:,:)=2; % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
% When ReturnFn is -Inf on one of the course grid points, we will allow fine index between that and the neighbouring course grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iteration

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory==2
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

aind=gpuArray(0:1:N_a-1); % already includes -1
aind2=gpuArray(1:1:N_a); % note, effectively no d, as loop over d2
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1, for lowmemory==1 semiz-block vectorisation

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
flag_ford2_jj=2*ones(N_a,N_semiz*N_z,N_d2,'gpuArray');

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
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
% (evaluated exactly on the fine grid; coarse pass uses the strided subset)
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

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d2, n_a, n_bothz, d2_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
        waszero=(ReturnMatrix==0);
        ReturnMatrix(becareful)=ReturnMatrix(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix now holds what was temp2 (avoids a full-size copy)
        ReturnMatrix(waszero)=-Inf;
        % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
        ReturnMatrix=ezc1*ReturnMatrix+ezc3*DiscountFactorParamsVec*shiftdim(WGmatrix,-1); % in-place: ReturnMatrix now holds what was entireRHS
        temp5=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0));
        ReturnMatrix(temp5)=ReturnMatrix(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
        ReturnMatrix(ReturnMatrix==0)=-Inf;
        % Treat standard problem as just being the first layer
        [~,maxindex]=max(ReturnMatrix,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_z
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d2,n_bothz,d2_gridvals,aprime_grid(aprimeindexes),a_grid,bothz_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
        waszero=(ReturnMatrix_ii==0);
        wasnegInf=(ReturnMatrix_ii==-Inf); % save raw -Inf pattern for the L2 flag (the in-place transform below overwrites ReturnMatrix_ii)
        ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
        ReturnMatrix_ii(waszero)=-Inf;
        % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
        ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrixfine(aprimeindexes),[N_d2*n2long,N_a,N_bothz]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
        temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
        ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*bothzind; % midpoint is n_d-by-1-by-n_a-by-n_z

        % L2 flag: detect -Inf on the coarse neighbour we'd put weight on
        L2offset      = ceil(maxindexL2/N_d2);
        linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*bothzind;
        linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzind;
        isInfLower    = wasnegInf(linidx_lower);
        isInfUpper    = wasnegInf(linidx_upper);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(4,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        Policy(1,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1 % parallel over semiz, loop over z

        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d2, n_a, [n_semiz,special_n_z], d2_gridvals, a_grid, z_valblock, ReturnFnParamsVec,1);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite but not zero
            waszero=(ReturnMatrix_z==0);
            ReturnMatrix_z(becareful)=ReturnMatrix_z(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_z now holds what was temp2 (avoids a full-size copy)
            ReturnMatrix_z(waszero)=-Inf;
            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
            ReturnMatrix_z=ezc1*ReturnMatrix_z+ezc3*DiscountFactorParamsVec*shiftdim(WGmatrix,-1); % in-place: ReturnMatrix_z now holds what was entireRHS
            temp5=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0));
            ReturnMatrix_z(temp5)=ReturnMatrix_z(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            ReturnMatrix_z(ReturnMatrix_z==0)=-Inf;
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(ReturnMatrix_z,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a-by-n_semiz
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d2,[n_semiz,special_n_z],d2_gridvals,aprime_grid(aprimeindexes),a_grid,z_valblock,ReturnFnParamsVec,2);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            waszero=(ReturnMatrix_ii==0);
            wasnegInf=(ReturnMatrix_ii==-Inf); % save raw -Inf pattern for the L2 flag (the in-place transform below overwrites ReturnMatrix_ii)
            ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
            ReturnMatrix_ii(waszero)=-Inf;
            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
            ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrixfine(aprimeindexes),[N_d2*n2long,N_a,N_semiz]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
            temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
            ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,semizblock,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*semizind; % midpoint is n_d-by-1-by-n_a-by-n_semiz

            % L2 flag: detect -Inf on the coarse neighbour we'd put weight on
            L2offset      = ceil(maxindexL2/N_d2);
            linidx_lower  = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*semizind;
            linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizind;
            isInfLower    = wasnegInf(linidx_lower);
            isInfUpper    = wasnegInf(linidx_upper);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(4,:,semizblock,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            Policy(1,:,semizblock,N_j)=d_ind; % d2
            Policy(2,:,semizblock,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,semizblock,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % aprimeL2ind
        end

    elseif vfoptions.lowmemory>=2 % joint loop over bothz

        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d2, n_a, special_n_bothz, d2_gridvals, a_grid, z_val, ReturnFnParamsVec,1);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite but not zero
            waszero=(ReturnMatrix_z==0);
            ReturnMatrix_z(becareful)=ReturnMatrix_z(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_z now holds what was temp2 (avoids a full-size copy)
            ReturnMatrix_z(waszero)=-Inf;
            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
            ReturnMatrix_z=ezc1*ReturnMatrix_z+ezc3*DiscountFactorParamsVec*shiftdim(WGmatrix,-1); % in-place: ReturnMatrix_z now holds what was entireRHS
            temp5=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0));
            ReturnMatrix_z(temp5)=ReturnMatrix_z(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            ReturnMatrix_z(ReturnMatrix_z==0)=-Inf;
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(ReturnMatrix_z,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d2,special_n_bothz,d2_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
            waszero=(ReturnMatrix_ii==0);
            wasnegInf=(ReturnMatrix_ii==-Inf); % save raw -Inf pattern for the L2 flag (the in-place transform below overwrites ReturnMatrix_ii)
            ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
            ReturnMatrix_ii(waszero)=-Inf;
            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
            ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrixfine(aprimeindexes),[N_d2*n2long,N_a]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
            temp5=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0));
            ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind; % midpoint is n_d-by-1-by-n_a

            % L2 flag: detect -Inf on the coarse neighbour we'd put weight on
            L2offset      = ceil(maxindexL2/N_d2);
            linidx_lower  = d_ind                   + N_d2*n2long*aind;
            linidx_upper  = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
            isInfLower    = wasnegInf(linidx_lower);
            isInfUpper    = wasnegInf(linidx_upper);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(4,:,z_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            Policy(1,:,z_c,N_j)=d_ind; % d2
            Policy(2,:,z_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,z_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % aprimeL2ind
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);    % switch V_Jplus1 into Kron form ([N_a,N_bothz]; was erroneously [N_a,N_semiz,N_z])

    % Part of Epstein-Zin is before taking expectation (d2-independent, so done once)
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0;

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
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

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, n_bothz, d2_val, a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
            % (aprime,a,z)
            becareful=logical(isfinite(ReturnMatrix_d2).*(ReturnMatrix_d2~=0)); % finite but not zero
            temp2=ReturnMatrix_d2;
            temp2(becareful)=ReturnMatrix_d2(becareful).^ezc2(N_j);
            temp2(ReturnMatrix_d2==0)=-Inf;
            entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;
            temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
            entireRHS(temp5)=entireRHS(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS(entireRHS==0)=-Inf;
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS,[],1); % no d1, loop over d2

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-n_bothz
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a-by-n_bothz
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
            becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_d2ii;
            temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
            aprimez=aprimeindexes+n2aprime*bothzind; % the current aprime
            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimez(:)),[n2long,N_a,N_bothz]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
            isInfLower    = (ReturnMatrix_d2ii(1,      :, :) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(n2long, :, :) == -Inf);
            inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
            inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
            flag_ford2_jj(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            allind=aind2+N_a*bothzind; % loweredge is 1-by-n_a-by-n_bothz
            midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoint(allind));

        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,N_j)=V_jj;
        Policy(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
        Policy(2,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]); % midpoint
        Policy(3,:,:,N_j)=aprimeL2_ind; % aprimeL2ind
        Policy(4,:,:,N_j)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);

    elseif vfoptions.lowmemory==1 % parallel over semiz, loop over z
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

                % Calc the condl expectation term (except beta): loop z, vectorize over semiz
                EV_d2z=temp.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
                EVinterp_d2z=interp1(a_grid,EV_d2z,aprime_grid); % [n2aprime, 1, N_semiz]

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

                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, [n_semiz,special_n_z], d2_val, a_grid, z_valblock, ReturnFnParamsVec,0);
                becareful=logical(isfinite(ReturnMatrix_d2z).*(ReturnMatrix_d2z~=0)); % finite but not zero
                temp2=ReturnMatrix_d2z;
                temp2(becareful)=ReturnMatrix_d2z(becareful).^ezc2(N_j);
                temp2(ReturnMatrix_d2z==0)=-Inf;
                entireRHS_z=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;
                temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
                entireRHS_z(temp5)=entireRHS_z(temp5).^ezc7(N_j);
                entireRHS_z(entireRHS_z==0)=-Inf;
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(entireRHS_z,[],1); % no d1, loop over d2

                % Turn maxindex into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is 1-by-n_a-by-n_semiz
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
                % aprime possibilities are n2long-by-n_a-by-n_semiz
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, aprime_grid(aprimeindexes), a_grid, z_valblock, ReturnFnParamsVec,5);
                becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_d2ii;
                temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
                aprimez=aprimeindexes+n2aprime*semizind; % the current aprime (offset into the semiz block)
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimez(:)),[n2long,N_a,N_semiz]);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,semizblock,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,semizblock,d2_c)=shiftdim(maxindex,1);

                % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
                isInfLower    = (ReturnMatrix_d2ii(1,      :, :) == -Inf);
                isInfUpper    = (ReturnMatrix_d2ii(n2long, :, :) == -Inf);
                inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
                inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
                flag_ford2_jj(:,semizblock,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                allind=aind2+N_a*semizind; % loweredge is 1-by-n_a-by-n_semiz
                midpoint_ford2_jj(:,semizblock,d2_c)=squeeze(midpoint(allind));
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,N_j)=V_jj;
        Policy(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
        Policy(2,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]); % midpoint
        Policy(3,:,:,N_j)=aprimeL2_ind; % aprimeL2ind
        Policy(4,:,:,N_j)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);

    elseif vfoptions.lowmemory>=2 % joint loop over bothz
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
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

                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, special_n_bothz, d2_val, a_grid, z_val, ReturnFnParamsVec,0);
                becareful=logical(isfinite(ReturnMatrix_d2z).*(ReturnMatrix_d2z~=0)); % finite but not zero
                temp2=ReturnMatrix_d2z;
                temp2(becareful)=ReturnMatrix_d2z(becareful).^ezc2(N_j);
                temp2(ReturnMatrix_d2z==0)=-Inf;
                entireRHS_z=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;
                temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
                entireRHS_z(temp5)=entireRHS_z(temp5).^ezc7(N_j);
                entireRHS_z(entireRHS_z==0)=-Inf;
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(entireRHS_z,[],1); % no d1, loop over d2

                % Turn maxindex into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is 1-by-n_a
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
                % aprime possibilities are n2long-by-n_a
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_bothz, d2_val, aprime_grid(aprimeindexes), a_grid, z_val, ReturnFnParamsVec,5);
                becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_d2ii;
                temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimeindexes(:)),[n2long,N_a]);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,z_c,d2_c)=shiftdim(maxindex,1);

                % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
                isInfLower    = (ReturnMatrix_d2ii(1,      :) == -Inf);
                isInfUpper    = (ReturnMatrix_d2ii(n2long, :) == -Inf);
                inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
                inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
                flag_ford2_jj(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                midpoint_ford2_jj(:,z_c,d2_c)=midpoint;
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,N_j)=V_jj;
        Policy(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
        Policy(2,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]); % midpoint
        Policy(3,:,:,N_j)=aprimeL2_ind; % aprimeL2ind
        Policy(4,:,:,N_j)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
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

    % If there is a warm-glow, evaluate the warmglowfn (exactly on the fine grid; coarse pass uses the strided subset)
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
        WGmatrixfine=WGmatrixfineraw;
        WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(jj);
        WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
        WGmatrix=WGmatrixfine(1:(n2short+1):end); % coarse-grid subset
        WGmatrix=WGmatrix(:); % column over the coarse aprime grid
    end

    EVpre=V(:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation (d2-independent, so done once)
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj)); % reverse order

            EV_d2=temp.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid); % (aprime,1,bothz)

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

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, n_bothz, d2_val, a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,0);
            % (aprime,a,z)
            becareful=logical(isfinite(ReturnMatrix_d2).*(ReturnMatrix_d2~=0)); % finite but not zero
            temp2=ReturnMatrix_d2;
            temp2(becareful)=ReturnMatrix_d2(becareful).^ezc2(jj);
            temp2(ReturnMatrix_d2==0)=-Inf;
            entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;
            temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
            entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS(entireRHS==0)=-Inf;
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS,[],1); % no d1, loop over d2

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-n_bothz
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a-by-n_bothz
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_bothz, d2_val, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
            becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_d2ii;
            temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
            aprimez=aprimeindexes+n2aprime*bothzind; % the current aprime
            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimez(:)),[n2long,N_a,N_bothz]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
            entireRHS_ii(entireRHS_ii==0)=-Inf;
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);

            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1); % aprimeL2ind

            % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
            isInfLower    = (ReturnMatrix_d2ii(1,      :, :) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(n2long, :, :) == -Inf);
            inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
            inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
            flag_ford2_jj(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            allind=aind2+N_a*bothzind; % loweredge is 1-by-n_a-by-n_bothz
            midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoint(allind)); % midpoint

        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,jj)=V_jj;
        Policy(1,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
        Policy(2,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]); % midpoint
        Policy(3,:,:,jj)=aprimeL2_ind; % aprimeL2ind
        Policy(4,:,:,jj)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);


    elseif vfoptions.lowmemory==1 % parallel over semiz, loop over z
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj)); % reverse order

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

                % Calc the condl expectation term (except beta): loop z, vectorize over semiz
                EV_d2z=temp.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
                EVinterp_d2z=interp1(a_grid,EV_d2z,aprime_grid); % [n2aprime, 1, N_semiz]

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

                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, [n_semiz,special_n_z], d2_val, a_grid, z_valblock, ReturnFnParamsVec,0);
                becareful=logical(isfinite(ReturnMatrix_d2z).*(ReturnMatrix_d2z~=0)); % finite but not zero
                temp2=ReturnMatrix_d2z;
                temp2(becareful)=ReturnMatrix_d2z(becareful).^ezc2(jj);
                temp2(ReturnMatrix_d2z==0)=-Inf;
                entireRHS_z=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;
                temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
                entireRHS_z(temp5)=entireRHS_z(temp5).^ezc7(jj);
                entireRHS_z(entireRHS_z==0)=-Inf;
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(entireRHS_z,[],1); % no d1, loop over d2

                % Turn maxindex into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is 1-by-n_a-by-n_semiz
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
                % aprime possibilities are n2long-by-n_a-by-n_semiz
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, aprime_grid(aprimeindexes), a_grid, z_valblock, ReturnFnParamsVec,5);
                becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_d2ii;
                temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
                aprimez=aprimeindexes+n2aprime*semizind; % the current aprime (offset into the semiz block)
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimez(:)),[n2long,N_a,N_semiz]);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,semizblock,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,semizblock,d2_c)=shiftdim(maxindex,1);

                % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
                isInfLower    = (ReturnMatrix_d2ii(1,      :, :) == -Inf);
                isInfUpper    = (ReturnMatrix_d2ii(n2long, :, :) == -Inf);
                inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
                inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
                flag_ford2_jj(:,semizblock,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                allind=aind2+N_a*semizind; % loweredge is 1-by-n_a-by-n_semiz
                midpoint_ford2_jj(:,semizblock,d2_c)=squeeze(midpoint(allind));
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,jj)=V_jj;
        Policy(1,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
        Policy(2,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]); % midpoint
        Policy(3,:,:,jj)=aprimeL2_ind; % aprimeL2ind
        Policy(4,:,:,jj)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);

    elseif vfoptions.lowmemory>=2 % joint loop over bothz
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj)); % reverse order

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
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

                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, special_n_bothz, d2_val, a_grid, z_val, ReturnFnParamsVec,0);
                becareful=logical(isfinite(ReturnMatrix_d2z).*(ReturnMatrix_d2z~=0)); % finite but not zero
                temp2=ReturnMatrix_d2z;
                temp2(becareful)=ReturnMatrix_d2z(becareful).^ezc2(jj);
                temp2(ReturnMatrix_d2z==0)=-Inf;
                entireRHS_z=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;
                temp5=logical(isfinite(entireRHS_z).*(entireRHS_z~=0));
                entireRHS_z(temp5)=entireRHS_z(temp5).^ezc7(jj);
                entireRHS_z(entireRHS_z==0)=-Inf;
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(entireRHS_z,[],1); % no d1, loop over d2

                % Turn maxindex into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is 1-by-n_a
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
                % aprime possibilities are n2long-by-n_a
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_bothz, d2_val, aprime_grid(aprimeindexes), a_grid, z_val, ReturnFnParamsVec,5);
                becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_d2ii;
                temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_d2ii==0)=-Inf;
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4interp(aprimeindexes(:)),[n2long,N_a]);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,z_c,d2_c)=shiftdim(maxindex,1);

                % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
                isInfLower    = (ReturnMatrix_d2ii(1,      :) == -Inf);
                isInfUpper    = (ReturnMatrix_d2ii(n2long, :) == -Inf);
                inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
                inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
                flag_ford2_jj(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                midpoint_ford2_jj(:,z_c,d2_c)=midpoint;
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,jj)=V_jj;
        Policy(1,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
        Policy(2,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]); % midpoint
        Policy(3,:,:,jj)=aprimeL2_ind; % aprimeL2ind
        Policy(4,:,:,jj)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    end

end

%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% Policy=squeeze(Policy(1,:,:,:,:)+N_d2*(Policy(2,:,:,:,:)-1)+N_d2*N_a*(Policy(3,:,:,:,:)-1)+N_d2*N_a*(n2short+2)*(Policy(4,:,:,:,:)-1));

end
