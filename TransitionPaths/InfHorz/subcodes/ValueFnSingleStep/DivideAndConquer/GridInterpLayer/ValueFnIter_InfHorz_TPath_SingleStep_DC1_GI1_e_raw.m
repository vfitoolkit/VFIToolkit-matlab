function [V,Policy2]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_e_raw(Vnext,n_d,n_a,n_z,n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step with divide-and-conquer and grid interpolation layer, with d, with z and iid e.
% Vnext is [N_a,N_z,N_e]; pi_z is [N_z,N_zprime]; pi_e is [N_e,1], the distribution of next period e.
% lowmemory: 0 is everything at once; 1 loops over e; 2 loops over z and e.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% Preallocate
if vfoptions.lowmemory==0
    midpoints=zeros(N_d,1,N_a,N_z,N_e,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e
    midpoints=zeros(N_d,1,N_a,N_z,'gpuArray');
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2 % loops over z and e
    midpoints=zeros(N_d,1,N_a,'gpuArray');
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
end

aind=0:1:N_a-1; % already includes -1
zind=shiftdim((0:1:N_z-1),-1); % already includes -1
eind=shiftdim((0:1:N_e-1),-2); % already includes -1
altzind=shiftdim((0:1:N_z-1),-2); % already includes -1


% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));


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

V=zeros(N_a,N_z,N_e,'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z,e (extra channel for PolicyL2flag pilot)

%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the condl expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
% First e', then z'
EV=Vnext.*shiftdim(pi_e,-2); % [N_a,N_z,N_e], aprime-by-zprime-by-eprime
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_z], aprime-by-zprime
EV=EV.*shiftdim(pi_z',-1); % [N_a,N_zprime,N_z]
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over z', leaving a singular second dimension, [N_a,1,N_z]
DiscountedEV=DiscountFactorParamsVec*shiftdim(EV,-1); % [1,aprime,1,z] — pre-discounted; broadcasts over d (and e) at every use site

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid); % [n2aprime,1,N_z]
DiscountedEVinterp=DiscountFactorParamsVec*shiftdim(EVinterp,-1); % [1,aprime,1,z] — pre-discounted

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals, e_gridvals, ReturnFnParamsVec,1);
    % (d,aprime,level1 a,z,e)

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % autoexpands over d, a and e

    % First, we want aprime conditional on (d,1,a,z,e)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints(:,1,level1ii,:,:)=maxindex1;

    % Attempt for improved version
    maxgap=max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1); % max over d,z,e
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(:,1,ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-1-by-n_z-by-n_e
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z-by-n_e
            % (reshape keeps a_grid(aprimeindexes) this shape when n_d=n_z=n_e=1, where indexing the column a_grid with a row would return a column)
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals, e_gridvals, ReturnFnParamsVec,6);
            % ReturnMatrix_ii is n_d-by-maxgap(ii)+1-by-length(curraindex)-by-n_z-by-n_e
            aprimez=aprimeindexes+N_a*altzind; % index into DiscountedEV [1,N_a,1,N_z] (z is 4th dim of aprimeindexes); EV does not depend on e, so no e offset
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimez(:)),[N_d,(maxgap(ii)+1),1,N_z,N_e]); % singleton dim 3 broadcasts over curraindex
            [~,maxindex]=max(entireRHS_ii,[],2);
            midpoints(:,1,curraindex,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii,:,:);
            midpoints(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1,1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints=max(min(midpoints,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is n_d-by-1-by-n_a-by-n_z-by-n_e
    aprimeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
    % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals,e_gridvals,ReturnFnParamsVec,2);
    % ReturnMatrix_ii is (n_d*n2long)-by-n_a-by-n_z-by-n_e
    aprimez=aprimeindexes+n2aprime*altzind; % index into DiscountedEVinterp [1,n2aprime,1,N_z]; EV does not depend on e, so no e offset
    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimez(:)),[N_d*n2long,N_a,N_z,N_e]);
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V=shiftdim(Vtempii,1); % [N_a,N_z,N_e]
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind+N_d*N_a*zind+N_d*N_a*N_z*eind; % midpoint is n_d-by-1-by-n_a-by-n_z-by-n_e
    Policy(1,:,:,:)=d_ind; % d
    Policy(2,:,:,:)=shiftdim(squeeze(midpoints(allind)),-1); % midpoint
    Policy(3,:,:,:)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    L2offset = ceil(maxindexL2/N_d);
    linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*zind + N_d*n2long*N_a*N_z*eind; % (d,L2=1,a,z,e) of the (n_d*n2long)-by-n_a-by-n_z-by-n_e ReturnMatrix_ii
    linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zind + N_d*n2long*N_a*N_z*eind; % (d,L2=n2long,a,z,e)
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    Policy(4,:,:,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

elseif vfoptions.lowmemory==1 % loop over e
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_gridvals, e_val, ReturnFnParamsVec,1);
        % (d,aprime,level1 a,z)

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % autoexpands over d and a

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints(:,1,level1ii,:)=maxindex1;

        % Attempt for improved version
        maxgap=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1); % max over d,z
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(:,1,ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
                % (reshape keeps a_grid(aprimeindexes) this shape when n_d=n_z=1, where indexing the column a_grid with a row would return a column)
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals, e_val, ReturnFnParamsVec,6);
                % ReturnMatrix_ii is n_d-by-maxgap(ii)+1-by-length(curraindex)-by-n_z
                aprimez=aprimeindexes+N_a*altzind; % index into DiscountedEV [1,N_a,1,N_z]
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimez(:)),[N_d,(maxgap(ii)+1),1,N_z]); % singleton dim 3 broadcasts over curraindex
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints=max(min(midpoints,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a-by-n_z
        aprimeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
        ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals,e_val,ReturnFnParamsVec,2);
        % ReturnMatrix_ii_e is (n_d*n2long)-by-n_a-by-n_z
        aprimez=aprimeindexes+n2aprime*altzind; % index into DiscountedEVinterp [1,n2aprime,1,N_z]
        entireRHS_ii_e=ReturnMatrix_ii_e+reshape(DiscountedEVinterp(aprimez(:)),[N_d*n2long,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
        V(:,:,e_c)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind; % midpoint is n_d-by-1-by-n_a-by-n_z
        Policy(1,:,:,e_c)=d_ind; % d
        Policy(2,:,:,e_c)=shiftdim(squeeze(midpoints(allind)),-1); % midpoint
        Policy(3,:,:,e_c)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset = ceil(maxindexL2/N_d);
        linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*zind; % (d,L2=1,a,z) of the (n_d*n2long)-by-n_a-by-n_z ReturnMatrix_ii_e
        linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zind; % (d,L2=n2long,a,z)
        isInfLower = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(4,:,:,e_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    end

elseif vfoptions.lowmemory==2 % loop over z and e
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV(:,:,:,z_c); % [1,N_a]
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,z_c); % [1,n2aprime]

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1);
            % (d,aprime,level1 a)

            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z; % autoexpands over d and a

            % First, we want aprime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints(:,1,level1ii)=maxindex1;

            % Attempt for improved version
            maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1); % max over d
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(:,1,ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1
                    % (reshape keeps a_grid(aprimeindexes) this shape when n_d=1, where indexing the column a_grid with a row would return a column)
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,6);
                    % ReturnMatrix_ii is n_d-by-maxgap(ii)+1-by-length(curraindex)
                    entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_z(aprimeindexes(:)),[N_d,(maxgap(ii)+1)]); % autoexpands over curraindex in 3rd dim
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints(:,1,curraindex)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    midpoints(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex));
                end
            end

            % Turn this into the 'midpoint'
            midpoints=max(min(midpoints,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a
            aprimeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii_ze=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
            % ReturnMatrix_ii_ze is (n_d*n2long)-by-n_a
            entireRHS_ii_ze=ReturnMatrix_ii_ze+reshape(DiscountedEVinterp_z(aprimeindexes(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
            V(:,z_c,e_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,z_c,e_c)=d_ind; % d
            Policy(2,:,z_c,e_c)=shiftdim(squeeze(midpoints(allind)),-1); % midpoint
            Policy(3,:,z_c,e_c)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d);
            linidx_lower = d_ind                  + N_d*n2long*aind; % (d,L2=1,a) of the (n_d*n2long)-by-n_a ReturnMatrix_ii_ze
            linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind; % (d,L2=n2long,a)
            isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(4,:,z_c,e_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end
end

%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%% Policy in transition paths
l_d=length(n_d);
Policy2=zeros(l_d+3,N_a,N_z,N_e); % +3 = midpoint, aprimeL2ind, L2flag (pilot)
% sort d variables
Policy2(1,:,:,:)=rem(Policy(1,:,:,:)-1,n_d(1))+1;
if l_d>1
    if l_d>2
        for ii=2:l_d-1
            Policy2(ii,:,:,:)=rem(ceil(Policy(1,:,:,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
        end
    end
    Policy2(l_d,:,:,:)=ceil(Policy(1,:,:,:)/prod(n_d(1:l_d-1)));
end
% rest are already in right shape
Policy2(l_d+1:end,:,:,:)=Policy(2:4,:,:,:);

end
