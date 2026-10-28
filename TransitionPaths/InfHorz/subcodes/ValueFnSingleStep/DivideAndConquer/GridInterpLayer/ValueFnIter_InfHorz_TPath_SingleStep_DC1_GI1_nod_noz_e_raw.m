function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_nod_noz_e_raw(Vnext,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step with divide-and-conquer and grid interpolation layer, no d, no z, with iid e.
% Vnext is [N_a,N_e]; pi_e is [N_e,1], the distribution of next period e.
% lowmemory: 0 is everything at once; 1 (and 2) loop over e.

N_a=prod(n_a);
N_e=prod(n_e);

% Preallocate
if vfoptions.lowmemory==0
    midpoints=zeros(1,N_a,N_e,'gpuArray');
elseif vfoptions.lowmemory>=1 % loops over e
    midpoints=zeros(1,N_a,'gpuArray');
    special_n_e=ones(1,length(n_e));
end

aind=0:1:N_a-1; % already includes -1
eind=shiftdim((0:1:N_e-1),-1); % already includes -1

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

V=zeros(N_a,N_e,'gpuArray');
Policy=zeros(3,N_a,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,e (extra channel for PolicyL2flag pilot)

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the condl expectation term (except beta), computed once and used by every lowmemory branch
EV=Vnext.*shiftdim(pi_e,-1); % [N_a,N_e], aprime-by-eprime
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,2); % sum over e', [N_a,1]; EV does not depend on e (it is iid), so is broadcast over e

% Interpolate EV over aprime_grid
EVinterp=interp1(a_grid,EV,aprime_grid);

if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals, ReturnFnParamsVec,1);
    % (aprime,level1 a,e), n_a-by-level1n-by-n_e

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [~,maxindex1]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints(1,level1ii,:)=maxindex1;

    % Attempt for improved version
    maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3); % max over e
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-n_e
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are maxgap(ii)+1-by-1-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals, ReturnFnParamsVec,2);
            % ReturnMatrix_ii is maxgap(ii)+1-by-length(curraindex)-by-n_e
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimeindexes(:)),[maxgap(ii)+1,1,N_e]); % EV does not depend on e, so no e offset; autoexpands over curraindex in 2nd dim
            [~,maxindex]=max(entireRHS_ii,[],1);
            midpoints(1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(1,ii,:);
            midpoints(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints=max(min(midpoints,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    % midpoint is 1-by-n_a-by-n_e
    aprimeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a-by-n_e
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_e,aprime_grid(aprimeindexes),a_grid,e_gridvals,ReturnFnParamsVec,2);
    % ReturnMatrix_ii is n2long-by-n_a-by-n_e
    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a,N_e]); % EV does not depend on e, so no e offset in the index
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V=shiftdim(Vtempii,1); % [N_a,N_e]
    Policy(1,:,:)=midpoints; % midpoint
    Policy(2,:,:)=maxindexL2; % aprimeL2ind
    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    linidx_lower = 1      + n2long*aind + n2long*N_a*eind; % (1,a,e) of the n2long-by-n_a-by-n_e ReturnMatrix_ii
    linidx_upper = n2long + n2long*aind + n2long*N_a*eind; % (n2long,a,e)
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,:) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

elseif vfoptions.lowmemory>=1 % loop over e

    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);
        % (aprime,level1 a), n_a-by-level1n

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;

        % Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints(1,level1ii)=maxindex1;

        % Attempt for improved version
        maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are maxgap(ii)+1-by-1
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                % ReturnMatrix_ii is maxgap(ii)+1-by-length(curraindex)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(aprimeindexes); % EV(aprimeindexes) is maxgap(ii)+1-by-1, autoexpands over curraindex
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints(1,curraindex)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,ii);
                midpoints(1,curraindex)=repelem(loweredge,1,length(curraindex));
            end
        end

        % Turn this into the 'midpoint'
        midpoints=max(min(midpoints,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a
        aprimeindexes=(midpoints+(midpoints-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a
        ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,special_n_e,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
        % ReturnMatrix_ii_e is n2long-by-n_a
        entireRHS_ii_e=ReturnMatrix_ii_e+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
        [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
        V(:,e_c)=shiftdim(Vtempii,1);
        Policy(1,:,e_c)=midpoints; % midpoint
        Policy(2,:,e_c)=maxindexL2; % aprimeL2ind
        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_lower = 1      + n2long*aind; % (1,a) of the n2long-by-n_a ReturnMatrix_ii_e
        linidx_upper = n2long + n2long*aind; % (n2long,a)
        isInfLower = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        Policy(3,:,e_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    end
end

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:)=Policy(1,:,:)-adjust; % lower grid point
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)


end
