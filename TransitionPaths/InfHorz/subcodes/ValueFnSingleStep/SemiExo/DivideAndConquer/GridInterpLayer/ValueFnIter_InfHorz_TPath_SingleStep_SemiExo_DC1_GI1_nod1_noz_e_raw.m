function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_DC1_GI1_nod1_noz_e_raw(Vnext,n_d2,n_a,n_semiz,n_e, d2_gridvals, a_grid, semiz_gridvals, e_gridvals, pi_semiz, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step with divide-and-conquer and grid interpolation layer, with semi-exogenous state semiz, with d2 (determines the transition of semiz) and no d1, no z, with iid e.
% Vnext is [N_a,N_semiz,N_e] (next period value fn)
% pi_semiz is [N_semiz,N_semizprime,N_d2] (this period's transition); pi_e is [N_e,1], the distribution of next period e
% Policy is returned as kron rows [d2; aprime lower grid point; L2 index (1 to ngridinterp+2); L2flag], [4,N_a,N_semiz,N_e]
% lowmemory: 0 is everything at once; 1 loops over e.
% With lowmemory=1 the first (monotonicity) level is solved for every e before the second level, so that maxgap is the
% max over e, exactly as with lowmemory=0 (and so the two give identical answers).

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%%
special_n_d2=ones(1,length(n_d2));

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));

if vfoptions.lowmemory==0
    midpoints=zeros(1,N_a,N_semiz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e
    midpoints=zeros(1,N_a,N_semiz,'gpuArray');
    maxindex1=zeros(1,length(level1ii),N_semiz,N_e,'gpuArray'); % first level, for all e
    special_n_e=ones(1,length(n_e));
end

semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1, [1,1,N_semiz]

% Preallocate
V_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray'); % L2 index on the second layer
midpoint_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
L2flag_ford2=2*ones(N_a,N_semiz,N_e,N_d2,'gpuArray'); % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
% When ReturnFn is -Inf on one of the coarse grid points, we allow fine index between that and the neighbouring coarse grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iterating

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

%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the condl expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
% First e', then semiz' (given d2)
EV=reshape(Vnext,[N_a,N_semiz,N_e]).*reshape(pi_e,[1,1,N_e]); % [N_a,N_semizprime,N_eprime]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_semizprime]
EV=EV.*permute(pi_semiz,[4,2,1,3]); % [N_a,N_semizprime,N_semiz,N_d2]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,2),[N_a,N_semiz,N_d2]); % sum over semiz', aprime-by-semiz-by-d2

% Interpolate EV over aprime_grid
EVinterp=reshape(interp1(a_grid,reshape(EV,[N_a,numel(EV)/N_a]),aprime_grid),[n2aprime,N_semiz,N_d2]); % aprime(fine)-by-semiz-by-d2

%%
if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        EV_d2=EV(:,:,d2_c); % [N_a,N_semiz]
        EVinterp_d2=EVinterp(:,:,d2_c); % [n2aprime,N_semiz]

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals, e_gridvals, ReturnFnParamsVec,4); % [N_a,level1n,N_semiz,N_e]
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2,[N_a,1,N_semiz]); % autoexpands over a and e
        % First, we want aprime conditional on (1,a,semiz,e)
        [~,maxindex1]=max(entireRHS_ii,[],1); % no d1

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints(1,level1ii,:,:)=maxindex1;

        % Second level based on monotonicity
        maxgap=max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3); % max over semiz and e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1-by-n_semiz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(curraindex), semiz_gridvals, e_gridvals, ReturnFnParamsVec,5); % [maxgap(ii)+1,length(curraindex),N_semiz,N_e]
                aprimez=aprimeindexes+N_a*semizind; % index into EV_d2 [N_a,N_semiz]; EV does not depend on e, so no e offset
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_semiz,N_e]); % singleton dim 2 broadcasts over curraindex
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints(1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,ii,:,:);
                midpoints(1,curraindex,:,:)=repelem(loweredge,1,length(curraindex),1,1);
            end
        end

        % Now for the interpolation layer
        % Turn midpoints into the 'midpoint'
        midpoint=max(min(midpoints,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a-by-n_semiz-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a-by-n_semiz-by-n_e
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, reshape(aprime_grid(aprimeindexes),size(aprimeindexes)), a_grid, semiz_gridvals, e_gridvals, ReturnFnParamsVec,5); % [n2long,N_a,N_semiz,N_e]
        aprimez=aprimeindexes+n2aprime*semizind; % index into EVinterp_d2 [n2aprime,N_semiz]; EV does not depend on e, so no e offset
        entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_semiz,N_e]);
        [Vtemp,maxindex]=max(entireRHS_ii,[],1);

        V_ford2(:,:,:,d2_c)=reshape(Vtemp,[N_a,N_semiz,N_e]);
        Policy_ford2(:,:,:,d2_c)=reshape(maxindex,[N_a,N_semiz,N_e]); % aprimeL2ind
        midpoint_ford2(:,:,:,d2_c)=reshape(midpoint,[N_a,N_semiz,N_e]);

        % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
        isInfLower    = (ReturnMatrix_d2ii(1,      :, :, :) == -Inf);
        isInfUpper    = (ReturnMatrix_d2ii(n2long, :, :, :) == -Inf);
        inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
        inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
        L2flag_ford2(:,:,:,d2_c) = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),[N_a,N_semiz,N_e]);
    end

elseif vfoptions.lowmemory==1
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        EV_d2=EV(:,:,d2_c); % [N_a,N_semiz]
        EVinterp_d2=EVinterp(:,:,d2_c); % [n2aprime,N_semiz]

        % n-Monotonicity, first level for every e
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals, e_val, ReturnFnParamsVec,4); % [N_a,level1n,N_semiz]
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2,[N_a,1,N_semiz]); % autoexpands over a
            % First, we want aprime conditional on (1,a,semiz)
            [~,maxindex1_e]=max(entireRHS_ii,[],1); % no d1
            maxindex1(1,:,:,e_c)=maxindex1_e;
        end
        % Second level based on monotonicity
        maxgap=max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3); % max over semiz and e (same as lowmemory=0)

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            maxindex1_e=maxindex1(1,:,:,e_c); % [1,level1n,N_semiz]

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints(1,level1ii,:)=maxindex1_e;

            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1_e(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(curraindex), semiz_gridvals, e_val, ReturnFnParamsVec,5); % [maxgap(ii)+1,length(curraindex),N_semiz]
                    aprimez=aprimeindexes+N_a*semizind; % index into EV_d2 [N_a,N_semiz]
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_semiz]); % singleton dim 2 broadcasts over curraindex
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints(1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1_e(1,ii,:);
                    midpoints(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
                end
            end

            % Now for the interpolation layer
            % Turn midpoints into the 'midpoint'
            midpoint=max(min(midpoints,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-n_semiz
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a-by-n_semiz
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, reshape(aprime_grid(aprimeindexes),size(aprimeindexes)), a_grid, semiz_gridvals, e_val, ReturnFnParamsVec,5); % [n2long,N_a,N_semiz]
            aprimez=aprimeindexes+n2aprime*semizind; % index into EVinterp_d2 [n2aprime,N_semiz]
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_semiz]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2(:,:,e_c,d2_c)=reshape(Vtemp,[N_a,N_semiz]);
            Policy_ford2(:,:,e_c,d2_c)=reshape(maxindex,[N_a,N_semiz]); % aprimeL2ind
            midpoint_ford2(:,:,e_c,d2_c)=reshape(midpoint,[N_a,N_semiz]);

            % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
            isInfLower    = (ReturnMatrix_d2ii(1,      :, :) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(n2long, :, :) == -Inf);
            inLowerStrict = (maxindex >= 2)         & (maxindex <= n2short+1);
            inUpperStrict = (maxindex >= n2short+3) & (maxindex <= n2long-1);
            L2flag_ford2(:,:,e_c,d2_c) = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),[N_a,N_semiz]);
        end
    end
end

%% Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
[V,maxindex]=max(V_ford2,[],4); % max over d2, [N_a,N_semiz,N_e]
Policy=zeros(4,N_a,N_semiz,N_e,'gpuArray'); % [d2; aprime lower grid point; L2 index; L2flag]
Policy(1,:,:,:)=reshape(maxindex,[1,N_a,N_semiz,N_e]); % d2 is just maxindex
maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d2 that corresponds, make it this shape for addition just below
Policy(2,:,:,:)=reshape(midpoint_ford2((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]); % midpoint
Policy(3,:,:,:)=reshape(Policy_ford2((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]); % aprimeL2ind
Policy(4,:,:,:)=reshape(L2flag_ford2((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]); % L2flag

%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

end
