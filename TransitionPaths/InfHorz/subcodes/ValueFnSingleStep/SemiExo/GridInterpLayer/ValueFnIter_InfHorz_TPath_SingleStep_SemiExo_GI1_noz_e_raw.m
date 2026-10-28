function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_GI1_noz_e_raw(Vnext,n_d1,n_d2,n_a,n_semiz,n_e, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals, e_gridvals, pi_semiz, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step with grid interpolation layer, with semi-exogenous state semiz, with d1 and d2 (d2 determines the transition of semiz), no z, with iid e.
% Vnext is [N_a,N_semiz,N_e] (next period value fn)
% pi_semiz is [N_semiz,N_semizprime,N_d2] (this period's transition); pi_e is [N_e,1], the distribution of next period e
% Policy is returned as kron rows [d1; d2; aprime lower grid point; L2 index (1 to ngridinterp+2); L2flag], [5,N_a,N_semiz,N_e]
% lowmemory: 0 is everything at once; 1 loops over e.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%%
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

special_n_d=[n_d1,ones(1,length(n_d2))];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]); % version to use when looping over d2

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
end

aind=gpuArray(0:1:N_a-1); % already includes -1
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1, [1,1,N_semiz]
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-2); % already includes -1, [1,1,1,N_semiz]
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1, [1,1,1,N_e]

% Preallocate
V_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray'); % (d1,L2) index on the second layer
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
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        EVinterp_d2=EVinterp(:,:,d2_c); % [n2aprime,N_semiz]

        ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_semiz, n_e, d12c_gridvals, a_grid, semiz_gridvals, e_gridvals, ReturnFnParamsVec,1); % [N_d1,N_a,N_a,N_semiz,N_e]
        entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*reshape(EV(:,:,d2_c),[1,N_a,1,N_semiz]); % autoexpands over d1, a and e
        % Treat standard problem as just being the first layer
        [~,maxindex]=max(entireRHS,[],2);

        % Turn maxindex into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d1-by-1-by-n_a-by-n_semiz-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d1-by-n2long-by-n_a-by-n_semiz-by-n_e
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, reshape(aprime_grid(aprimeindexes),size(aprimeindexes)), a_grid, semiz_gridvals, e_gridvals, ReturnFnParamsVec,2); % [N_d1*n2long,N_a,N_semiz,N_e]
        aprimez=aprimeindexes+n2aprime*semizBind; % index into EVinterp_d2 [n2aprime,N_semiz]; EV does not depend on e, so no e offset
        entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_semiz,N_e]);
        [Vtemp,maxindex]=max(entireRHS_ii,[],1);

        V_ford2(:,:,:,d2_c)=reshape(Vtemp,[N_a,N_semiz,N_e]);
        Policy_ford2(:,:,:,d2_c)=reshape(maxindex,[N_a,N_semiz,N_e]);

        d1_ind=rem(maxindex-1,N_d1)+1;
        allind=d1_ind+N_d1*aind+N_d1*N_a*semizind+N_d1*N_a*N_semiz*eind; % midpoint is n_d1-by-1-by-n_a-by-n_semiz-by-n_e
        midpoint_ford2(:,:,:,d2_c)=reshape(midpoint(allind),[N_a,N_semiz,N_e]);

        % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
        L2offset      = ceil(maxindex/N_d1);
        linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*semizind + N_d1*n2long*N_a*N_semiz*eind; % (d1,L2=1,a,semiz,e) of ReturnMatrix_d2ii
        linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind + N_d1*n2long*N_a*N_semiz*eind; % (d1,L2=n2long,a,semiz,e)
        isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        L2flag_ford2(:,:,:,d2_c) = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),[N_a,N_semiz,N_e]);
    end

elseif vfoptions.lowmemory==1
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        EVinterp_d2=EVinterp(:,:,d2_c); % [n2aprime,N_semiz]

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);

            ReturnMatrix_d2e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_semiz, special_n_e, d12c_gridvals, a_grid, semiz_gridvals, e_val, ReturnFnParamsVec,1); % [N_d1,N_a,N_a,N_semiz]
            entireRHS_e=ReturnMatrix_d2e+DiscountFactorParamsVec*reshape(EV(:,:,d2_c),[1,N_a,1,N_semiz]); % autoexpands over d1 and a
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d1-by-1-by-n_a-by-n_semiz
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d1-by-n2long-by-n_a-by-n_semiz
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_semiz, special_n_e, d12c_gridvals, reshape(aprime_grid(aprimeindexes),size(aprimeindexes)), a_grid, semiz_gridvals, e_val, ReturnFnParamsVec,2); % [N_d1*n2long,N_a,N_semiz]
            aprimez=aprimeindexes+n2aprime*semizBind; % index into EVinterp_d2 [n2aprime,N_semiz]
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_semiz]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2(:,:,e_c,d2_c)=reshape(Vtemp,[N_a,N_semiz]);
            Policy_ford2(:,:,e_c,d2_c)=reshape(maxindex,[N_a,N_semiz]);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*semizind; % midpoint is n_d1-by-1-by-n_a-by-n_semiz
            midpoint_ford2(:,:,e_c,d2_c)=reshape(midpoint(allind),[N_a,N_semiz]);

            % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
            L2offset      = ceil(maxindex/N_d1);
            linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*semizind; % (d1,L2=1,a,semiz) of ReturnMatrix_d2ii
            linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind; % (d1,L2=n2long,a,semiz)
            isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            L2flag_ford2(:,:,e_c,d2_c) = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),[N_a,N_semiz]);
        end
    end
end

%% Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
[V,maxindex]=max(V_ford2,[],4); % max over d2, [N_a,N_semiz,N_e]
Policy=zeros(5,N_a,N_semiz,N_e,'gpuArray'); % [d1; d2; aprime lower grid point; L2 index; L2flag]
Policy(2,:,:,:)=reshape(maxindex,[1,N_a,N_semiz,N_e]); % d2 is just maxindex
maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d2 that corresponds, make it this shape for addition just below
d1aprimeL2_ind=reshape(Policy_ford2((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
Policy(1,:,:,:)=rem(d1aprimeL2_ind-1,N_d1)+1; % d1
Policy(3,:,:,:)=reshape(midpoint_ford2((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]); % midpoint
Policy(4,:,:,:)=ceil(d1aprimeL2_ind/N_d1); % aprimeL2ind
Policy(5,:,:,:)=reshape(L2flag_ford2((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]); % L2flag

%% Currently Policy(3,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(3,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust; % lower grid point
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

end
