function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_GI1_raw(Vnext,n_d1,n_d2,n_a,n_z,n_semiz, d1_gridvals, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step with grid interpolation layer, with semi-exogenous state semiz, with d1 and d2 (d2 determines the transition of semiz), with z.
% Vnext is [N_a,N_bothz] (next period value fn), bothz=(semiz,z) with semiz first
% pi_z is [N_z,N_zprime]; pi_semiz is [N_semiz,N_semizprime,N_d2] (this period's transition)
% Policy is returned as kron rows [d1; d2; aprime lower grid point; L2 index (1 to ngridinterp+2); L2flag], [5,N_a,N_bothz]
% lowmemory: 0 is everything at once; 1 loops over (semiz,z).

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%%
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

special_n_d=[n_d1,ones(1,length(n_d2))];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]); % version to use when looping over d2

if vfoptions.lowmemory==1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

aind=gpuArray(0:1:N_a-1); % already includes -1
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1, [1,1,N_bothz]
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-2); % already includes -1, [1,1,1,N_bothz]

bothz_gridvals=[repmat(semiz_gridvals,N_z,1),repelem(z_gridvals,N_semiz,1)];

% Preallocate
V_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
Policy_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray'); % (d1,L2) index on the second layer
midpoint_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2flag_ford2=2*ones(N_a,N_bothz,N_d2,'gpuArray'); % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
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
% First z', then semiz' (given d2)
EV=reshape(Vnext,[N_a,N_semiz,N_z]).*shiftdim(pi_z',-2); % [N_a,N_semizprime,N_zprime,N_z]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,3),[N_a,N_semiz,1,N_z]); % sum over z', aprime-by-semizprime-by-1-by-z
EV=EV.*permute(pi_semiz,[4,2,1,5,3]); % [N_a,N_semizprime,N_semiz,N_z,N_d2]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,2),[N_a,N_bothz,N_d2]); % sum over semiz', aprime-by-(semiz,z)-by-d2

% Interpolate EV over aprime_grid
EVinterp=reshape(interp1(a_grid,reshape(EV,[N_a,numel(EV)/N_a]),aprime_grid),[n2aprime,N_bothz,N_d2]); % aprime(fine)-by-(semiz,z)-by-d2

%%
if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        EVinterp_d2=EVinterp(:,:,d2_c); % [n2aprime,N_bothz]

        ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, n_bothz, d12c_gridvals, a_grid, bothz_gridvals, ReturnFnParamsVec,1); % [N_d1,N_a,N_a,N_bothz]
        entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*reshape(EV(:,:,d2_c),[1,N_a,1,N_bothz]);
        % Treat standard problem as just being the first layer
        [~,maxindex]=max(entireRHS,[],2);

        % Turn maxindex into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d1-by-1-by-n_a-by-n_bothz
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d1-by-n2long-by-n_a-by-n_bothz
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, reshape(aprime_grid(aprimeindexes),size(aprimeindexes)), a_grid, bothz_gridvals, ReturnFnParamsVec,2); % [N_d1*n2long,N_a,N_bothz]
        aprimez=aprimeindexes+n2aprime*bothzBind; % index into EVinterp_d2 [n2aprime,N_bothz]
        entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_bothz]);
        [Vtemp,maxindex]=max(entireRHS_ii,[],1);

        V_ford2(:,:,d2_c)=reshape(Vtemp,[N_a,N_bothz]);
        Policy_ford2(:,:,d2_c)=reshape(maxindex,[N_a,N_bothz]);

        d1_ind=rem(maxindex-1,N_d1)+1;
        allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind; % midpoint is n_d1-by-1-by-n_a-by-n_bothz
        midpoint_ford2(:,:,d2_c)=reshape(midpoint(allind),[N_a,N_bothz]);

        % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
        L2offset      = ceil(maxindex/N_d1);
        linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind; % (d1,L2=1,a,bothz) of ReturnMatrix_d2ii
        linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind; % (d1,L2=n2long,a,bothz)
        isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        L2flag_ford2(:,:,d2_c) = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),[N_a,N_bothz]);
    end

elseif vfoptions.lowmemory==1
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);

        for z_c=1:N_bothz
            z_val=bothz_gridvals(z_c,:);
            EVinterp_d2z=EVinterp(:,z_c,d2_c); % [n2aprime,1]

            ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, special_n_bothz, d12c_gridvals, a_grid, z_val, ReturnFnParamsVec,1); % [N_d1,N_a,N_a]
            entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*reshape(EV(:,z_c,d2_c),[1,N_a]);
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS_z,[],2);

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d1-by-1-by-n_a
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d1-by-n2long-by-n_a
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, reshape(aprime_grid(aprimeindexes),size(aprimeindexes)), a_grid, z_val, ReturnFnParamsVec,2); % [N_d1*n2long,N_a]
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2z(aprimeindexes),[N_d1*n2long,N_a]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2(:,z_c,d2_c)=reshape(Vtemp,[N_a,1]);
            Policy_ford2(:,z_c,d2_c)=reshape(maxindex,[N_a,1]);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind; % midpoint is n_d1-by-1-by-n_a
            midpoint_ford2(:,z_c,d2_c)=reshape(midpoint(allind),[N_a,1]);

            % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
            L2offset      = ceil(maxindex/N_d1);
            linidx_lower  = d1_ind                   + N_d1*n2long*aind; % (d1,L2=1,a) of ReturnMatrix_d2ii
            linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind; % (d1,L2=n2long,a)
            isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            L2flag_ford2(:,z_c,d2_c) = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),[N_a,1]);
        end
    end
end

%% Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
[V,maxindex]=max(V_ford2,[],3); % max over d2, [N_a,N_bothz]
Policy=zeros(5,N_a,N_bothz,'gpuArray'); % [d1; d2; aprime lower grid point; L2 index; L2flag]
Policy(2,:,:)=reshape(maxindex,[1,N_a,N_bothz]); % d2 is just maxindex
maxindex=reshape(maxindex,[N_a*N_bothz,1]); % This is the value of d2 that corresponds, make it this shape for addition just below
d1aprimeL2_ind=reshape(Policy_ford2((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
Policy(1,:,:)=rem(d1aprimeL2_ind-1,N_d1)+1; % d1
Policy(3,:,:)=reshape(midpoint_ford2((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]); % midpoint
Policy(4,:,:)=ceil(d1aprimeL2_ind/N_d1); % aprimeL2ind
Policy(5,:,:)=reshape(L2flag_ford2((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]); % L2flag

%% Currently Policy(3,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(3,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(3,:,:)=Policy(3,:,:)-adjust; % lower grid point
Policy(4,:,:)=adjust.*Policy(4,:,:)+(1-adjust).*(Policy(4,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

end
