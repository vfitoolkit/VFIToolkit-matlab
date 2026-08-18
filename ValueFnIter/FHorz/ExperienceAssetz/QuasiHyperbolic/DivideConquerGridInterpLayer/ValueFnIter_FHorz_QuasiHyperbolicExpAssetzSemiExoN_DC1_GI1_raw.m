function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoN_DC1_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset (z-dependent aprimeFn)
% z is exogenous markov state (required), semiz is semi-exog state
% aprimeFn = aprimeFn(d2, a2, z, ...)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest
% DC + GI splice (no L2flag scaffold): DC level1n n-Monotonicity coarse search,
% then GI L2 fine grid search around midpoint.

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
d2ind=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]; maps full d12-index to d2-component
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;

Valt=zeros(N_a,N_bothz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_j,'gpuArray');
% Policy storage with d1, d2, d3, a1prime_midpoint, a1primeL2ind
Policyalt=zeros(5,N_a,N_bothz,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_j,'gpuArray');
PolicyL2flagalt=2*ones(1,N_a,N_bothz,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_bothz,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
if vfoptions.lowmemory==0
    midpoint_alt=zeros(N_d12,1,N_a1,N_a2,N_bothz,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a1,N_a2,N_bothz,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint_alt=zeros(N_d12,1,N_a1,N_a2,N_semiz,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a1,N_a2,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint_alt=zeros(N_d12,1,N_a1,N_a2,'gpuArray');
    midpoint_tilde=zeros(N_d12,1,N_a1,N_a2,'gpuArray');
end

% Preallocate per-d3 storage
V_ford3_alt=zeros(N_a,N_bothz,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy4_ford3_alt=zeros(4,N_a,N_bothz,N_d3,'gpuArray');
Policy4_ford3_tilde=zeros(4,N_a,N_bothz,N_d3,'gpuArray');
flag_ford3_alt=2*ones(1,N_a,N_bothz,N_d3,'gpuArray');
flag_ford3_tilde=2*ones(1,N_a,N_bothz,N_d3,'gpuArray'); % L2 flag per d3, aggregated after d3 max

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
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-3); % already includes -1
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1

% Offset for linear indexing into [N_a, N_bothz] (semiz fastest within bothz)
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Period N_j could be done without looping over d3, but then it needs much more memory than the rest, and since looping for the other periods the runtime cost of looping here is negligible.
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Just keep the 'midpoint_alt' version of maxindex1 [as GI]
            midpoint_alt(:,1,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d12-by-1-by-1-by-n_a2-by-n_bothz
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d12-by-maxgap(ii)+1-by-1-by-n_a2-by-n_bothz
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=3, Refine=0
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoint_alt(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint_alt(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Turn this into the 'midpoint_alt'
            midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint_alt is n_d12-1-by-n_a1-by-n_a2-by-n_bothz
            a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint_alt
            % aprime possibilities are n_d12-by-n2long-by-n_a1-by-n_a2-by-n_bothz
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d12,N_a1prime,N_a1,N_a2,N_bothz]; Level=2, Refine=0
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind; % midpoint_alt is n_d12-by-1-by-n_a1-by-n_a2-by-n_bothz
            Policy4_ford3_alt(1,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_alt(2,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_alt(3,:,:,d3_c)=shiftdim(squeeze(midpoint_alt(allind)),-1); % a1prime midpoint_alt
            Policy4_ford3_alt(4,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1); % a1primeL2ind
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d12);
            linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
            linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3_alt(1,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end

    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % [1,1,N_semiz]
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

                % n-Monotonicity
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);

                % Just keep the 'midpoint_alt' version of maxindex1 [as GI]
                midpoint_alt(:,1,level1ii,:,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d12-by-1-by-1-by-n_a2-by-n_semiz
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d12-by-maxgap(ii)+1-by-1-by-n_a2-by-n_semiz
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        [~,maxindex]=max(ReturnMatrix_ii_z,[],2);
                        midpoint_alt(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint_alt(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                % Turn this into the 'midpoint_alt'
                midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint_alt is n_d12-1-by-n_a1-by-n_a2-by-n_semiz
                a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint_alt
                % aprime possibilities are n_d12-by-n2long-by-n_a1-by-n_a2-by-n_semiz
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0); % [N_d12,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
                V_ford3_alt(:,semizblock,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*semizBind; % midpoint_alt is n_d12-by-1-by-n_a1-by-n_a2-by-n_semiz
                Policy4_ford3_alt(1,:,semizblock,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_alt(2,:,semizblock,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_alt(3,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_alt(allind)),-1); % a1prime midpoint_alt
                Policy4_ford3_alt(4,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                isInfLower = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_alt(1,:,semizblock,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end

    elseif vfoptions.lowmemory==2 % joint loop over bothz

        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);

                % Just keep the 'midpoint_alt' version of maxindex1 [as GI]
                midpoint_alt(:,1,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d12-by-1-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d12-by-maxgap(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        [~,maxindex]=max(ReturnMatrix_ii_z,[],2);
                        midpoint_alt(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoint_alt(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                % Turn this into the 'midpoint_alt'
                midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint_alt is n_d12-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint_alt
                % aprime possibilities are n_d12-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d12,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind; % midpoint_alt is n_d12-by-1-by-n_a1-by-n_a2
                Policy4_ford3_alt(1,:,z_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_alt(2,:,z_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_alt(3,:,z_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind)),-1); % a1prime midpoint_alt
                Policy4_ford3_alt(4,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d12);
                linidx_lower = d_ind                   + N_d12*n2long*aind;
                linidx_upper = d_ind + N_d12*(n2long-1) + N_d12*n2long*aind;
                isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3_alt(1,:,z_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,N_j)=V_jj;
    Policyalt(3,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=4*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policyalt(1,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz]);
    Policyalt(2,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz]);
    Policyalt(4,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz]);
    Policyalt(5,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz]);
    flat_idx=(1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1);
    PolicyL2flagalt(1,:,:,N_j)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz]);
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy(:,:,:,N_j)=Policyalt(:,:,:,N_j);
    PolicyL2flag(:,:,:,N_j)=PolicyL2flagalt(:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_z], whereas aprimeProbs is [N_d2,N_a2,N_z]

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2,N_z]
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_z]
    aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1); % [N_d2*N_a1,N_a2,N_z]
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz); % [N_d2*N_a1, N_a2, N_bothz] (semiz fastest within bothz)
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]); % switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            EV_2D=reshape(EV,[N_a,N_bothz]);

            % Linear-indexing lookup
            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            % Interpolate EV over aprime_grid
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_bothz]
            % d1-dim is implicit singleton in DiscountedEV_alt/DiscountedEVinterp_alt, broadcasts at use sites

            % n-Monotonicity
            DiscountedEV_tilde=beta0beta*EVbase_qh;
            % Interpolate EV over aprime_grid
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_bothz]
            % d1-dim is implicit singleton in DiscountedEV_tilde/DiscountedEVinterp_tilde, broadcasts at use sites

            % n-Monotonicity

            % --- alt pass ---
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);

            % Just keep the 'midpoint_alt' version of maxindex1_alt [as GI]
            midpoint_alt(:,1,level1ii,:,:)=maxindex1_alt;

            % Attempt for improved version
            maxgap_alt=squeeze(max(max(max(maxindex1_alt(:,1,2:end,:,:)-maxindex1_alt(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap_alt(ii)>0
                    loweredge_alt=min(maxindex1_alt(:,1,ii,:,:),N_a1-maxgap_alt(ii)); % maxindex1_alt(ii,:), but avoid going off top of grid when we add maxgap_alt(ii) points
                    % loweredge_alt is n_d12-by-1-by-1-by-n_a2-by-n_bothz
                    a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                    % aprime possibilities are n_d12-by-maxgap_alt(ii)+1-by-1-by-n_a2-by-n_bothz
                    ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=3, Refine=0
                    d2aprimez_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind; % [N_d12,maxgap_alt+1,1,N_a2,N_bothz]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2,N_bothz]
                    entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt);
                    [~,maxindex_alt]=max(entireRHS_ii_d3_alt,[],2);
                    midpoint_alt(:,1,curraindex_alt,:,:)=maxindex_alt+(loweredge_alt-1);
                else
                    loweredge_alt=maxindex1_alt(:,1,ii,:,:);
                    midpoint_alt(:,1,curraindex_alt,:,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                end
            end

            % Turn this into the 'midpoint_alt'
            midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint_alt is n_d12-1-by-n_a1-by-n_a2-by-n_bothz
            a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint_alt
            % aprime possibilities are n_d12-by-n2long-by-n_a1-by-n_a2-by-n_bothz
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d12,N_a1prime,N_a1,N_a2,N_bothz]; Level=2, Refine=0
            d2a1primea2bothz_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind; % Note: EV does not depend on d1, but this still has d1 as part of the first dimension
            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii_alt,1);
            d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
            allind_alt=d_ind_alt+N_d12*aind+N_d12*N_a*bothzBind; % midpoint_alt is n_d12-by-1-by-n_a1-by-n_a2-by-n_bothz
            Policy4_ford3_alt(1,:,:,d3_c)=rem(d_ind_alt-1,N_d1)+1; % d1
            Policy4_ford3_alt(2,:,:,d3_c)=ceil(d_ind_alt/N_d1); % d2
            Policy4_ford3_alt(3,:,:,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1); % a1prime midpoint_alt
            Policy4_ford3_alt(4,:,:,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1); % a1primeL2ind
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset_alt = ceil(maxindexL2_alt/N_d12);
            linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
            linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
            isInfLower_alt = (ReturnMatrix_ii_d3_alt(linidx_lower_alt) == -Inf);
            isInfUpper_alt = (ReturnMatrix_ii_d3_alt(linidx_upper_alt) == -Inf);
            inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
            inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
            flag_ford3_alt(1,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

            % --- tilde pass ---
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);

            % Just keep the 'midpoint_tilde' version of maxindex1_tilde [as GI]
            midpoint_tilde(:,1,level1ii,:,:)=maxindex1_tilde;

            % Attempt for improved version
            maxgap_tilde=squeeze(max(max(max(maxindex1_tilde(:,1,2:end,:,:)-maxindex1_tilde(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap_tilde(ii)>0
                    loweredge_tilde=min(maxindex1_tilde(:,1,ii,:,:),N_a1-maxgap_tilde(ii)); % maxindex1_tilde(ii,:), but avoid going off top of grid when we add maxgap_tilde(ii) points
                    % loweredge_tilde is n_d12-by-1-by-1-by-n_a2-by-n_bothz
                    a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                    % aprime possibilities are n_d12-by-maxgap_tilde(ii)+1-by-1-by-n_a2-by-n_bothz
                    ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=3, Refine=0
                    d2aprimez_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind; % [N_d12,maxgap_tilde+1,1,N_a2,N_bothz]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2,N_bothz]
                    entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                    [~,maxindex_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                    midpoint_tilde(:,1,curraindex_tilde,:,:)=maxindex_tilde+(loweredge_tilde-1);
                else
                    loweredge_tilde=maxindex1_tilde(:,1,ii,:,:);
                    midpoint_tilde(:,1,curraindex_tilde,:,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                end
            end

            % Turn this into the 'midpoint_tilde'
            midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint_tilde is n_d12-1-by-n_a1-by-n_a2-by-n_bothz
            a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint_tilde
            % aprime possibilities are n_d12-by-n2long-by-n_a1-by-n_a2-by-n_bothz
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d12,N_a1prime,N_a1,N_a2,N_bothz]; Level=2, Refine=0
            d2a1primea2bothz_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind; % Note: EV does not depend on d1, but this still has d1 as part of the first dimension
            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii_tilde,1);
            d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
            allind_tilde=d_ind_tilde+N_d12*aind+N_d12*N_a*bothzBind; % midpoint_tilde is n_d12-by-1-by-n_a1-by-n_a2-by-n_bothz
            Policy4_ford3_tilde(1,:,:,d3_c)=rem(d_ind_tilde-1,N_d1)+1; % d1
            Policy4_ford3_tilde(2,:,:,d3_c)=ceil(d_ind_tilde/N_d1); % d2
            Policy4_ford3_tilde(3,:,:,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1); % a1prime midpoint_tilde
            Policy4_ford3_tilde(4,:,:,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1); % a1primeL2ind
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
            linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
            linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
            isInfLower_tilde = (ReturnMatrix_ii_d3_tilde(linidx_lower_tilde) == -Inf);
            isInfUpper_tilde = (ReturnMatrix_ii_d3_tilde(linidx_upper_tilde) == -Inf);
            inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
            inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
            flag_ford3_tilde(1,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
        end

    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3); % [1,1,1,1,N_semiz]
        semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % [1,1,N_semiz]
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

                EV=V_Jplus1.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz_next, N_semiz]
                EV(isnan(EV))=0;
                EV=sum(EV,2); % [N_a, 1, N_semiz]
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_full(:,:,semizblock);
                aprimeProbs_z(skipinterp)=0;
                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                EVbase_qh=reshape(entireEV_z,[N_d2,N_a1,1,N_a2,N_semiz]);
                DiscountedEV_alt=beta*EVbase_qh;
                DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]

                % n-Monotonicity
                DiscountedEV_tilde=beta0beta*EVbase_qh;
                DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]); % [N_d2,N_a1prime,1,N_a2,N_semiz]

                % n-Monotonicity

                % --- alt pass ---
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1_alt]=max(entireRHS_ii_d3z_alt,[],2);

                % Just keep the 'midpoint_alt' version of maxindex1_alt [as GI]
                midpoint_alt(:,1,level1ii,:,:)=maxindex1_alt;

                % Attempt for improved version
                maxgap_alt=squeeze(max(max(max(maxindex1_alt(:,1,2:end,:,:)-maxindex1_alt(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap_alt(ii)>0
                        loweredge_alt=min(maxindex1_alt(:,1,ii,:,:),N_a1-maxgap_alt(ii));
                        % loweredge_alt is n_d12-by-1-by-1-by-n_a2-by-n_semiz
                        a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                        ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        d2aprimez_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind; % linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2,N_semiz]
                        entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+DiscountedEV_alt(d2aprimez_alt);
                        [~,maxindex_alt]=max(entireRHS_ii_d3z_alt,[],2);
                        midpoint_alt(:,1,curraindex_alt,:,:)=maxindex_alt+(loweredge_alt-1);
                    else
                        loweredge_alt=maxindex1_alt(:,1,ii,:,:);
                        midpoint_alt(:,1,curraindex_alt,:,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                    end
                end

                % Turn this into the 'midpoint_alt'
                midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint_alt is n_d12-1-by-n_a1-by-n_a2-by-n_semiz
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint_alt
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0); % [N_d12,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
                d2a1primea2bothz_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3z_alt,[],1);
                V_ford3_alt(:,semizblock,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
                allind_alt=d_ind_alt+N_d12*aind+N_d12*N_a*semizBind; % midpoint_alt is n_d12-by-1-by-n_a1-by-n_a2-by-n_semiz
                Policy4_ford3_alt(1,:,semizblock,d3_c)=rem(d_ind_alt-1,N_d1)+1; % d1
                Policy4_ford3_alt(2,:,semizblock,d3_c)=ceil(d_ind_alt/N_d1); % d2
                Policy4_ford3_alt(3,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1); % a1prime midpoint_alt
                Policy4_ford3_alt(4,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_alt = ceil(maxindexL2_alt/N_d12);
                linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                isInfLower_alt = (ReturnMatrix_ii_d3z_alt(linidx_lower_alt) == -Inf);
                isInfUpper_alt = (ReturnMatrix_ii_d3z_alt(linidx_upper_alt) == -Inf);
                inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
                inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
                flag_ford3_alt(1,:,semizblock,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

                % --- tilde pass ---
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1_tilde]=max(entireRHS_ii_d3z_tilde,[],2);

                % Just keep the 'midpoint_tilde' version of maxindex1_tilde [as GI]
                midpoint_tilde(:,1,level1ii,:,:)=maxindex1_tilde;

                % Attempt for improved version
                maxgap_tilde=squeeze(max(max(max(maxindex1_tilde(:,1,2:end,:,:)-maxindex1_tilde(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap_tilde(ii)>0
                        loweredge_tilde=min(maxindex1_tilde(:,1,ii,:,:),N_a1-maxgap_tilde(ii));
                        % loweredge_tilde is n_d12-by-1-by-1-by-n_a2-by-n_semiz
                        a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                        ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        d2aprimez_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind; % linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2,N_semiz]
                        entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                        [~,maxindex_tilde]=max(entireRHS_ii_d3z_tilde,[],2);
                        midpoint_tilde(:,1,curraindex_tilde,:,:)=maxindex_tilde+(loweredge_tilde-1);
                    else
                        loweredge_tilde=maxindex1_tilde(:,1,ii,:,:);
                        midpoint_tilde(:,1,curraindex_tilde,:,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                    end
                end

                % Turn this into the 'midpoint_tilde'
                midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint_tilde is n_d12-1-by-n_a1-by-n_a2-by-n_semiz
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint_tilde
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0); % [N_d12,N_a1prime,N_a1,N_a2,N_semiz]; Level=2, Refine=0
                d2a1primea2bothz_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3z_tilde,[],1);
                V_ford3_tilde(:,semizblock,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
                allind_tilde=d_ind_tilde+N_d12*aind+N_d12*N_a*semizBind; % midpoint_tilde is n_d12-by-1-by-n_a1-by-n_a2-by-n_semiz
                Policy4_ford3_tilde(1,:,semizblock,d3_c)=rem(d_ind_tilde-1,N_d1)+1; % d1
                Policy4_ford3_tilde(2,:,semizblock,d3_c)=ceil(d_ind_tilde/N_d1); % d2
                Policy4_ford3_tilde(3,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1); % a1prime midpoint_tilde
                Policy4_ford3_tilde(4,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
                linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                isInfLower_tilde = (ReturnMatrix_ii_d3z_tilde(linidx_lower_tilde) == -Inf);
                isInfUpper_tilde = (ReturnMatrix_ii_d3z_tilde(linidx_upper_tilde) == -Inf);
                inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
                inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
                flag_ford3_tilde(1,:,semizblock,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
            end
        end

    elseif vfoptions.lowmemory==2 % joint loop over bothz

        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            % d1-dim is implicit singleton; broadcasts at use sites

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            % d1-dim is implicit singleton; broadcasts at use sites

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

            % --- alt pass ---
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,z_c);
                DiscountedEVinterp_z_alt=DiscountedEVinterp_alt(:,:,:,:,z_c);

                % n-Monotonicity
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+repelem(DiscountedEV_z_alt,N_d1,1,1,1);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1_alt]=max(entireRHS_ii_d3z_alt,[],2);

                % Just keep the 'midpoint_alt' version of maxindex1_alt [as GI]
                midpoint_alt(:,1,level1ii,:)=maxindex1_alt;

                % Attempt for improved version
                maxgap_alt=squeeze(max(max(maxindex1_alt(:,1,2:end,:)-maxindex1_alt(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap_alt(ii)>0
                        loweredge_alt=min(maxindex1_alt(:,1,ii,:),N_a1-maxgap_alt(ii)); % maxindex1_alt(ii,:), but avoid going off top of grid when we add maxgap_alt(ii) points
                        a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                        ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        d2aprime_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind; % [N_d12,maxgap_alt+1,1,N_a2]; linear index into DiscountedEV_z_alt [N_d2,N_a1,1,N_a2]
                        entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+DiscountedEV_z_alt(d2aprime_alt);
                        [~,maxindex_alt]=max(entireRHS_ii_d3z_alt,[],2);
                        midpoint_alt(:,1,curraindex_alt,:)=maxindex_alt+(loweredge_alt-1);
                    else
                        loweredge_alt=maxindex1_alt(:,1,ii,:);
                        midpoint_alt(:,1,curraindex_alt,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                    end
                end

                % Turn this into the 'midpoint_alt'
                midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint_alt is n_d12-1-by-n_a1-by-n_a2
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint_alt
                % aprime possibilities are n_d12-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d12,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
                d2a1primea2_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+reshape(DiscountedEVinterp_z_alt(d2a1primea2_alt(:)),[N_d12*n2long,N_a1*N_a2]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3z_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
                allind_alt=d_ind_alt+N_d12*aind; % midpoint_alt is n_d12-by-1-by-n_a1-by-n_a2
                Policy4_ford3_alt(1,:,z_c,d3_c)=rem(d_ind_alt-1,N_d1)+1; % d1
                Policy4_ford3_alt(2,:,z_c,d3_c)=ceil(d_ind_alt/N_d1); % d2
                Policy4_ford3_alt(3,:,z_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1); % a1prime midpoint_alt
                Policy4_ford3_alt(4,:,z_c,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_alt = ceil(maxindexL2_alt/N_d12);
                linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind;
                linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind;
                isInfLower_alt = (ReturnMatrix_ii_d3z_alt(linidx_lower_alt) == -Inf);
                isInfUpper_alt = (ReturnMatrix_ii_d3z_alt(linidx_upper_alt) == -Inf);
                inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
                inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
                flag_ford3_alt(1,:,z_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

            % --- tilde pass ---
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,z_c);
                DiscountedEVinterp_z_tilde=DiscountedEVinterp_tilde(:,:,:,:,z_c);

                % n-Monotonicity
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+repelem(DiscountedEV_z_tilde,N_d1,1,1,1);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1_tilde]=max(entireRHS_ii_d3z_tilde,[],2);

                % Just keep the 'midpoint_tilde' version of maxindex1_tilde [as GI]
                midpoint_tilde(:,1,level1ii,:)=maxindex1_tilde;

                % Attempt for improved version
                maxgap_tilde=squeeze(max(max(maxindex1_tilde(:,1,2:end,:)-maxindex1_tilde(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap_tilde(ii)>0
                        loweredge_tilde=min(maxindex1_tilde(:,1,ii,:),N_a1-maxgap_tilde(ii)); % maxindex1_tilde(ii,:), but avoid going off top of grid when we add maxgap_tilde(ii) points
                        a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                        ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=3, Refine=0
                        d2aprime_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind; % [N_d12,maxgap_tilde+1,1,N_a2]; linear index into DiscountedEV_z_tilde [N_d2,N_a1,1,N_a2]
                        entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+DiscountedEV_z_tilde(d2aprime_tilde);
                        [~,maxindex_tilde]=max(entireRHS_ii_d3z_tilde,[],2);
                        midpoint_tilde(:,1,curraindex_tilde,:)=maxindex_tilde+(loweredge_tilde-1);
                    else
                        loweredge_tilde=maxindex1_tilde(:,1,ii,:);
                        midpoint_tilde(:,1,curraindex_tilde,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                    end
                end

                % Turn this into the 'midpoint_tilde'
                midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint_tilde is n_d12-1-by-n_a1-by-n_a2
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint_tilde
                % aprime possibilities are n_d12-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0); % [N_d12,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
                d2a1primea2_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+reshape(DiscountedEVinterp_z_tilde(d2a1primea2_tilde(:)),[N_d12*n2long,N_a1*N_a2]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3z_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
                allind_tilde=d_ind_tilde+N_d12*aind; % midpoint_tilde is n_d12-by-1-by-n_a1-by-n_a2
                Policy4_ford3_tilde(1,:,z_c,d3_c)=rem(d_ind_tilde-1,N_d1)+1; % d1
                Policy4_ford3_tilde(2,:,z_c,d3_c)=ceil(d_ind_tilde/N_d1); % d2
                Policy4_ford3_tilde(3,:,z_c,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1); % a1prime midpoint_tilde
                Policy4_ford3_tilde(4,:,z_c,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1); % a1primeL2ind
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
                linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind;
                linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind;
                isInfLower_tilde = (ReturnMatrix_ii_d3z_tilde(linidx_lower_tilde) == -Inf);
                isInfUpper_tilde = (ReturnMatrix_ii_d3z_tilde(linidx_upper_tilde) == -Inf);
                inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
                inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
                flag_ford3_tilde(1,:,z_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,N_j)=V_jj;
    Policyalt(3,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=4*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policyalt(1,:,:,N_j)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz]);
    Policyalt(2,:,:,N_j)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz]);
    Policyalt(4,:,:,N_j)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz]);
    Policyalt(5,:,:,N_j)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz]);
    flat_idx=(1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1);
    PolicyL2flagalt(1,:,:,N_j)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz]);

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,N_j)=V_jj;
    Policy(3,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=4*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policy(1,:,:,N_j)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz]);
    Policy(2,:,:,N_j)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz]);
    Policy(5,:,:,N_j)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz]);
    flat_idx=(1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1);
    PolicyL2flag(1,:,:,N_j)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_bothz]);

end

%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);

    EVpre=Valt(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            % d1-dim is implicit singleton; broadcasts at use sites

            % n-Monotonicity
            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            % d1-dim is implicit singleton; broadcasts at use sites

            % n-Monotonicity

            % --- alt pass ---
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

            [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
            midpoint_alt(:,1,level1ii,:,:)=maxindex1_alt;

            maxgap_alt=squeeze(max(max(max(maxindex1_alt(:,1,2:end,:,:)-maxindex1_alt(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_alt(ii)>0
                    loweredge_alt=min(maxindex1_alt(:,1,ii,:,:),N_a1-maxgap_alt(ii));
                    a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                    ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt);
                    [~,maxindex_alt]=max(entireRHS_ii_d3_alt,[],2);
                    midpoint_alt(:,1,curraindex_alt,:,:)=maxindex_alt+(loweredge_alt-1);
                else
                    loweredge_alt=maxindex1_alt(:,1,ii,:,:);
                    midpoint_alt(:,1,curraindex_alt,:,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                end
            end

            midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
            a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothz_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtempii_alt,1);
            d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
            allind_alt=d_ind_alt+N_d12*aind+N_d12*N_a*bothzBind;
            Policy4_ford3_alt(1,:,:,d3_c)=rem(d_ind_alt-1,N_d1)+1;
            Policy4_ford3_alt(2,:,:,d3_c)=ceil(d_ind_alt/N_d1);
            Policy4_ford3_alt(3,:,:,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
            Policy4_ford3_alt(4,:,:,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset_alt = ceil(maxindexL2_alt/N_d12);
            linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
            linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
            isInfLower_alt = (ReturnMatrix_ii_d3_alt(linidx_lower_alt) == -Inf);
            isInfUpper_alt = (ReturnMatrix_ii_d3_alt(linidx_upper_alt) == -Inf);
            inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
            inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
            flag_ford3_alt(1,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

            % --- tilde pass ---
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

            [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
            midpoint_tilde(:,1,level1ii,:,:)=maxindex1_tilde;

            maxgap_tilde=squeeze(max(max(max(maxindex1_tilde(:,1,2:end,:,:)-maxindex1_tilde(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap_tilde(ii)>0
                    loweredge_tilde=min(maxindex1_tilde(:,1,ii,:,:),N_a1-maxgap_tilde(ii));
                    a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                    ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                    entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                    [~,maxindex_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                    midpoint_tilde(:,1,curraindex_tilde,:,:)=maxindex_tilde+(loweredge_tilde-1);
                else
                    loweredge_tilde=maxindex1_tilde(:,1,ii,:,:);
                    midpoint_tilde(:,1,curraindex_tilde,:,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                end
            end

            midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
            a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothz_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtempii_tilde,1);
            d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
            allind_tilde=d_ind_tilde+N_d12*aind+N_d12*N_a*bothzBind;
            Policy4_ford3_tilde(1,:,:,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
            Policy4_ford3_tilde(2,:,:,d3_c)=ceil(d_ind_tilde/N_d1);
            Policy4_ford3_tilde(3,:,:,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
            Policy4_ford3_tilde(4,:,:,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
            linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
            linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*bothzBind;
            isInfLower_tilde = (ReturnMatrix_ii_d3_tilde(linidx_lower_tilde) == -Inf);
            isInfUpper_tilde = (ReturnMatrix_ii_d3_tilde(linidx_upper_tilde) == -Inf);
            inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
            inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
            flag_ford3_tilde(1,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
        end

    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3); % [1,1,1,1,N_semiz]
        semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % [1,1,N_semiz]
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz_next, N_semiz]
                EV(isnan(EV))=0;
                EV=sum(EV,2); % [N_a, 1, N_semiz]
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_full(:,:,semizblock);
                aprimeProbs_z(skipinterp)=0;
                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                EVbase_qh=reshape(entireEV_z,[N_d2,N_a1,1,N_a2,N_semiz]);
                DiscountedEV_alt=beta*EVbase_qh;
                DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                % n-Monotonicity
                DiscountedEV_tilde=beta0beta*EVbase_qh;
                DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);

                % n-Monotonicity

                % --- alt pass ---
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

                [~,maxindex1_alt]=max(entireRHS_ii_d3z_alt,[],2);
                midpoint_alt(:,1,level1ii,:,:)=maxindex1_alt;

                maxgap_alt=squeeze(max(max(max(maxindex1_alt(:,1,2:end,:,:)-maxindex1_alt(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_alt(ii)>0
                        loweredge_alt=min(maxindex1_alt(:,1,ii,:,:),N_a1-maxgap_alt(ii));
                        a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                        ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0);
                        d2aprimez_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                        entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+DiscountedEV_alt(d2aprimez_alt);
                        [~,maxindex_alt]=max(entireRHS_ii_d3z_alt,[],2);
                        midpoint_alt(:,1,curraindex_alt,:,:)=maxindex_alt+(loweredge_alt-1);
                    else
                        loweredge_alt=maxindex1_alt(:,1,ii,:,:);
                        midpoint_alt(:,1,curraindex_alt,:,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                d2a1primea2bothz_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+reshape(DiscountedEVinterp_alt(d2a1primea2bothz_alt(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3z_alt,[],1);
                V_ford3_alt(:,semizblock,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
                allind_alt=d_ind_alt+N_d12*aind+N_d12*N_a*semizBind;
                Policy4_ford3_alt(1,:,semizblock,d3_c)=rem(d_ind_alt-1,N_d1)+1;
                Policy4_ford3_alt(2,:,semizblock,d3_c)=ceil(d_ind_alt/N_d1);
                Policy4_ford3_alt(3,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                Policy4_ford3_alt(4,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_alt = ceil(maxindexL2_alt/N_d12);
                linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                isInfLower_alt = (ReturnMatrix_ii_d3z_alt(linidx_lower_alt) == -Inf);
                isInfUpper_alt = (ReturnMatrix_ii_d3z_alt(linidx_upper_alt) == -Inf);
                inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
                inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
                flag_ford3_alt(1,:,semizblock,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

                % --- tilde pass ---
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

                [~,maxindex1_tilde]=max(entireRHS_ii_d3z_tilde,[],2);
                midpoint_tilde(:,1,level1ii,:,:)=maxindex1_tilde;

                maxgap_tilde=squeeze(max(max(max(maxindex1_tilde(:,1,2:end,:,:)-maxindex1_tilde(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_tilde(ii)>0
                        loweredge_tilde=min(maxindex1_tilde(:,1,ii,:,:),N_a1-maxgap_tilde(ii));
                        a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                        ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, ReturnFnParamsVec,3,0);
                        d2aprimez_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                        entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+DiscountedEV_tilde(d2aprimez_tilde);
                        [~,maxindex_tilde]=max(entireRHS_ii_d3z_tilde,[],2);
                        midpoint_tilde(:,1,curraindex_tilde,:,:)=maxindex_tilde+(loweredge_tilde-1);
                    else
                        loweredge_tilde=maxindex1_tilde(:,1,ii,:,:);
                        midpoint_tilde(:,1,curraindex_tilde,:,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                d2a1primea2bothz_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+reshape(DiscountedEVinterp_tilde(d2a1primea2bothz_tilde(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3z_tilde,[],1);
                V_ford3_tilde(:,semizblock,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
                allind_tilde=d_ind_tilde+N_d12*aind+N_d12*N_a*semizBind;
                Policy4_ford3_tilde(1,:,semizblock,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
                Policy4_ford3_tilde(2,:,semizblock,d3_c)=ceil(d_ind_tilde/N_d1);
                Policy4_ford3_tilde(3,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                Policy4_ford3_tilde(4,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
                linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind + N_d12*n2long*N_a*semizBind;
                isInfLower_tilde = (ReturnMatrix_ii_d3z_tilde(linidx_lower_tilde) == -Inf);
                isInfUpper_tilde = (ReturnMatrix_ii_d3z_tilde(linidx_upper_tilde) == -Inf);
                inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
                inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
                flag_ford3_tilde(1,:,semizblock,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
            end
        end

    elseif vfoptions.lowmemory==2 % joint loop over bothz

        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEVinterp_alt=permute(interp1(a1_gridvals,permute(DiscountedEV_alt,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            % d1-dim is implicit singleton; broadcasts at use sites

            DiscountedEV_tilde=beta0beta*EVbase_qh;
            DiscountedEVinterp_tilde=permute(interp1(a1_gridvals,permute(DiscountedEV_tilde,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            % d1-dim is implicit singleton; broadcasts at use sites

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

            % --- alt pass ---
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,z_c);
                DiscountedEVinterp_z_alt=DiscountedEVinterp_alt(:,:,:,:,z_c);

                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+repelem(DiscountedEV_z_alt,N_d1,1,1,1);

                [~,maxindex1_alt]=max(entireRHS_ii_d3z_alt,[],2);
                midpoint_alt(:,1,level1ii,:)=maxindex1_alt;

                maxgap_alt=squeeze(max(max(maxindex1_alt(:,1,2:end,:)-maxindex1_alt(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_alt=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_alt(ii)>0
                        loweredge_alt=min(maxindex1_alt(:,1,ii,:),N_a1-maxgap_alt(ii));
                        a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                        ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_alt(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes_alt), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d2aprime_alt=d2ind+N_d2*(a1primeindexes_alt-1)+N_d2*N_a1*a2ind;
                        entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+DiscountedEV_z_alt(d2aprime_alt);
                        [~,maxindex_alt]=max(entireRHS_ii_d3z_alt,[],2);
                        midpoint_alt(:,1,curraindex_alt,:)=maxindex_alt+(loweredge_alt-1);
                    else
                        loweredge_alt=maxindex1_alt(:,1,ii,:);
                        midpoint_alt(:,1,curraindex_alt,:)=repelem(loweredge_alt,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_alt=max(min(midpoint_alt,n_a1(1)-1),2);
                a1primeindexesfine_alt=(midpoint_alt+(midpoint_alt-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_alt=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine_alt), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                d2a1primea2_alt=d2ind+N_d2*(a1primeindexesfine_alt-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z_alt=ReturnMatrix_ii_d3z_alt+reshape(DiscountedEVinterp_z_alt(d2a1primea2_alt(:)),[N_d12*n2long,N_a1*N_a2]);
                [Vtempii_alt,maxindexL2_alt]=max(entireRHS_ii_d3z_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtempii_alt,1);
                d_ind_alt=rem(maxindexL2_alt-1,N_d12)+1;
                allind_alt=d_ind_alt+N_d12*aind;
                Policy4_ford3_alt(1,:,z_c,d3_c)=rem(d_ind_alt-1,N_d1)+1;
                Policy4_ford3_alt(2,:,z_c,d3_c)=ceil(d_ind_alt/N_d1);
                Policy4_ford3_alt(3,:,z_c,d3_c)=shiftdim(squeeze(midpoint_alt(allind_alt)),-1);
                Policy4_ford3_alt(4,:,z_c,d3_c)=shiftdim(ceil(maxindexL2_alt/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_alt = ceil(maxindexL2_alt/N_d12);
                linidx_lower_alt = d_ind_alt                   + N_d12*n2long*aind;
                linidx_upper_alt = d_ind_alt + N_d12*(n2long-1) + N_d12*n2long*aind;
                isInfLower_alt = (ReturnMatrix_ii_d3z_alt(linidx_lower_alt) == -Inf);
                isInfUpper_alt = (ReturnMatrix_ii_d3z_alt(linidx_upper_alt) == -Inf);
                inLowerStrict_alt = (L2offset_alt >= 2)         & (L2offset_alt <= n2short+1);
                inUpperStrict_alt = (L2offset_alt >= n2short+3) & (L2offset_alt <= n2long-1);
                flag_ford3_alt(1,:,z_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_alt & isInfLower_alt) - (inUpperStrict_alt & isInfUpper_alt)),-1);

            % --- tilde pass ---
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,z_c);
                DiscountedEVinterp_z_tilde=DiscountedEVinterp_tilde(:,:,:,:,z_c);

                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+repelem(DiscountedEV_z_tilde,N_d1,1,1,1);

                [~,maxindex1_tilde]=max(entireRHS_ii_d3z_tilde,[],2);
                midpoint_tilde(:,1,level1ii,:)=maxindex1_tilde;

                maxgap_tilde=squeeze(max(max(maxindex1_tilde(:,1,2:end,:)-maxindex1_tilde(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex_tilde=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap_tilde(ii)>0
                        loweredge_tilde=min(maxindex1_tilde(:,1,ii,:),N_a1-maxgap_tilde(ii));
                        a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                        ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_tilde(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes_tilde), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d2aprime_tilde=d2ind+N_d2*(a1primeindexes_tilde-1)+N_d2*N_a1*a2ind;
                        entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+DiscountedEV_z_tilde(d2aprime_tilde);
                        [~,maxindex_tilde]=max(entireRHS_ii_d3z_tilde,[],2);
                        midpoint_tilde(:,1,curraindex_tilde,:)=maxindex_tilde+(loweredge_tilde-1);
                    else
                        loweredge_tilde=maxindex1_tilde(:,1,ii,:);
                        midpoint_tilde(:,1,curraindex_tilde,:)=repelem(loweredge_tilde,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint_tilde=max(min(midpoint_tilde,n_a1(1)-1),2);
                a1primeindexesfine_tilde=(midpoint_tilde+(midpoint_tilde-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z_tilde=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine_tilde), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                d2a1primea2_tilde=d2ind+N_d2*(a1primeindexesfine_tilde-1)+N_d2*N_a1prime*a2ind;
                entireRHS_ii_d3z_tilde=ReturnMatrix_ii_d3z_tilde+reshape(DiscountedEVinterp_z_tilde(d2a1primea2_tilde(:)),[N_d12*n2long,N_a1*N_a2]);
                [Vtempii_tilde,maxindexL2_tilde]=max(entireRHS_ii_d3z_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtempii_tilde,1);
                d_ind_tilde=rem(maxindexL2_tilde-1,N_d12)+1;
                allind_tilde=d_ind_tilde+N_d12*aind;
                Policy4_ford3_tilde(1,:,z_c,d3_c)=rem(d_ind_tilde-1,N_d1)+1;
                Policy4_ford3_tilde(2,:,z_c,d3_c)=ceil(d_ind_tilde/N_d1);
                Policy4_ford3_tilde(3,:,z_c,d3_c)=shiftdim(squeeze(midpoint_tilde(allind_tilde)),-1);
                Policy4_ford3_tilde(4,:,z_c,d3_c)=shiftdim(ceil(maxindexL2_tilde/N_d12),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset_tilde = ceil(maxindexL2_tilde/N_d12);
                linidx_lower_tilde = d_ind_tilde                   + N_d12*n2long*aind;
                linidx_upper_tilde = d_ind_tilde + N_d12*(n2long-1) + N_d12*n2long*aind;
                isInfLower_tilde = (ReturnMatrix_ii_d3z_tilde(linidx_lower_tilde) == -Inf);
                isInfUpper_tilde = (ReturnMatrix_ii_d3z_tilde(linidx_upper_tilde) == -Inf);
                inLowerStrict_tilde = (L2offset_tilde >= 2)         & (L2offset_tilde <= n2short+1);
                inUpperStrict_tilde = (L2offset_tilde >= n2short+3) & (L2offset_tilde <= n2long-1);
                flag_ford3_tilde(1,:,z_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict_tilde & isInfLower_tilde) - (inUpperStrict_tilde & isInfUpper_tilde)),-1);
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,jj)=V_jj;
    Policyalt(3,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=4*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policyalt(1,:,:,jj)=reshape(Policy4_ford3_alt(1+temp),[1,N_a,N_bothz]);
    Policyalt(2,:,:,jj)=reshape(Policy4_ford3_alt(2+temp),[1,N_a,N_bothz]);
    Policyalt(4,:,:,jj)=reshape(Policy4_ford3_alt(3+temp),[1,N_a,N_bothz]);
    Policyalt(5,:,:,jj)=reshape(Policy4_ford3_alt(4+temp),[1,N_a,N_bothz]);
    flat_idx=(1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1);
    PolicyL2flagalt(1,:,:,jj)=reshape(flag_ford3_alt(flat_idx),[1,N_a,N_bothz]);

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,jj)=V_jj;
    Policy(3,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=4*( (1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1) -1);
    Policy(1,:,:,jj)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz]);
    Policy(2,:,:,jj)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz]);
    Policy(4,:,:,jj)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz]);
    Policy(5,:,:,jj)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz]);
    flat_idx=(1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1);
    PolicyL2flag(1,:,:,jj)=reshape(flag_ford3_tilde(flat_idx),[1,N_a,N_bothz]);

end


%% With grid interpolation, switch from midpoint to lower grid index
% Currently Policy(4,:) is the midpoint, and Policy(5,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(4,:) to 'lower grid point' and then have Policy(5,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(5,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(4,:,:,:)=Policy(4,:,:,:)-adjust; % lower grid point
Policy(5,:,:,:)=adjust.*Policy(5,:,:,:)+(1-adjust).*(Policy(5,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy; PolicyL2flag];

adjustalt=(Policyalt(5,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policyalt(4,:,:,:)=Policyalt(4,:,:,:)-adjustalt; % lower grid point
Policyalt(5,:,:,:)=adjustalt.*Policyalt(5,:,:,:)+(1-adjustalt).*(Policyalt(5,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policyalt=[Policyalt; PolicyL2flagalt];


end
