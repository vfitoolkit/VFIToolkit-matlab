function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI2A_nod1_noz_raw(n_d2, n_a, n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive QH + SemiExo + GI2A (grid interpolation layer on the first endogenous
% state only, a2 enumerated in full), no d1, no z, no e.
%
% Naive: Valt_j   = max u + beta*E[Valt_{j+1}]         (exponential discounter)
%        Vtilde_j = max u + beta_0*beta*E[Valt_{j+1}]  (agent's choice)
% The two discount factors generally pick different midpoints, so each pass
% re-derives its own midpoint/a1primeindexes/layer-2 return matrix.

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);

Valt=zeros(N_a,N_semiz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz,N_j,'gpuArray');
% Policy: 4 channels [d2, a1prime midpoint, a2prime, a1prime L2]
Policy=zeros(4,N_a,N_semiz,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
Policyalt=zeros(4,N_a,N_semiz,N_j,'gpuArray'); % exponential discounter's optimal choice
PolicyL2flagalt=2*ones(1,N_a,N_semiz,N_j,'gpuArray');

%% Split a into a1 and a2 (a1 is interpolated, a2 is on the standard grid)
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

%% Grid interpolation
n2short=vfoptions.ngridinterp; % evenly spaced points between each a1 grid pair
n2long=vfoptions.ngridinterp*2+3; % total a1prime points around midpoint at layer 2
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

%% Precompute indexing helpers (treating semiz like z)
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);     % singleton on first dim
semizind =shiftdim(gpuArray(0:1:N_semiz-1),-1);  % singleton on first dim
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-4);  % singleton on first four dims (for aprime in layer 2)
a12ind=gpuArray(0:1:N_a1*N_a2-1);

special_n_d2=ones(1,length(n_d2));

% lowmemory: which shocks are looped vs vectorised (spec: =1 loop semiz)
if vfoptions.lowmemory==1
    special_n_semiz=ones(1,length(n_semiz));
end

%% Preallocate per-d2 storage
V_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
mid_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
L2a1_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
L2a2_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
flag_ford2=2*ones(N_a,N_semiz,N_d2,'gpuArray');
V_ford2alt=zeros(N_a,N_semiz,N_d2,'gpuArray');
mid_ford2alt=zeros(N_a,N_semiz,N_d2,'gpuArray');
L2a1_ford2alt=zeros(N_a,N_semiz,N_d2,'gpuArray');
L2a2_ford2alt=zeros(N_a,N_semiz,N_d2,'gpuArray');
flag_ford2alt=2*ones(N_a,N_semiz,N_d2,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No continuation. d2 still appears in ReturnFn — loop over d2_c.
  if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);

        % Layer 1: coarse
        % CreateReturnFnMatrix_Disc_DC2A returns shape [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_z] (with special_n_d2=[1] -> N_d=1; semiz used as z)
        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        [~,maxindex]=max(ReturnMatrix,[],2); % max over a1prime (dim 2). maxindex shape: [1,1,N_a2,N_a1,N_a2,N_semiz]
        midpoint=max(min(maxindex,n_a1-1),2);

        % Layer 2: fine around midpoint
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % shape [1,n2long,N_a2,N_a1,N_a2,N_semiz]
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        % With N_d=1, maxindexL2 ranges 1..n2long*N_a2: a1primeL2 fastest, a2prime slowest
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        flag_ford2(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        % midpoint shape [1,1,N_a2,N_a1,N_a2,N_semiz]; linear index:
        % i = maxindexL2a2 + N_a2*a12ind + N_a2*N_a*semizind  (a12ind = (a1-1)+N_a1*(a2-1) ranges 0..N_a-1)
        mid_ford2(:,:,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

  elseif vfoptions.lowmemory==1 % loop semiz
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for semiz_c=1:N_semiz
            semiz_val=semiz_gridvals_J(semiz_c,:,N_j);

            % Layer 1: coarse
            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_semiz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 1, 0);
            [~,maxindex]=max(ReturnMatrix,[],2); % max over a1prime (dim 2)
            midpoint=max(min(maxindex,n_a1-1),2);

            % Layer 2: fine around midpoint
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_semiz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 2, 0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,semiz): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,semiz_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,semiz_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,semiz_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a2,1);
        end
    end
  end

    % Take max over d2_c
    [V_jj,d2_max]=max(V_ford2,[],3); % V_jj, d2_max shape: [N_a, N_semiz]
    Valt(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(d2_max,-1);
    d2_max_lin=reshape(d2_max,[N_a*N_semiz,1]);
    idx=(1:N_a*N_semiz)'+(N_a*N_semiz)*(d2_max_lin-1); % into ford2 arrays [N_a,N_semiz,N_d2]
    Policy(2,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz]);
    PolicyL2flag(1,:,:,N_j)=reshape(flag_ford2(idx),[1,N_a,N_semiz]);

    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    % terminal period: QH and exponential discounter coincide
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j);
    PolicyL2flagalt(1,:,:,N_j)=PolicyL2flag(1,:,:,N_j);
else
    % Using V_Jplus1 — include continuation value
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;
    V_next=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_semiz=pi_semiz_J(:,:,d2_c,N_j); % [N_semiz_from, N_semiz_to]

      if vfoptions.lowmemory==0
        % Expectation over semiz' given (semiz_from, d2): EV[a, semiz_from] = sum_{semiz_to} pi_semiz(from, to) * V(a, semiz_to)
        EV=V_next.*shiftdim(pi_semiz',-1); % V[N_a,N_semiz_to,1] .* [1,N_semiz_to,N_semiz_from] -> [N_a,N_semiz_to,N_semiz_from]
        EV(isnan(EV))=0;
        EV=sum(EV,2); % collapse semiz_to dim. EV shape: [N_a, 1, N_semiz_from]
        EV=reshape(EV,[N_a1,N_a2,1,1,N_semiz]); % shape for broadcasting with ReturnMatrix
        EVinterp=interp1(a1_grid,EV,a1prime_grid); % interpolate EV over a1 fine grid -> [N_a1fine, N_a2, 1, 1, N_semiz]

        % Layer 1
        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        %% Valt (beta) -- exponential discounter (also gives Policyalt)
        entireRHSalt=ReturnMatrix+beta*shiftdim(EV,-1);
        [~,maxindexalt]=max(entireRHSalt,[],2);
        midpointalt=max(min(maxindexalt,n_a1-1),2);

        % Layer 2
        a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1prime_grid(a1primeindexesalt), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprimealt=a1primeindexesalt+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimealt),[n2long*N_a2,N_a,N_semiz]);
        [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
        maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
        maxindexL2a2alt=ceil(maxindexL2alt/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_loweralt  = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        linidx_upperalt  = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
        inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
        inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
        flag_ford2alt(:,:,d2_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), 1);

        V_ford2alt(:,:,d2_c)=shiftdim(Vtempiialt,1);
        mid_ford2alt(:,:,d2_c)=midpointalt(maxindexL2a2alt+N_a2*a12ind+N_a2*N_a*semizind);
        L2a1_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a1alt,1);
        L2a2_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a2alt,1);

        %% Vtilde (beta0*beta) -- the QH agent's own choice
        entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        % Layer 2
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        flag_ford2(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);

      elseif vfoptions.lowmemory==1 % loop semiz
        for semiz_c=1:N_semiz
            semiz_val=semiz_gridvals_J(semiz_c,:,N_j);

            % Expectation over semiz' given (semiz_c, d2): EV[a] = sum_{semiz_to} pi_semiz(semiz_c, semiz_to) * V(a, semiz_to)
            EV=V_next.*shiftdim(pi_semiz(semiz_c,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);

            % Layer 1
            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_semiz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 1, 0);
            %% Valt (beta) -- exponential discounter (also gives Policyalt)
            entireRHSalt=ReturnMatrix+beta*shiftdim(EV,-1);
            [~,maxindexalt]=max(entireRHSalt,[],2);
            midpointalt=max(min(maxindexalt,n_a1-1),2);

            % Layer 2
            a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_semiz, d2_val, a1prime_grid(a1primeindexesalt), a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 2, 0);
            aprimealt=a1primeindexesalt+N_a1fine*a2ind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimealt),[n2long*N_a2,N_a]);
            [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
            maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
            maxindexL2a2alt=ceil(maxindexL2alt/n2long);

            % L2 flag (per d2,semiz): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_loweralt  = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            linidx_upperalt  = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
            inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
            inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
            flag_ford2alt(:,semiz_c,d2_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), 1);

            V_ford2alt(:,semiz_c,d2_c)=shiftdim(Vtempiialt,1);
            mid_ford2alt(:,semiz_c,d2_c)=midpointalt(maxindexL2a2alt+N_a2*a12ind);
            L2a1_ford2alt(:,semiz_c,d2_c)=shiftdim(maxindexL2a1alt,1);
            L2a2_ford2alt(:,semiz_c,d2_c)=shiftdim(maxindexL2a2alt,1);

            %% Vtilde (beta0*beta) -- the QH agent's own choice
            entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            % Layer 2
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_semiz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,semiz): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,semiz_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,semiz_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,semiz_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a2,1);
        end
      end
    end

    [Valt_jj,d2_maxalt]=max(V_ford2alt,[],3);
    Valt(:,:,N_j)=Valt_jj;
    Policyalt(1,:,:,N_j)=shiftdim(d2_maxalt,-1);
    d2_maxalt_lin=reshape(d2_maxalt,[N_a*N_semiz,1]);
    idxalt=(1:N_a*N_semiz)'+(N_a*N_semiz)*(d2_maxalt_lin-1);
    Policyalt(2,:,:,N_j)=reshape(mid_ford2alt(idxalt), [1,N_a,N_semiz]);
    Policyalt(3,:,:,N_j)=reshape(L2a2_ford2alt(idxalt),[1,N_a,N_semiz]);
    Policyalt(4,:,:,N_j)=reshape(L2a1_ford2alt(idxalt),[1,N_a,N_semiz]);
    PolicyL2flagalt(1,:,:,N_j)=reshape(flag_ford2alt(idxalt),[1,N_a,N_semiz]);

    [V_jj,d2_max]=max(V_ford2,[],3);
    Vtilde(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(d2_max,-1);
    d2_max_lin=reshape(d2_max,[N_a*N_semiz,1]);
    idx=(1:N_a*N_semiz)'+(N_a*N_semiz)*(d2_max_lin-1);
    Policy(2,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz]);
    PolicyL2flag(1,:,:,N_j)=reshape(flag_ford2(idx),[1,N_a,N_semiz]);
end

%% Backward iteration
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    V_next=Valt(:,:,jj+1); % naive: continuation is the exponential value fn

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_semiz=pi_semiz_J(:,:,d2_c,jj);

      if vfoptions.lowmemory==0
        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);

        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1, 0);
        %% Valt (beta) -- exponential discounter (also gives Policyalt)
        entireRHSalt=ReturnMatrix+beta*shiftdim(EV,-1);
        [~,maxindexalt]=max(entireRHSalt,[],2);
        midpointalt=max(min(maxindexalt,n_a1-1),2);

        a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1prime_grid(a1primeindexesalt), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprimealt=a1primeindexesalt+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimealt),[n2long*N_a2,N_a,N_semiz]);
        [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
        maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
        maxindexL2a2alt=ceil(maxindexL2alt/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_loweralt  = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        linidx_upperalt  = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
        inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
        inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
        flag_ford2alt(:,:,d2_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), 1);

        V_ford2alt(:,:,d2_c)=shiftdim(Vtempiialt,1);
        mid_ford2alt(:,:,d2_c)=midpointalt(maxindexL2a2alt+N_a2*a12ind+N_a2*N_a*semizind);
        L2a1_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a1alt,1);
        L2a2_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a2alt,1);

        %% Vtilde (beta0*beta) -- the QH agent's own choice
        entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        flag_ford2(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);

      elseif vfoptions.lowmemory==1 % loop semiz
        for semiz_c=1:N_semiz
            semiz_val=semiz_gridvals_J(semiz_c,:,jj);

            EV=V_next.*shiftdim(pi_semiz(semiz_c,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);

            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_semiz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 1, 0);
            %% Valt (beta) -- exponential discounter (also gives Policyalt)
            entireRHSalt=ReturnMatrix+beta*shiftdim(EV,-1);
            [~,maxindexalt]=max(entireRHSalt,[],2);
            midpointalt=max(min(maxindexalt,n_a1-1),2);

            a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_semiz, d2_val, a1prime_grid(a1primeindexesalt), a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 2, 0);
            aprimealt=a1primeindexesalt+N_a1fine*a2ind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimealt),[n2long*N_a2,N_a]);
            [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
            maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
            maxindexL2a2alt=ceil(maxindexL2alt/n2long);

            % L2 flag (per d2,semiz): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_loweralt  = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            linidx_upperalt  = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
            inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
            inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
            flag_ford2alt(:,semiz_c,d2_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), 1);

            V_ford2alt(:,semiz_c,d2_c)=shiftdim(Vtempiialt,1);
            mid_ford2alt(:,semiz_c,d2_c)=midpointalt(maxindexL2a2alt+N_a2*a12ind);
            L2a1_ford2alt(:,semiz_c,d2_c)=shiftdim(maxindexL2a1alt,1);
            L2a2_ford2alt(:,semiz_c,d2_c)=shiftdim(maxindexL2a2alt,1);

            %% Vtilde (beta0*beta) -- the QH agent's own choice
            entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_semiz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,semiz): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,semiz_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,semiz_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,semiz_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a2,1);
        end
      end
    end

    [Valt_jj,d2_maxalt]=max(V_ford2alt,[],3);
    Valt(:,:,jj)=Valt_jj;
    Policyalt(1,:,:,jj)=shiftdim(d2_maxalt,-1);
    d2_maxalt_lin=reshape(d2_maxalt,[N_a*N_semiz,1]);
    idxalt=(1:N_a*N_semiz)'+(N_a*N_semiz)*(d2_maxalt_lin-1);
    Policyalt(2,:,:,jj)=reshape(mid_ford2alt(idxalt), [1,N_a,N_semiz]);
    Policyalt(3,:,:,jj)=reshape(L2a2_ford2alt(idxalt),[1,N_a,N_semiz]);
    Policyalt(4,:,:,jj)=reshape(L2a1_ford2alt(idxalt),[1,N_a,N_semiz]);
    PolicyL2flagalt(1,:,:,jj)=reshape(flag_ford2alt(idxalt),[1,N_a,N_semiz]);

    [V_jj,d2_max]=max(V_ford2,[],3);
    Vtilde(:,:,jj)=V_jj;
    Policy(1,:,:,jj)=shiftdim(d2_max,-1);
    d2_max_lin=reshape(d2_max,[N_a*N_semiz,1]);
    idx=(1:N_a*N_semiz)'+(N_a*N_semiz)*(d2_max_lin-1);
    Policy(2,:,:,jj)=reshape(mid_ford2(idx), [1,N_a,N_semiz]);
    Policy(3,:,:,jj)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz]);
    Policy(4,:,:,jj)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz]);
    PolicyL2flag(1,:,:,jj)=reshape(flag_ford2(idx),[1,N_a,N_semiz]);
end


%% Convert Policy(2) from midpoint to lower grid point, Policy(4) from -n2short-1:1+n2short range to 1:n2short+2 range
adjust=(Policy(4,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

adjustalt=(Policyalt(4,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policyalt(2,:,:,:)=Policyalt(2,:,:,:)-adjustalt; % lower grid point
Policyalt(4,:,:,:)=adjustalt.*Policyalt(4,:,:,:)+(1-adjustalt).*(Policyalt(4,:,:,:)-n2short-1);

Policyalt=[Policyalt; PolicyL2flagalt];

% Policy=Policy(1,:,:,:)+N_d2*(Policy(2,:,:,:)-1)+N_d2*N_a1*(Policy(3,:,:,:)-1)+N_d2*N_a1*N_a2*(Policy(4,:,:,:)-1)+N_d2*N_a1*N_a2*(n2short+2)*(PolicyL2flag-1);


end
