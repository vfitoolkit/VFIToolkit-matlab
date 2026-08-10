function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI2A_nod1_raw(n_d2, n_a, n_z, n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive QH + SemiExo + GI2A (grid interpolation layer on the first endogenous
% state only, a2 enumerated in full), no d1, with z, no e.
%
% Naive: Valt_j   = max u + beta*E[Valt_{j+1}]         (exponential discounter)
%        Vtilde_j = max u + beta_0*beta*E[Valt_{j+1}]  (agent's choice)
% The two discount factors generally pick different midpoints, so each pass
% re-derives its own midpoint/a1primeindexes/layer-2 return matrix.

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;

Valt=zeros(N_a,N_bothz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_j,'gpuArray');
% Policy: 4 channels [d2, a1prime midpoint, a2prime, a1prime L2]
Policy=zeros(4,N_a,N_bothz,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_bothz,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
Policyalt=zeros(4,N_a,N_bothz,N_j,'gpuArray'); % exponential discounter's optimal choice
PolicyL2flagalt=2*ones(1,N_a,N_bothz,N_j,'gpuArray');

%% Split a into a1 and a2 (a1 is interpolated, a2 is on the standard grid)
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

%% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

%% Precompute indexing helpers (treating bothz like z)
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);
zind =shiftdim(gpuArray(0:1:N_bothz-1),-1);
zBind=shiftdim(gpuArray(0:1:N_bothz-1),-4);
a12ind=gpuArray(0:1:N_a1*N_a2-1);

special_n_d2=ones(1,length(n_d2));

% lowmemory: which shocks are looped vs vectorised (spec: =1 loop z, vectorise semiz; =2 joint loop over bothz)
if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
    semizind =shiftdim(gpuArray(0:1:N_semiz-1),-1); % semiz-block analogue of zind (L1)
    semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-4); % semiz-block analogue of zBind (L1)
elseif vfoptions.lowmemory==2
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

%% Preallocate per-d2 storage
V_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
mid_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2a1_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2a2_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
flag_ford2=2*ones(N_a,N_bothz,N_d2,'gpuArray');
V_ford2alt=zeros(N_a,N_bothz,N_d2,'gpuArray');
mid_ford2alt=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2a1_ford2alt=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2a2_ford2alt=zeros(N_a,N_bothz,N_d2,'gpuArray');
flag_ford2alt=2*ones(N_a,N_bothz,N_d2,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

  if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);

        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        flag_ford2(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

  elseif vfoptions.lowmemory==1 % loop z, vectorise semiz
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);

            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 1, 0);
            [~,maxindex]=max(ReturnMatrix,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 2, 0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,z): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,semizblock,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,semizblock,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,semizblock,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a2,1);
        end
    end

  elseif vfoptions.lowmemory==2 % joint loop over bothz
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);

            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 1, 0);
            [~,maxindex]=max(ReturnMatrix,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 2, 0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,z): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,z_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,z_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a2,1);
        end
    end
  end

    [V_jj,d2_max]=max(V_ford2,[],3);
    Valt(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(d2_max,-1);
    d2_max_lin=reshape(d2_max,[N_a*N_bothz,1]);
    idx=(1:N_a*N_bothz)'+(N_a*N_bothz)*(d2_max_lin-1);
    Policy(2,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_bothz]);
    Policy(3,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz]);
    PolicyL2flag(1,:,:,N_j)=reshape(flag_ford2(idx),[1,N_a,N_bothz]);

    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    % terminal period: QH and exponential discounter coincide
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j);
    PolicyL2flagalt(1,:,:,N_j)=PolicyL2flag(1,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;
    V_next=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

      if vfoptions.lowmemory==0
        EV=V_next.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_bothz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);

        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        %% Valt (beta) -- exponential discounter (also gives Policyalt)
        entireRHSalt=ReturnMatrix+beta*shiftdim(EV,-1);
        [~,maxindexalt]=max(entireRHSalt,[],2);
        midpointalt=max(min(maxindexalt,n_a1-1),2);

        a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1prime_grid(a1primeindexesalt), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprimealt=a1primeindexesalt+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimealt),[n2long*N_a2,N_a,N_bothz]);
        [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
        maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
        maxindexL2a2alt=ceil(maxindexL2alt/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_loweralt  = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upperalt  = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
        inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
        inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
        flag_ford2alt(:,:,d2_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), 1);

        V_ford2alt(:,:,d2_c)=shiftdim(Vtempiialt,1);
        mid_ford2alt(:,:,d2_c)=midpointalt(maxindexL2a2alt+N_a2*a12ind+N_a2*N_a*zind);
        L2a1_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a1alt,1);
        L2a2_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a2alt,1);

        %% Vtilde (beta0*beta) -- the QH agent's own choice
        entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_bothz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        flag_ford2(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);

      elseif vfoptions.lowmemory==1 % loop z, vectorise semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            EV=V_next.*shiftdim(pi_bothz(semizblock,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);

            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 1, 0);
            %% Valt (beta) -- exponential discounter (also gives Policyalt)
            entireRHSalt=ReturnMatrix+beta*shiftdim(EV,-1);
            [~,maxindexalt]=max(entireRHSalt,[],2);
            midpointalt=max(min(maxindexalt,n_a1-1),2);

            a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1prime_grid(a1primeindexesalt), a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 2, 0);
            aprimealt=a1primeindexesalt+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimealt),[n2long*N_a2,N_a,N_semiz]);
            [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
            maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
            maxindexL2a2alt=ceil(maxindexL2alt/n2long);

            % L2 flag (per d2,z): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_loweralt  = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upperalt  = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
            inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
            inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
            flag_ford2alt(:,semizblock,d2_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), 1);

            V_ford2alt(:,semizblock,d2_c)=shiftdim(Vtempiialt,1);
            mid_ford2alt(:,semizblock,d2_c)=midpointalt(maxindexL2a2alt+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2alt(:,semizblock,d2_c)=shiftdim(maxindexL2a1alt,1);
            L2a2_ford2alt(:,semizblock,d2_c)=shiftdim(maxindexL2a2alt,1);

            %% Vtilde (beta0*beta) -- the QH agent's own choice
            entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,z): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,semizblock,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,semizblock,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,semizblock,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a2,1);
        end

      elseif vfoptions.lowmemory==2 % joint loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            EV=V_next.*shiftdim(pi_bothz(z_c,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);

            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 1, 0);
            %% Valt (beta) -- exponential discounter (also gives Policyalt)
            entireRHSalt=ReturnMatrix+beta*shiftdim(EV,-1);
            [~,maxindexalt]=max(entireRHSalt,[],2);
            midpointalt=max(min(maxindexalt,n_a1-1),2);

            a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1prime_grid(a1primeindexesalt), a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 2, 0);
            aprimealt=a1primeindexesalt+N_a1fine*a2ind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimealt),[n2long*N_a2,N_a]);
            [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
            maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
            maxindexL2a2alt=ceil(maxindexL2alt/n2long);

            % L2 flag (per d2,z): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_loweralt  = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            linidx_upperalt  = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
            inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
            inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
            flag_ford2alt(:,z_c,d2_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), 1);

            V_ford2alt(:,z_c,d2_c)=shiftdim(Vtempiialt,1);
            mid_ford2alt(:,z_c,d2_c)=midpointalt(maxindexL2a2alt+N_a2*a12ind);
            L2a1_ford2alt(:,z_c,d2_c)=shiftdim(maxindexL2a1alt,1);
            L2a2_ford2alt(:,z_c,d2_c)=shiftdim(maxindexL2a2alt,1);

            %% Vtilde (beta0*beta) -- the QH agent's own choice
            entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,z): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,z_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,z_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a2,1);
        end
      end
    end

    [Valt_jj,d2_maxalt]=max(V_ford2alt,[],3);
    Valt(:,:,N_j)=Valt_jj;
    Policyalt(1,:,:,N_j)=shiftdim(d2_maxalt,-1);
    d2_maxalt_lin=reshape(d2_maxalt,[N_a*N_bothz,1]);
    idxalt=(1:N_a*N_bothz)'+(N_a*N_bothz)*(d2_maxalt_lin-1);
    Policyalt(2,:,:,N_j)=reshape(mid_ford2alt(idxalt), [1,N_a,N_bothz]);
    Policyalt(3,:,:,N_j)=reshape(L2a2_ford2alt(idxalt),[1,N_a,N_bothz]);
    Policyalt(4,:,:,N_j)=reshape(L2a1_ford2alt(idxalt),[1,N_a,N_bothz]);
    PolicyL2flagalt(1,:,:,N_j)=reshape(flag_ford2alt(idxalt),[1,N_a,N_bothz]);

    [V_jj,d2_max]=max(V_ford2,[],3);
    Vtilde(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(d2_max,-1);
    d2_max_lin=reshape(d2_max,[N_a*N_bothz,1]);
    idx=(1:N_a*N_bothz)'+(N_a*N_bothz)*(d2_max_lin-1);
    Policy(2,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_bothz]);
    Policy(3,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz]);
    PolicyL2flag(1,:,:,N_j)=reshape(flag_ford2(idx),[1,N_a,N_bothz]);
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
        pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

      if vfoptions.lowmemory==0
        EV=V_next.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_bothz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);

        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1, 0);
        %% Valt (beta) -- exponential discounter (also gives Policyalt)
        entireRHSalt=ReturnMatrix+beta*shiftdim(EV,-1);
        [~,maxindexalt]=max(entireRHSalt,[],2);
        midpointalt=max(min(maxindexalt,n_a1-1),2);

        a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1prime_grid(a1primeindexesalt), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprimealt=a1primeindexesalt+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimealt),[n2long*N_a2,N_a,N_bothz]);
        [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
        maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
        maxindexL2a2alt=ceil(maxindexL2alt/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_loweralt  = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upperalt  = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
        inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
        inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
        flag_ford2alt(:,:,d2_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), 1);

        V_ford2alt(:,:,d2_c)=shiftdim(Vtempiialt,1);
        mid_ford2alt(:,:,d2_c)=midpointalt(maxindexL2a2alt+N_a2*a12ind+N_a2*N_a*zind);
        L2a1_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a1alt,1);
        L2a2_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a2alt,1);

        %% Vtilde (beta0*beta) -- the QH agent's own choice
        entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_bothz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        flag_ford2(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);

      elseif vfoptions.lowmemory==1 % loop z, vectorise semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,jj);
            EV=V_next.*shiftdim(pi_bothz(semizblock,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);

            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 1, 0);
            %% Valt (beta) -- exponential discounter (also gives Policyalt)
            entireRHSalt=ReturnMatrix+beta*shiftdim(EV,-1);
            [~,maxindexalt]=max(entireRHSalt,[],2);
            midpointalt=max(min(maxindexalt,n_a1-1),2);

            a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1prime_grid(a1primeindexesalt), a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 2, 0);
            aprimealt=a1primeindexesalt+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimealt),[n2long*N_a2,N_a,N_semiz]);
            [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
            maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
            maxindexL2a2alt=ceil(maxindexL2alt/n2long);

            % L2 flag (per d2,z): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_loweralt  = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upperalt  = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
            inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
            inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
            flag_ford2alt(:,semizblock,d2_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), 1);

            V_ford2alt(:,semizblock,d2_c)=shiftdim(Vtempiialt,1);
            mid_ford2alt(:,semizblock,d2_c)=midpointalt(maxindexL2a2alt+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2alt(:,semizblock,d2_c)=shiftdim(maxindexL2a1alt,1);
            L2a2_ford2alt(:,semizblock,d2_c)=shiftdim(maxindexL2a2alt,1);

            %% Vtilde (beta0*beta) -- the QH agent's own choice
            entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,z): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,semizblock,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,semizblock,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,semizblock,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a2,1);
        end

      elseif vfoptions.lowmemory==2 % joint loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,jj);
            EV=V_next.*shiftdim(pi_bothz(z_c,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);

            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 1, 0);
            %% Valt (beta) -- exponential discounter (also gives Policyalt)
            entireRHSalt=ReturnMatrix+beta*shiftdim(EV,-1);
            [~,maxindexalt]=max(entireRHSalt,[],2);
            midpointalt=max(min(maxindexalt,n_a1-1),2);

            a1primeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1prime_grid(a1primeindexesalt), a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 2, 0);
            aprimealt=a1primeindexesalt+N_a1fine*a2ind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimealt),[n2long*N_a2,N_a]);
            [Vtempiialt,maxindexL2alt]=max(entireRHS_iialt,[],1);
            maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
            maxindexL2a2alt=ceil(maxindexL2alt/n2long);

            % L2 flag (per d2,z): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_loweralt  = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            linidx_upperalt  = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            isInfLoweralt    = (ReturnMatrix_iialt(linidx_loweralt) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(linidx_upperalt) == -Inf);
            inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
            inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
            flag_ford2alt(:,z_c,d2_c) = shiftdim(2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt), 1);

            V_ford2alt(:,z_c,d2_c)=shiftdim(Vtempiialt,1);
            mid_ford2alt(:,z_c,d2_c)=midpointalt(maxindexL2a2alt+N_a2*a12ind);
            L2a1_ford2alt(:,z_c,d2_c)=shiftdim(maxindexL2a1alt,1);
            L2a2_ford2alt(:,z_c,d2_c)=shiftdim(maxindexL2a2alt,1);

            %% Vtilde (beta0*beta) -- the QH agent's own choice
            entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,z): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,z_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,z_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a2,1);
        end
      end
    end

    [Valt_jj,d2_maxalt]=max(V_ford2alt,[],3);
    Valt(:,:,jj)=Valt_jj;
    Policyalt(1,:,:,jj)=shiftdim(d2_maxalt,-1);
    d2_maxalt_lin=reshape(d2_maxalt,[N_a*N_bothz,1]);
    idxalt=(1:N_a*N_bothz)'+(N_a*N_bothz)*(d2_maxalt_lin-1);
    Policyalt(2,:,:,jj)=reshape(mid_ford2alt(idxalt), [1,N_a,N_bothz]);
    Policyalt(3,:,:,jj)=reshape(L2a2_ford2alt(idxalt),[1,N_a,N_bothz]);
    Policyalt(4,:,:,jj)=reshape(L2a1_ford2alt(idxalt),[1,N_a,N_bothz]);
    PolicyL2flagalt(1,:,:,jj)=reshape(flag_ford2alt(idxalt),[1,N_a,N_bothz]);

    [V_jj,d2_max]=max(V_ford2,[],3);
    Vtilde(:,:,jj)=V_jj;
    Policy(1,:,:,jj)=shiftdim(d2_max,-1);
    d2_max_lin=reshape(d2_max,[N_a*N_bothz,1]);
    idx=(1:N_a*N_bothz)'+(N_a*N_bothz)*(d2_max_lin-1);
    Policy(2,:,:,jj)=reshape(mid_ford2(idx), [1,N_a,N_bothz]);
    Policy(3,:,:,jj)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz]);
    Policy(4,:,:,jj)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz]);
    PolicyL2flag(1,:,:,jj)=reshape(flag_ford2(idx),[1,N_a,N_bothz]);
end


%% Convert Policy(2) from midpoint to lower grid point, Policy(4) from -n2short-1:1+n2short to 1:n2short+2
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

adjustalt=(Policyalt(4,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policyalt(2,:,:,:)=Policyalt(2,:,:,:)-adjustalt; % lower grid point
Policyalt(4,:,:,:)=adjustalt.*Policyalt(4,:,:,:)+(1-adjustalt).*(Policyalt(4,:,:,:)-n2short-1);

Policyalt=[Policyalt; PolicyL2flagalt];

% Policy=Policy(1,:,:,:)+N_d2*(Policy(2,:,:,:)-1)+N_d2*N_a1*(Policy(3,:,:,:)-1)+N_d2*N_a1*N_a2*(Policy(4,:,:,:)-1)+N_d2*N_a1*N_a2*(n2short+2)*(PolicyL2flag-1);


end
