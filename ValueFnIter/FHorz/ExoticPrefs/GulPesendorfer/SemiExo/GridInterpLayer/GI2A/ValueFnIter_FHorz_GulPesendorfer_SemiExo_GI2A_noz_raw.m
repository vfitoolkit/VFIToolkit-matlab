function [V,Policy]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI2A_noz_raw(n_d1, n_d2, n_a, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% SemiExo + GI2A (two-endo grid interpolation), with d1, no z, no e.
% Combines SemiExo d2/pi_semiz iteration with the 2-endo GI2A machinery; d1 is in the ReturnFn but not in semiz transitions.
% Gul-Pesendorfer with a semi-exogenous state and the grid interpolation layer over two
% endogenous states: interpolation applies to a1prime only, a2prime is fully scanned in the
% fine stage. The tempted objective u+v+beta*EV goes through the standard per-d2 two-stage
% machinery (the coarse midpoint/argmax and the fine a1prime window are those of the TEMPTED
% objective, with the L2 -Inf flag based on u+v); the most-tempting term is the max of v over
% the FINE a1prime grid jointly with a2prime, found by the same two-stage scheme but around
% v's OWN coarse argmax (otherwise the chosen fine point could be more tempting than the
% coarse max of v, making the self-control cost negative): a per-d2 fine max is collected
% alongside each inner solve and the max over d2 is subtracted from V after the outer max
% (the '-max v' term is constant w.r.t. the choice given the state, so the subtraction after
% the d2-max is exact).

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
% Policy: 5 channels [d1, d2, a1prime midpoint, a2prime, a1prime L2]
Policy=zeros(6,N_a,N_semiz,N_j,'gpuArray');
Policy(6,:,:,:)=2; % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

%% Split a
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

%% Combine d1 and d2 grids (per-d2 slice used inside loop)
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

%% Indexing helpers
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);
semizind=gpuArray(0:1:N_semiz-1);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-4);
a12ind=gpuArray(0:1:N_a1*N_a2-1)';

% lowmemory: which shocks are looped vs vectorised ({semiz} spec: =1 loop semiz)
if vfoptions.lowmemory==1
    special_n_semiz=ones(1,length(n_semiz));
end

%% Preallocate per-d2 storage
V_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
d1_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
mid_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
L2a1_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
L2a2_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
flag_ford2=2*ones(N_a,N_semiz,N_d2,'gpuArray');
MostTempting_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray'); % per-d2 fine (d1,a1prime,a2prime) max of the temptation

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

  if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);

        % Coarse grid
        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec, 1, 0);
        [~,maxindex]=max(ReturnMatrix+TemptationMatrix,[],2);

        % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a1-1),2);
        a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a1prime_grid(a1primeindexesT), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec, 2, 0);
        MostTempting_ford2(:,:,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,a1prime,a2prime) max for this d2

        midpoint=max(min(maxindex,n_a1-1),2);

        % Fine grid
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec, 2, 0);
        Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
        [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);
        maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
        maxindexL2a=ceil(maxindexL2/N_d1);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d1, a2prime)
        mL2d1 = shiftdim(maxindexL2d1,1);
        mL2a1 = shiftdim(maxindexL2a1,1);
        mL2a2 = shiftdim(maxindexL2a2,1);
        linidx_lower  = mL2d1                  + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind;
        linidx_upper  = mL2d1 + N_d1*(n2long-1) + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind;
        isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
        isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
        inLowerStrict = (mL2a1 >= 2)         & (mL2a1 <= n2short+1);
        inUpperStrict = (mL2a1 >= n2short+3) & (mL2a1 <= n2long-1);
        flag_ford2(:,:,d2_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        d1_ford2(:,:,d2_c)=shiftdim(maxindexL2d1,1);
        mid_ford2(:,:,d2_c)=midpoint(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind+N_d1*N_a2*N_a*semizind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

  elseif vfoptions.lowmemory==1 % loop semiz
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        for semiz_c=1:N_semiz
            semiz_val=semiz_gridvals_J(semiz_c,:,N_j);

            % Coarse grid
            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 1, 0);
            TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_val, TemptationFnParamsVec, 1, 0);
            [~,maxindex]=max(ReturnMatrix+TemptationMatrix,[],2);

            % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
            [~,maxindexT]=max(TemptationMatrix,[],2);
            midpointT=max(min(maxindexT,n_a1-1),2);
            a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a1prime_grid(a1primeindexesT), a2_grid, a1_grid, a2_grid, semiz_val, TemptationFnParamsVec, 2, 0);
            MostTempting_ford2(:,semiz_c,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,a1prime,a2prime) max for this d2

            midpoint=max(min(maxindex,n_a1-1),2);

            % Fine grid
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 2, 0);
            TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, TemptationFnParamsVec, 2, 0);
            Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
            [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);
            maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
            maxindexL2a=ceil(maxindexL2/N_d1);
            maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2a/n2long);

            % L2 flag (per d2,semiz): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d1, a2prime)
            mL2d1 = shiftdim(maxindexL2d1,1);
            mL2a1 = shiftdim(maxindexL2a1,1);
            mL2a2 = shiftdim(maxindexL2a2,1);
            linidx_lower  = mL2d1                  + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind;
            linidx_upper  = mL2d1 + N_d1*(n2long-1) + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind;
            isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
            isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
            inLowerStrict = (mL2a1 >= 2)         & (mL2a1 <= n2short+1);
            inUpperStrict = (mL2a1 >= n2short+3) & (mL2a1 <= n2long-1);
            flag_ford2(:,semiz_c,d2_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            V_ford2(:,semiz_c,d2_c)=shiftdim(Vtempii,1);
            d1_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2d1,1);
            mid_ford2(:,semiz_c,d2_c)=midpoint(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind);
            L2a1_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a2,1);
        end
    end
  end

    [V_jj,d2_max]=max(V_ford2,[],3);
    V(:,:,N_j)=V_jj-max(MostTempting_ford2,[],3); % subtract the most-tempting term
    Policy(2,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_semiz;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(1,:,:,N_j)=reshape(d1_ford2(idx), [1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(mid_ford2(idx),[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz]);
    Policy(5,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz]);
    Policy(6,:,:,N_j)=reshape(flag_ford2(idx),[1,N_a,N_semiz]);
else
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j));
    V_next=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

  if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);

        % Coarse grid
        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec, 1, 0);
        entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);

        % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a1-1),2);
        a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a1prime_grid(a1primeindexesT), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec, 2, 0);
        MostTempting_ford2(:,:,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,a1prime,a2prime) max for this d2

        midpoint=max(min(maxindex,n_a1-1),2);

        % Fine grid
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), TemptationFnParamsVec, 2, 0);
        Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d1*n2long*N_a2,N_a,N_semiz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
        maxindexL2a=ceil(maxindexL2/N_d1);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d1, a2prime)
        mL2d1 = shiftdim(maxindexL2d1,1);
        mL2a1 = shiftdim(maxindexL2a1,1);
        mL2a2 = shiftdim(maxindexL2a2,1);
        linidx_lower  = mL2d1                  + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind;
        linidx_upper  = mL2d1 + N_d1*(n2long-1) + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind;
        isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
        isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
        inLowerStrict = (mL2a1 >= 2)         & (mL2a1 <= n2short+1);
        inUpperStrict = (mL2a1 >= n2short+3) & (mL2a1 <= n2long-1);
        flag_ford2(:,:,d2_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        d1_ford2(:,:,d2_c)=shiftdim(maxindexL2d1,1);
        mid_ford2(:,:,d2_c)=midpoint(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind+N_d1*N_a2*N_a*semizind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

  elseif vfoptions.lowmemory==1 % loop semiz
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_semiz=pi_semiz_J(:,:,d2_c,N_j);
        for semiz_c=1:N_semiz
            semiz_val=semiz_gridvals_J(semiz_c,:,N_j);

            EV=V_next.*shiftdim(pi_semiz(semiz_c,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2,1,1,1]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);

            % Coarse grid
            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 1, 0);
            TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_val, TemptationFnParamsVec, 1, 0);
            entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);
            [~,maxindex]=max(entireRHS,[],2);

            % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
            [~,maxindexT]=max(TemptationMatrix,[],2);
            midpointT=max(min(maxindexT,n_a1-1),2);
            a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a1prime_grid(a1primeindexesT), a2_grid, a1_grid, a2_grid, semiz_val, TemptationFnParamsVec, 2, 0);
            MostTempting_ford2(:,semiz_c,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,a1prime,a2prime) max for this d2

            midpoint=max(min(maxindex,n_a1-1),2);

            % Fine grid
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 2, 0);
            TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, TemptationFnParamsVec, 2, 0);
            Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d1*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
            maxindexL2a=ceil(maxindexL2/N_d1);
            maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2a/n2long);

            % L2 flag (per d2,semiz): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d1, a2prime)
            mL2d1 = shiftdim(maxindexL2d1,1);
            mL2a1 = shiftdim(maxindexL2a1,1);
            mL2a2 = shiftdim(maxindexL2a2,1);
            linidx_lower  = mL2d1                  + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind;
            linidx_upper  = mL2d1 + N_d1*(n2long-1) + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind;
            isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
            isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
            inLowerStrict = (mL2a1 >= 2)         & (mL2a1 <= n2short+1);
            inUpperStrict = (mL2a1 >= n2short+3) & (mL2a1 <= n2long-1);
            flag_ford2(:,semiz_c,d2_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            V_ford2(:,semiz_c,d2_c)=shiftdim(Vtempii,1);
            d1_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2d1,1);
            mid_ford2(:,semiz_c,d2_c)=midpoint(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind);
            L2a1_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a2,1);
        end
    end
  end

    [V_jj,d2_max]=max(V_ford2,[],3);
    V(:,:,N_j)=V_jj-max(MostTempting_ford2,[],3); % subtract the most-tempting term
    Policy(2,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_semiz;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(1,:,:,N_j)=reshape(d1_ford2(idx), [1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(mid_ford2(idx),[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz]);
    Policy(5,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz]);
    Policy(6,:,:,N_j)=reshape(flag_ford2(idx),[1,N_a,N_semiz]);
end

%% Backward iteration
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));

    V_next=V(:,:,jj+1);

  if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_semiz=pi_semiz_J(:,:,d2_c,jj);

        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);

        % Coarse grid
        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1, 0);
        TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), TemptationFnParamsVec, 1, 0);
        entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);

        % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a1-1),2);
        a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a1prime_grid(a1primeindexesT), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), TemptationFnParamsVec, 2, 0);
        MostTempting_ford2(:,:,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,a1prime,a2prime) max for this d2

        midpoint=max(min(maxindex,n_a1-1),2);

        % Fine grid
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), TemptationFnParamsVec, 2, 0);
        Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d1*n2long*N_a2,N_a,N_semiz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
        maxindexL2a=ceil(maxindexL2/N_d1);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d1, a2prime)
        mL2d1 = shiftdim(maxindexL2d1,1);
        mL2a1 = shiftdim(maxindexL2a1,1);
        mL2a2 = shiftdim(maxindexL2a2,1);
        linidx_lower  = mL2d1                  + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind;
        linidx_upper  = mL2d1 + N_d1*(n2long-1) + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind;
        isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
        isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
        inLowerStrict = (mL2a1 >= 2)         & (mL2a1 <= n2short+1);
        inUpperStrict = (mL2a1 >= n2short+3) & (mL2a1 <= n2long-1);
        flag_ford2(:,:,d2_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        d1_ford2(:,:,d2_c)=shiftdim(maxindexL2d1,1);
        mid_ford2(:,:,d2_c)=midpoint(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind+N_d1*N_a2*N_a*semizind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

  elseif vfoptions.lowmemory==1 % loop semiz
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_semiz=pi_semiz_J(:,:,d2_c,jj);
        for semiz_c=1:N_semiz
            semiz_val=semiz_gridvals_J(semiz_c,:,jj);

            EV=V_next.*shiftdim(pi_semiz(semiz_c,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2,1,1,1]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);

            % Coarse grid
            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 1, 0);
            TemptationMatrix=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_val, TemptationFnParamsVec, 1, 0);
            entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);
            [~,maxindex]=max(entireRHS,[],2);

            % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
            [~,maxindexT]=max(TemptationMatrix,[],2);
            midpointT=max(min(maxindexT,n_a1-1),2);
            a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a1prime_grid(a1primeindexesT), a2_grid, a1_grid, a2_grid, semiz_val, TemptationFnParamsVec, 2, 0);
            MostTempting_ford2(:,semiz_c,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,a1prime,a2prime) max for this d2

            midpoint=max(min(maxindex,n_a1-1),2);

            % Fine grid
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, ReturnFnParamsVec, 2, 0);
            TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, special_n_semiz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, TemptationFnParamsVec, 2, 0);
            Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d1*n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
            maxindexL2a=ceil(maxindexL2/N_d1);
            maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2a/n2long);

            % L2 flag (per d2,semiz): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d1, a2prime)
            mL2d1 = shiftdim(maxindexL2d1,1);
            mL2a1 = shiftdim(maxindexL2a1,1);
            mL2a2 = shiftdim(maxindexL2a2,1);
            linidx_lower  = mL2d1                  + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind;
            linidx_upper  = mL2d1 + N_d1*(n2long-1) + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind;
            isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
            isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
            inLowerStrict = (mL2a1 >= 2)         & (mL2a1 <= n2short+1);
            inUpperStrict = (mL2a1 >= n2short+3) & (mL2a1 <= n2long-1);
            flag_ford2(:,semiz_c,d2_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            V_ford2(:,semiz_c,d2_c)=shiftdim(Vtempii,1);
            d1_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2d1,1);
            mid_ford2(:,semiz_c,d2_c)=midpoint(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind);
            L2a1_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semiz_c,d2_c)=shiftdim(maxindexL2a2,1);
        end
    end
  end

    [V_jj,d2_max]=max(V_ford2,[],3);
    V(:,:,jj)=V_jj-max(MostTempting_ford2,[],3); % subtract the most-tempting term
    Policy(2,:,:,jj)=shiftdim(d2_max,-1);
    M=N_a*N_semiz;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(1,:,:,jj)=reshape(d1_ford2(idx), [1,N_a,N_semiz]);
    Policy(3,:,:,jj)=reshape(mid_ford2(idx),[1,N_a,N_semiz]);
    Policy(4,:,:,jj)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz]);
    Policy(5,:,:,jj)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz]);
    Policy(6,:,:,jj)=reshape(flag_ford2(idx),[1,N_a,N_semiz]);
end


%% Convert Policy(3) from midpoint to lower grid point, Policy(5) from -n2short-1:1+n2short to 1:n2short+2
adjust=(Policy(5,:,:,:)<1+n2short+1);
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust;
Policy(5,:,:,:)=Policy(5,:,:,:)-(n2short+1)*(~adjust);

% Policy=Policy(1,:,:,:)+N_d1*(Policy(2,:,:,:)-1)+N_d*(Policy(3,:,:,:)-1)+N_d*N_a1*(Policy(4,:,:,:)-1)+N_d*N_a1*N_a2*(Policy(5,:,:,:)-1)+N_d*N_a1*N_a2*(n2short+2)*(Policy(6,:,:,:)-1);


end
