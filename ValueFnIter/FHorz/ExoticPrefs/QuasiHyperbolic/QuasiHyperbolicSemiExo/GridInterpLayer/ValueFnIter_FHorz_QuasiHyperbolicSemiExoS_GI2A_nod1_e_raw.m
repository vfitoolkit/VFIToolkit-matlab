function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_GI2A_nod1_e_raw(n_d2, n_a, n_z, n_semiz, n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated QH + SemiExo + GI2A (grid interpolation layer on the first endogenous
% state only, a2 enumerated in full), no d1, with z, with e.
%
% Sophisticated: Vhat_j      = max u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EVinterp_at_optimal_aprime
% Only one maximisation, and the continuation is read off the interpolated
% EV (EVfine) at the winning (d2, a1prime, a2prime).

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

Vhat=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
% Policy: 4 channels [d2, a1prime midpoint, a2prime, a1prime L2]
Policy=zeros(4,N_a,N_bothz,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

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

%% Indexing helpers
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);
zind =shiftdim(gpuArray(0:1:N_bothz-1),-1);
eind =shiftdim(gpuArray(0:1:N_e-1),-2);
zBind=shiftdim(gpuArray(0:1:N_bothz-1),-4);
a12ind=gpuArray(0:1:N_a1*N_a2-1);

special_n_d2=ones(1,length(n_d2));

% lowmemory: which shocks are looped vs vectorised (spec: =1 loop e; =2 outer z/inner e, vec semiz; =3 joint bothz/inner e)
if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
    semizind =shiftdim(gpuArray(0:1:N_semiz-1),-1); % semiz-block analogue of zind (L2)
    semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-4); % semiz-block analogue of zBind (L2)
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_e=ones(1,length(n_e));
end

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

pi_e_J=shiftdim(pi_e_J,-2); % Move e probabilities to third dimension

%% Preallocate
V_ford2=zeros(N_a,N_bothz,N_e,N_d2,'gpuArray');
Vunderbar_ford2=zeros(N_a,N_bothz,N_e,N_d2,'gpuArray');
mid_ford2=zeros(N_a,N_bothz,N_e,N_d2,'gpuArray');
L2a1_ford2=zeros(N_a,N_bothz,N_e,N_d2,'gpuArray');
L2a2_ford2=zeros(N_a,N_bothz,N_e,N_d2,'gpuArray');
flag_ford2=2*ones(N_a,N_bothz,N_e,N_d2,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

  if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);

        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_bothz*eind;
        linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_bothz*eind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        flag_ford2(:,:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        V_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,:,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind+N_a2*N_a*N_bothz*eind);
        L2a1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

  elseif vfoptions.lowmemory==1 % loop e, vectorise bothz
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1, 0);
            [~,maxindex]=max(ReturnMatrix,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2, 0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,e): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,:,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,:,e_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,:,e_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
            L2a1_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a2,1);
        end
    end

  elseif vfoptions.lowmemory==2 % outer z / inner e, vectorise semiz
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_valblock, e_val, ReturnFnParamsVec, 1, 0);
                [~,maxindex]=max(ReturnMatrix,[],2);
                midpoint=max(min(maxindex,n_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, e_val, ReturnFnParamsVec, 2, 0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                maxindexL2a1=rem(maxindexL2-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2/n2long);

                % L2 flag (per d2,z,e): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
                linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
                linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford2(:,semizblock,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                V_ford2(:,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                mid_ford2(:,semizblock,e_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
                L2a1_ford2(:,semizblock,e_c,d2_c)=shiftdim(maxindexL2a1,1);
                L2a2_ford2(:,semizblock,e_c,d2_c)=shiftdim(maxindexL2a2,1);
            end
        end
    end

  elseif vfoptions.lowmemory==3 % joint bothz / inner e
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_val, e_val, ReturnFnParamsVec, 1, 0);
                [~,maxindex]=max(ReturnMatrix,[],2);
                midpoint=max(min(maxindex,n_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, e_val, ReturnFnParamsVec, 2, 0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                maxindexL2a1=rem(maxindexL2-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2/n2long);

                % L2 flag (per d2,z,e): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
                linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford2(:,z_c,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                V_ford2(:,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                mid_ford2(:,z_c,e_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind);
                L2a1_ford2(:,z_c,e_c,d2_c)=shiftdim(maxindexL2a1,1);
                L2a2_ford2(:,z_c,e_c,d2_c)=shiftdim(maxindexL2a2,1);
            end
        end
    end
  end

    [V_jj,d2_max]=max(V_ford2,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(1,:,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_bothz*N_e;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(2,:,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford2(idx),[1,N_a,N_bothz,N_e]);

    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j); % terminal period: no continuation
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;
    V_next=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*pi_e_J(1,1,:,N_j+1),3); % integrate over e' -> [N_a, N_bothz]

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

      if vfoptions.lowmemory==0
        EV=V_next.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_bothz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);

        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        EVfine=reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_bothz,N_e]);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_bothz*eind;
        linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_bothz*eind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        flag_ford2(:,:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        V_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,:,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind+N_a2*N_a*N_bothz*eind);
        L2a1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % Vunderbar for this d2: continuation read off the interpolated EV at the chosen (a1prime,a2prime)
        linidx=reshape(maxindexL2,[1,N_a*N_bothz*N_e])+n2long*N_a2*(0:N_a*N_bothz*N_e-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_bothz,N_e]);
        Vunderbar_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;

      elseif vfoptions.lowmemory==1 % loop e, vectorise bothz
        EV=V_next.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_bothz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1, 0);
            entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
            EVfine=reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_bothz]);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,e): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,:,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,:,e_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,:,e_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
            L2a1_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a2,1);

            % Vunderbar for this (d2,e)
            linidx=reshape(maxindexL2,[1,N_a*N_bothz])+n2long*N_a2*(0:N_a*N_bothz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_bothz]);
            Vunderbar_ford2(:,:,e_c,d2_c)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;
        end

      elseif vfoptions.lowmemory==2 % outer z / inner e, vectorise semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            EV=V_next.*shiftdim(pi_bothz(semizblock,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_valblock, e_val, ReturnFnParamsVec, 1, 0);
                entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
                [~,maxindex]=max(entireRHS,[],2);
                midpoint=max(min(maxindex,n_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, e_val, ReturnFnParamsVec, 2, 0);
                aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
                EVfine=reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                maxindexL2a1=rem(maxindexL2-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2/n2long);

                % L2 flag (per d2,z,e): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
                linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
                linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford2(:,semizblock,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                V_ford2(:,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                mid_ford2(:,semizblock,e_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
                L2a1_ford2(:,semizblock,e_c,d2_c)=shiftdim(maxindexL2a1,1);
                L2a2_ford2(:,semizblock,e_c,d2_c)=shiftdim(maxindexL2a2,1);

                % Vunderbar for this (d2,semizblock,e)
                linidx=reshape(maxindexL2,[1,N_a*N_semiz])+n2long*N_a2*(0:N_a*N_semiz-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
                Vunderbar_ford2(:,semizblock,e_c,d2_c)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;
            end
        end

      elseif vfoptions.lowmemory==3 % joint bothz / inner e
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            EV=V_next.*shiftdim(pi_bothz(z_c,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_val, e_val, ReturnFnParamsVec, 1, 0);
                entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
                [~,maxindex]=max(entireRHS,[],2);
                midpoint=max(min(maxindex,n_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, e_val, ReturnFnParamsVec, 2, 0);
                aprime=a1primeindexes+N_a1fine*a2ind;
                EVfine=reshape(EVinterp(aprime),[n2long*N_a2,N_a]);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                maxindexL2a1=rem(maxindexL2-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2/n2long);

                % L2 flag (per d2,z,e): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
                linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford2(:,z_c,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                V_ford2(:,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                mid_ford2(:,z_c,e_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind);
                L2a1_ford2(:,z_c,e_c,d2_c)=shiftdim(maxindexL2a1,1);
                L2a2_ford2(:,z_c,e_c,d2_c)=shiftdim(maxindexL2a2,1);

                % Vunderbar for this (d2,z,e)
                linidx=reshape(maxindexL2,[1,N_a])+n2long*N_a2*(0:N_a-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
                Vunderbar_ford2(:,z_c,e_c,d2_c)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;
            end
        end
      end
    end

    [V_jj,d2_max]=max(V_ford2,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(1,:,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_bothz*N_e;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(2,:,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford2(idx),[1,N_a,N_bothz,N_e]);
    Vunderbar(:,:,:,N_j)=reshape(Vunderbar_ford2(idx),[N_a,N_bothz,N_e]);
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

    V_next=sum(Vunderbar(:,:,:,jj+1).*pi_e_J(1,1,:,jj+1),3); % sophisticated: continuation is Vunderbar

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

      if vfoptions.lowmemory==0
        EV=V_next.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_bothz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);

        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1, 0);
        entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        EVfine=reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_bothz,N_e]);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
        linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_bothz*eind;
        linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind + n2long*N_a2*N_a*N_bothz*eind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        flag_ford2(:,:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

        V_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,:,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind+N_a2*N_a*N_bothz*eind);
        L2a1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % Vunderbar for this d2: continuation read off the interpolated EV at the chosen (a1prime,a2prime)
        linidx=reshape(maxindexL2,[1,N_a*N_bothz*N_e])+n2long*N_a2*(0:N_a*N_bothz*N_e-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_bothz,N_e]);
        Vunderbar_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;

      elseif vfoptions.lowmemory==1 % loop e, vectorise bothz
        EV=V_next.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_bothz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1, 0);
            entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
            EVfine=reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_bothz]);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            % L2 flag (per d2,e): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
            linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
            linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford2(:,:,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            V_ford2(:,:,e_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,:,e_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
            L2a1_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a2,1);

            % Vunderbar for this (d2,e)
            linidx=reshape(maxindexL2,[1,N_a*N_bothz])+n2long*N_a2*(0:N_a*N_bothz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_bothz]);
            Vunderbar_ford2(:,:,e_c,d2_c)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;
        end

      elseif vfoptions.lowmemory==2 % outer z / inner e, vectorise semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,jj);
            EV=V_next.*shiftdim(pi_bothz(semizblock,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_valblock, e_val, ReturnFnParamsVec, 1, 0);
                entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
                [~,maxindex]=max(entireRHS,[],2);
                midpoint=max(min(maxindex,n_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, e_val, ReturnFnParamsVec, 2, 0);
                aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
                EVfine=reshape(EVinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                maxindexL2a1=rem(maxindexL2-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2/n2long);

                % L2 flag (per d2,z,e): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
                linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
                linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford2(:,semizblock,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                V_ford2(:,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                mid_ford2(:,semizblock,e_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
                L2a1_ford2(:,semizblock,e_c,d2_c)=shiftdim(maxindexL2a1,1);
                L2a2_ford2(:,semizblock,e_c,d2_c)=shiftdim(maxindexL2a2,1);

                % Vunderbar for this (d2,semizblock,e)
                linidx=reshape(maxindexL2,[1,N_a*N_semiz])+n2long*N_a2*(0:N_a*N_semiz-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
                Vunderbar_ford2(:,semizblock,e_c,d2_c)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;
            end
        end

      elseif vfoptions.lowmemory==3 % joint bothz / inner e
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,jj);
            EV=V_next.*shiftdim(pi_bothz(z_c,:)',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a1,N_a2]);
            EVinterp=interp1(a1_grid,EV,a1prime_grid);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid, a2_grid, z_val, e_val, ReturnFnParamsVec, 1, 0);
                entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
                [~,maxindex]=max(entireRHS,[],2);
                midpoint=max(min(maxindex,n_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, e_val, ReturnFnParamsVec, 2, 0);
                aprime=a1primeindexes+N_a1fine*a2ind;
                EVfine=reshape(EVinterp(aprime),[n2long*N_a2,N_a]);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                maxindexL2a1=rem(maxindexL2-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2/n2long);

                % L2 flag (per d2,z,e): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen a2prime)
                linidx_lower  = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                linidx_upper  = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford2(:,z_c,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                V_ford2(:,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                mid_ford2(:,z_c,e_c,d2_c)=midpoint(maxindexL2a2+N_a2*a12ind);
                L2a1_ford2(:,z_c,e_c,d2_c)=shiftdim(maxindexL2a1,1);
                L2a2_ford2(:,z_c,e_c,d2_c)=shiftdim(maxindexL2a2,1);

                % Vunderbar for this (d2,z,e)
                linidx=reshape(maxindexL2,[1,N_a])+n2long*N_a2*(0:N_a-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
                Vunderbar_ford2(:,z_c,e_c,d2_c)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;
            end
        end
      end
    end

    [V_jj,d2_max]=max(V_ford2,[],4);
    Vhat(:,:,:,jj)=V_jj;
    Policy(1,:,:,:,jj)=shiftdim(d2_max,-1);
    M=N_a*N_bothz*N_e;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(2,:,:,:,jj)=reshape(mid_ford2(idx), [1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,jj)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz,N_e]);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford2(idx),[1,N_a,N_bothz,N_e]);
    Vunderbar(:,:,:,jj)=reshape(Vunderbar_ford2(idx),[N_a,N_bothz,N_e]);
end


%% Convert Policy(2) from midpoint to lower grid point, Policy(4) from -n2short-1:1+n2short to 1:n2short+2
adjust=(Policy(4,:,:,:,:)<1+n2short+1);
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust;
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

% Policy=Policy(1,:,:,:,:)+N_d2*(Policy(2,:,:,:,:)-1)+N_d2*N_a1*(Policy(3,:,:,:,:)-1)+N_d2*N_a1*N_a2*(Policy(4,:,:,:,:)-1)+N_d2*N_a1*N_a2*(n2short+2)*(PolicyL2flag-1);

end
