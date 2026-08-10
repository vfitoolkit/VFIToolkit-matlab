function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC2A_GI2A_nod1_raw(n_d2, n_a, n_z, n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive QH + SemiExo + DC2A_GI2A: two-endo, divide-and-conquer on first endo + grid interpolation layer.
% Naive: Valt_j   = max u + beta*E[Valt_{j+1}]         (exponential discounter)
%        Vtilde_j = max u + beta_0*beta*E[Valt_{j+1}]  (agent's actual choice)
% SemiExo + DC2A_GI2A (two-endo grid interpolation + divide-and-conquer on first endo), no d1, with z, no e.
% bothz=[semiz,z].

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;

Valt=zeros(N_a,N_bothz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_bothz,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper
PolicyL2flagalt=2*ones(1,N_a,N_bothz,N_j,'gpuArray');
% Policy: 4 channels [d2, a1prime midpoint, a2prime, a1prime L2]
Policy=zeros(4,N_a,N_bothz,N_j,'gpuArray');
Policyalt=zeros(4,N_a,N_bothz,N_j,'gpuArray'); % exponential discounter's optimal choice

%% Split a
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity on a1
level1ii=round(linspace(1,N_a1,vfoptions.level1n));

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

%% Indexing helpers
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

%% Preallocate
V_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
V_ford2alt=zeros(N_a,N_bothz,N_d2,'gpuArray');
mid_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
mid_ford2alt=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2a1_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2a1_ford2alt=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2a2_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2a2_ford2alt=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2flag_ford2=2*ones(N_a,N_bothz,N_d2,'gpuArray');
L2flag_ford2alt=2*ones(N_a,N_bothz,N_d2,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

  if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_bothz,'gpuArray');

        % Layer 1 sparse: midpoints at level1ii
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

        % Refine between level1 points
        maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3, 0);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_jj(1,1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,1,:,ii,:,:);
                midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        % Layer 2 fine GI
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
    end

  elseif vfoptions.lowmemory==1 % loop z, vectorise semiz
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,'gpuArray');

            % Layer 1 sparse
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_valblock, ReturnFnParamsVec, 1, 0);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, z_valblock, ReturnFnParamsVec, 3, 0);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoints_jj(1,1,:,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:,:);
                    midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
                end
            end

            % Layer 2 fine GI
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 2, 0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2(:,semizblock,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,semizblock,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,z (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2(:,semizblock,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end
    end

  elseif vfoptions.lowmemory==2 % joint loop over bothz
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,'gpuArray');

            % Layer 1 sparse
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, ReturnFnParamsVec, 1, 0);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:)=maxindex1;

            maxgap=squeeze(max(max(maxindex1(1,1,:,2:end,:)-maxindex1(1,1,:,1:end-1,:),[],3),[],5));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, z_val, ReturnFnParamsVec, 3, 0);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoints_jj(1,1,:,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:);
                    midpoints_jj(1,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                end
            end

            % Layer 2 fine GI
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 2, 0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2(:,z_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,z_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,z (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end
    end
  end

    [V_jj,d2_max]=max(V_ford2,[],3);
    Valt(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_bothz;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(2,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_bothz]);
    Policy(3,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz]);
    PolicyL2flag(1,:,:,N_j)=reshape(L2flag_ford2(idx),[1,N_a,N_bothz]);

    % Terminal period: QH agent and exponential discounter coincide
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
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
        midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_bothz,'gpuArray');

        EV=V_next.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EVund=reshape(EV,[N_a1,N_a2,1,1,N_bothz]);
        EVundinterp=interp1(a1_grid,EVund,a1prime_grid);

        % Layer 1 sparse
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);

        %% Valt (beta) -- exponential discounter (also gives Policyalt)
        DiscountedEV=beta*EVund;
        DiscountedEVinterp=beta*EVundinterp;
        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

        maxgap_V=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap_V(ii));
                a1primeindexes=loweredge+shiftdim((0:1:maxgap_V(ii))',-1);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3, 1);
                aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(reshape(aprime,[(maxgap_V(ii)+1)*1,N_a2,1,N_a2,N_bothz]));
                [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
            else
                loweredge=maxindex1(1,1,:,ii,:,:);
                midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        % Layer 2 fine
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a,N_bothz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2alt(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2alt(:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
        L2a1_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2alt(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

        %% Vtilde (beta0*beta) -- the QH agent's own choice
        DiscountedEV=beta0beta*EVund;
        DiscountedEVinterp=beta0beta*EVundinterp;
        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3, 1);
                aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(reshape(aprime,[(maxgap(ii)+1)*1,N_a2,1,N_a2,N_bothz]));
                [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
            else
                loweredge=maxindex1(1,1,:,ii,:,:);
                midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        % Layer 2 fine
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a,N_bothz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

      elseif vfoptions.lowmemory==1 % loop z, vectorise semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            EV_d2z=V_next.*shiftdim(pi_bothz(semizblock,:)',-1);
            EV_d2z(isnan(EV_d2z))=0;
            EV_d2z=sum(EV_d2z,2);
            EV_d2zund=reshape(EV_d2z,[N_a1,N_a2,1,1,N_semiz]);
            EV_d2zundinterp=interp1(a1_grid,EV_d2zund,a1prime_grid);
            midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,'gpuArray');

            % Layer 1 sparse
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_valblock, ReturnFnParamsVec, 1, 0);

            %% Valt (beta) -- exponential discounter (also gives Policyalt)
            DiscountedEV_d2z=beta*EV_d2zund;
            DiscountedEV_d2zinterp=beta*EV_d2zundinterp;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV_d2z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

            maxgap_V=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap_V(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap_V(ii))',-1);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, z_valblock, ReturnFnParamsVec, 3, 1);
                    aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*semizBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap_V(ii)+1)*1,N_a2,1,N_a2,N_semiz]));
                    [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                    midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:,:);
                    midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
                end
            end

            % Layer 2 fine GI
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEV_d2zinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2alt(:,semizblock,d2_c)=shiftdim(Vtempii,1);
            mid_ford2alt(:,semizblock,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2alt(:,semizblock,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2alt(:,semizblock,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,z (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2alt(:,semizblock,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            %% Vtilde (beta0*beta) -- the QH agent's own choice
            DiscountedEV_d2z=beta0beta*EV_d2zund;
            DiscountedEV_d2zinterp=beta0beta*EV_d2zundinterp;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV_d2z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, z_valblock, ReturnFnParamsVec, 3, 1);
                    aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*semizBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap(ii)+1)*1,N_a2,1,N_a2,N_semiz]));
                    [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                    midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:,:);
                    midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
                end
            end

            % Layer 2 fine GI
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEV_d2zinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2(:,semizblock,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,semizblock,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,z (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2(:,semizblock,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end

      elseif vfoptions.lowmemory==2 % joint loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            EV_d2z=V_next.*shiftdim(pi_bothz(z_c,:)',-1);
            EV_d2z(isnan(EV_d2z))=0;
            EV_d2z=sum(EV_d2z,2);
            EV_d2zund=reshape(EV_d2z,[N_a1,N_a2]);
            EV_d2zundinterp=interp1(a1_grid,EV_d2zund,a1prime_grid);
            midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,'gpuArray');

            % Layer 1 sparse
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, ReturnFnParamsVec, 1, 0);

            %% Valt (beta) -- exponential discounter (also gives Policyalt)
            DiscountedEV_d2z=beta*EV_d2zund;
            DiscountedEV_d2zinterp=beta*EV_d2zundinterp;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV_d2z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:)=maxindex1;

            maxgap_V=squeeze(max(max(maxindex1(1,1,:,2:end,:)-maxindex1(1,1,:,1:end-1,:),[],3),[],5));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:),N_a1-maxgap_V(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap_V(ii))',-1);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, z_val, ReturnFnParamsVec, 3, 1);
                    aprime=a1primeindexes+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap_V(ii)+1)*1,N_a2,1,N_a2]));
                    [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                    midpoints_jj(1,1,:,curraindex,:)=shiftdim(maxindex,-1)+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:);
                    midpoints_jj(1,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                end
            end

            % Layer 2 fine GI
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEV_d2zinterp(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2alt(:,z_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2alt(:,z_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2alt(:,z_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2alt(:,z_c,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,z (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2alt(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            %% Vtilde (beta0*beta) -- the QH agent's own choice
            DiscountedEV_d2z=beta0beta*EV_d2zund;
            DiscountedEV_d2zinterp=beta0beta*EV_d2zundinterp;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV_d2z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:)=maxindex1;

            maxgap=squeeze(max(max(maxindex1(1,1,:,2:end,:)-maxindex1(1,1,:,1:end-1,:),[],3),[],5));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, z_val, ReturnFnParamsVec, 3, 1);
                    aprime=a1primeindexes+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap(ii)+1)*1,N_a2,1,N_a2]));
                    [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                    midpoints_jj(1,1,:,curraindex,:)=shiftdim(maxindex,-1)+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:);
                    midpoints_jj(1,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                end
            end

            % Layer 2 fine GI
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEV_d2zinterp(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2(:,z_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,z_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,z (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end
      end
    end

    [Valt_jj,d2_maxalt]=max(V_ford2alt,[],3);
    Valt(:,:,N_j)=Valt_jj;
    Policyalt(1,:,:,N_j)=shiftdim(d2_maxalt,-1);
    M=N_a*N_bothz;
    d2_maxalt_lin=reshape(d2_maxalt,[M,1]);
    idxalt=(1:M)'+M*(d2_maxalt_lin-1);
    Policyalt(2,:,:,N_j)=reshape(mid_ford2alt(idxalt), [1,N_a,N_bothz]);
    Policyalt(3,:,:,N_j)=reshape(L2a2_ford2alt(idxalt),[1,N_a,N_bothz]);
    Policyalt(4,:,:,N_j)=reshape(L2a1_ford2alt(idxalt),[1,N_a,N_bothz]);
    PolicyL2flagalt(1,:,:,N_j)=reshape(L2flag_ford2alt(idxalt),[1,N_a,N_bothz]);

    [V_jj,d2_max]=max(V_ford2,[],3);
    Vtilde(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_bothz;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(2,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_bothz]);
    Policy(3,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz]);
    PolicyL2flag(1,:,:,N_j)=reshape(L2flag_ford2(idx),[1,N_a,N_bothz]);
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

    V_next=Valt(:,:,jj+1);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

      if vfoptions.lowmemory==0
        midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_bothz,'gpuArray');

        EV=V_next.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EVund=reshape(EV,[N_a1,N_a2,1,1,N_bothz]);
        EVundinterp=interp1(a1_grid,EVund,a1prime_grid);

        % Layer 1 sparse
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1, 0);

        %% Valt (beta) -- exponential discounter (also gives Policyalt)
        DiscountedEV=beta*EVund;
        DiscountedEVinterp=beta*EVundinterp;
        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

        maxgap_V=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap_V(ii));
                a1primeindexes=loweredge+shiftdim((0:1:maxgap_V(ii))',-1);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3, 1);
                aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(reshape(aprime,[(maxgap_V(ii)+1)*1,N_a2,1,N_a2,N_bothz]));
                [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
            else
                loweredge=maxindex1(1,1,:,ii,:,:);
                midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a,N_bothz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2alt(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2alt(:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
        L2a1_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2alt(:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2alt(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

        %% Vtilde (beta0*beta) -- the QH agent's own choice
        DiscountedEV=beta0beta*EVund;
        DiscountedEVinterp=beta0beta*EVundinterp;
        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3, 1);
                aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*zBind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(reshape(aprime,[(maxgap(ii)+1)*1,N_a2,1,N_a2,N_bothz]));
                [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
            else
                loweredge=maxindex1(1,1,:,ii,:,:);
                midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*zBind;
        entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a,N_bothz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*zind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*zind;
        isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

      elseif vfoptions.lowmemory==1 % loop z, vectorise semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,jj);
            EV_d2z=V_next.*shiftdim(pi_bothz(semizblock,:)',-1);
            EV_d2z(isnan(EV_d2z))=0;
            EV_d2z=sum(EV_d2z,2);
            EV_d2zund=reshape(EV_d2z,[N_a1,N_a2,1,1,N_semiz]);
            EV_d2zundinterp=interp1(a1_grid,EV_d2zund,a1prime_grid);
            midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,'gpuArray');

            % Layer 1 sparse
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_valblock, ReturnFnParamsVec, 1, 0);

            %% Valt (beta) -- exponential discounter (also gives Policyalt)
            DiscountedEV_d2z=beta*EV_d2zund;
            DiscountedEV_d2zinterp=beta*EV_d2zundinterp;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV_d2z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

            maxgap_V=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap_V(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap_V(ii))',-1);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, z_valblock, ReturnFnParamsVec, 3, 1);
                    aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*semizBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap_V(ii)+1)*1,N_a2,1,N_a2,N_semiz]));
                    [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                    midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:,:);
                    midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
                end
            end

            % Layer 2 fine GI
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEV_d2zinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2alt(:,semizblock,d2_c)=shiftdim(Vtempii,1);
            mid_ford2alt(:,semizblock,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2alt(:,semizblock,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2alt(:,semizblock,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,z (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2alt(:,semizblock,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

            %% Vtilde (beta0*beta) -- the QH agent's own choice
            DiscountedEV_d2z=beta0beta*EV_d2zund;
            DiscountedEV_d2zinterp=beta0beta*EV_d2zundinterp;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV_d2z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, z_valblock, ReturnFnParamsVec, 3, 1);
                    aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*semizBind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap(ii)+1)*1,N_a2,1,N_a2,N_semiz]));
                    [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                    midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:,:);
                    midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
                end
            end

            % Layer 2 fine GI
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, [n_semiz,special_n_z], d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_valblock, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEV_d2zinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2(:,semizblock,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,semizblock,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,semizblock,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,z (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2(:,semizblock,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end

      elseif vfoptions.lowmemory==2 % joint loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,jj);
            EV_d2z=V_next.*shiftdim(pi_bothz(z_c,:)',-1);
            EV_d2z(isnan(EV_d2z))=0;
            EV_d2z=sum(EV_d2z,2);
            EV_d2zund=reshape(EV_d2z,[N_a1,N_a2]);
            EV_d2zundinterp=interp1(a1_grid,EV_d2zund,a1prime_grid);
            midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,'gpuArray');

            % Layer 1 sparse
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_val, ReturnFnParamsVec, 1, 0);

            %% Valt (beta) -- exponential discounter (also gives Policyalt)
            DiscountedEV_d2z=beta*EV_d2zund;
            DiscountedEV_d2zinterp=beta*EV_d2zundinterp;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV_d2z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:)=maxindex1;

            maxgap_V=squeeze(max(max(maxindex1(1,1,:,2:end,:)-maxindex1(1,1,:,1:end-1,:),[],3),[],5));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:),N_a1-maxgap_V(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap_V(ii))',-1);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, z_val, ReturnFnParamsVec, 3, 1);
                    aprime=a1primeindexes+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap_V(ii)+1)*1,N_a2,1,N_a2]));
                    [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                    midpoints_jj(1,1,:,curraindex,:)=shiftdim(maxindex,-1)+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:);
                    midpoints_jj(1,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                end
            end

            % Layer 2 fine GI
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEV_d2zinterp(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2alt(:,z_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2alt(:,z_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2alt(:,z_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2alt(:,z_c,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,z (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2alt(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            %% Vtilde (beta0*beta) -- the QH agent's own choice
            DiscountedEV_d2z=beta0beta*EV_d2zund;
            DiscountedEV_d2zinterp=beta0beta*EV_d2zundinterp;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV_d2z,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:)=maxindex1;

            maxgap=squeeze(max(max(maxindex1(1,1,:,2:end,:)-maxindex1(1,1,:,1:end-1,:),[],3),[],5));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, z_val, ReturnFnParamsVec, 3, 1);
                    aprime=a1primeindexes+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_d2z(reshape(aprime,[(maxgap(ii)+1)*1,N_a2,1,N_a2]));
                    [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                    midpoints_jj(1,1,:,curraindex,:)=shiftdim(maxindex,-1)+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:);
                    midpoints_jj(1,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                end
            end

            % Layer 2 fine GI
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, special_n_bothz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(DiscountedEV_d2zinterp(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2(:,z_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,z_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind);
            L2a1_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,z_c,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,z (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower = (ReturnMatrix_ii_dc(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_dc(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end
      end
    end

    [Valt_jj,d2_maxalt]=max(V_ford2alt,[],3);
    Valt(:,:,jj)=Valt_jj;
    Policyalt(1,:,:,jj)=shiftdim(d2_maxalt,-1);
    M=N_a*N_bothz;
    d2_maxalt_lin=reshape(d2_maxalt,[M,1]);
    idxalt=(1:M)'+M*(d2_maxalt_lin-1);
    Policyalt(2,:,:,jj)=reshape(mid_ford2alt(idxalt), [1,N_a,N_bothz]);
    Policyalt(3,:,:,jj)=reshape(L2a2_ford2alt(idxalt),[1,N_a,N_bothz]);
    Policyalt(4,:,:,jj)=reshape(L2a1_ford2alt(idxalt),[1,N_a,N_bothz]);
    PolicyL2flagalt(1,:,:,jj)=reshape(L2flag_ford2alt(idxalt),[1,N_a,N_bothz]);

    [V_jj,d2_max]=max(V_ford2,[],3);
    Vtilde(:,:,jj)=V_jj;
    Policy(1,:,:,jj)=shiftdim(d2_max,-1);
    M=N_a*N_bothz;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(2,:,:,jj)=reshape(mid_ford2(idx), [1,N_a,N_bothz]);
    Policy(3,:,:,jj)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz]);
    Policy(4,:,:,jj)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz]);
    PolicyL2flag(1,:,:,jj)=reshape(L2flag_ford2(idx),[1,N_a,N_bothz]);
end


%% Convert Policy(2) from midpoint to lower grid point, Policy(4) from -n2short-1:1+n2short to 1:n2short+2
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

adjustalt=(Policyalt(4,:,:,:)<1+n2short+1);
Policyalt(2,:,:,:)=Policyalt(2,:,:,:)-adjustalt;
Policyalt(4,:,:,:)=adjustalt.*Policyalt(4,:,:,:)+(1-adjustalt).*(Policyalt(4,:,:,:)-n2short-1);

Policyalt=[Policyalt; PolicyL2flagalt];

% Policy=Policy(1,:,:,:)+N_d2*(Policy(2,:,:,:)-1)+N_d2*N_a1*(Policy(3,:,:,:)-1)+N_d2*N_a1*N_a2*(Policy(4,:,:,:)-1)+N_d2*N_a1*N_a2*(n2short+2)*(PolicyL2flag-1);


end
