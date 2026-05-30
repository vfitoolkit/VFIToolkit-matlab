function [V,Policy]=ValueFnIter_FHorz_SemiExo_GI2A_noz_e_raw(n_d1, n_d2, n_a, n_semiz, n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% SemiExo + GI2A (two-endo grid interpolation), with d1, no z, with e.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_semiz,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

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

%% Combine d1 and d2 grids
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

%% Indexing helpers
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);
semizind =gpuArray(0:1:N_semiz-1);
eind =shiftdim(gpuArray(0:1:N_e-1),-1);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-4);
a12ind=gpuArray(0:1:N_a1*N_a2-1)';

pi_e_J=shiftdim(pi_e_J,-2); % Move e probabilities to third dimension

%% Preallocate
V_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
d1_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
mid_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
L2a1_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
L2a2_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
flag_ford2=2*ones(N_a,N_semiz,N_e,N_d2,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);

        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
        maxindexL2a=ceil(maxindexL2/N_d1);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d1, a2prime)
        mL2d1 = shiftdim(maxindexL2d1,1);
        mL2a1 = shiftdim(maxindexL2a1,1);
        mL2a2 = shiftdim(maxindexL2a2,1);
        linidx_lower  = mL2d1                  + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind + N_d1*n2long*N_a2*N_a*N_semiz*eind;
        linidx_upper  = mL2d1 + N_d1*(n2long-1) + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind + N_d1*n2long*N_a2*N_a*N_semiz*eind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (mL2a1 >= 2)         & (mL2a1 <= n2short+1);
        inUpperStrict = (mL2a1 >= n2short+3) & (mL2a1 <= n2long-1);
        flag_ford2(:,:,:,d2_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1);
        d1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2d1,1);
        mid_ford2(:,:,:,d2_c)=midpoint(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind+N_d1*N_a2*N_a*semizind+N_d1*N_a2*N_a*N_semiz*eind);
        L2a1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

    [V_jj,d2_max]=max(V_ford2,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_semiz*N_e;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(1,:,:,:,N_j)=reshape(d1_ford2(idx), [1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(mid_ford2(idx),[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford2(idx),[1,N_a,N_semiz,N_e]);
else
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j));
    V_next=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*pi_e_J(1,1,:,N_j),3);

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);

        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d1*n2long*N_a2,N_a,N_semiz,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
        maxindexL2a=ceil(maxindexL2/N_d1);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d1, a2prime)
        mL2d1 = shiftdim(maxindexL2d1,1);
        mL2a1 = shiftdim(maxindexL2a1,1);
        mL2a2 = shiftdim(maxindexL2a2,1);
        linidx_lower  = mL2d1                  + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind + N_d1*n2long*N_a2*N_a*N_semiz*eind;
        linidx_upper  = mL2d1 + N_d1*(n2long-1) + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind + N_d1*n2long*N_a2*N_a*N_semiz*eind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (mL2a1 >= 2)         & (mL2a1 <= n2short+1);
        inUpperStrict = (mL2a1 >= n2short+3) & (mL2a1 <= n2long-1);
        flag_ford2(:,:,:,d2_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1);
        d1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2d1,1);
        mid_ford2(:,:,:,d2_c)=midpoint(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind+N_d1*N_a2*N_a*semizind+N_d1*N_a2*N_a*N_semiz*eind);
        L2a1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

    [V_jj,d2_max]=max(V_ford2,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_semiz*N_e;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(1,:,:,:,N_j)=reshape(d1_ford2(idx), [1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(mid_ford2(idx),[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford2(idx),[1,N_a,N_semiz,N_e]);
end

%% Backward iteration
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));

    V_next=sum(V(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_semiz=pi_semiz_J(:,:,d2_c,jj);

        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EV=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        EVinterp=interp1(a1_grid,EV,a1prime_grid);

        ReturnMatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1, 0);
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,n_a1-1),2);

        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d, n_semiz, n_e, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprime),[N_d1*n2long*N_a2,N_a,N_semiz,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
        maxindexL2a=ceil(maxindexL2/N_d1);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        % L2 flag (per d2): detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d1, a2prime)
        mL2d1 = shiftdim(maxindexL2d1,1);
        mL2a1 = shiftdim(maxindexL2a1,1);
        mL2a2 = shiftdim(maxindexL2a2,1);
        linidx_lower  = mL2d1                  + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind + N_d1*n2long*N_a2*N_a*N_semiz*eind;
        linidx_upper  = mL2d1 + N_d1*(n2long-1) + N_d1*n2long*(mL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind + N_d1*n2long*N_a2*N_a*N_semiz*eind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (mL2a1 >= 2)         & (mL2a1 <= n2short+1);
        inUpperStrict = (mL2a1 >= n2short+3) & (mL2a1 <= n2long-1);
        flag_ford2(:,:,:,d2_c) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1);
        d1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2d1,1);
        mid_ford2(:,:,:,d2_c)=midpoint(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind+N_d1*N_a2*N_a*semizind+N_d1*N_a2*N_a*N_semiz*eind);
        L2a1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

    [V_jj,d2_max]=max(V_ford2,[],4);
    V(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(d2_max,-1);
    M=N_a*N_semiz*N_e;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(1,:,:,:,jj)=reshape(d1_ford2(idx), [1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,jj)=reshape(mid_ford2(idx),[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,jj)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz,N_e]);
    Policy(5,:,:,:,jj)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz,N_e]);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford2(idx),[1,N_a,N_semiz,N_e]);
end


%% Convert Policy(3) from midpoint to lower grid point, Policy(5) from -n2short-1:1+n2short to 1:n2short+2
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

% Policy=Policy(1,:,:,:,:)+N_d1*(Policy(2,:,:,:,:)-1)+N_d*(Policy(3,:,:,:,:)-1)+N_d*N_a1*(Policy(4,:,:,:,:)-1)+N_d*N_a1*N_a2*(Policy(5,:,:,:,:)-1)+N_d*N_a1*N_a2*(n2short+2)*(PolicyL2flag-1);


end
