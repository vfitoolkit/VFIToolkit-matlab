function [V,Policy]=ValueFnIter_FHorz_SemiExo_DC2A_GI2A_nod1_noz_raw(n_d2, n_a, n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% SemiExo + DC2A_GI2A: two-endo, divide-and-conquer on first endo + grid interpolation layer.
% Combines SemiExo d2/pi_semiz iteration with DC2A_GI2A machinery (DC sparse search on a1, then GI fine layer).

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper
% Policy: 4 channels [d2, a1prime midpoint, a2prime, a1prime L2]
Policy=zeros(4,N_a,N_semiz,N_j,'gpuArray');

%% Split a into a1 and a2
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity (divide-and-conquer) on a1
level1ii=round(linspace(1,N_a1,vfoptions.level1n));

% Grid interpolation on a1
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

%% Precompute indexing helpers (treating semiz like z)
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);     % singleton on first dim
semizind =shiftdim(gpuArray(0:1:N_semiz-1),-1);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-4);
a12ind=gpuArray(0:1:N_a1*N_a2-1);

special_n_d2=ones(1,length(n_d2));

%% Preallocate per-d2 storage
V_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
mid_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
L2a1_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
L2a2_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
L2flag_ford2=2*ones(N_a,N_semiz,N_d2,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No continuation. Per-d2 loop (d2 still in ReturnFn).
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,'gpuArray');

        % Layer 1 sparse (level1ii): get coarse midpoints
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2); % max over a1prime
        % maxindex1 shape: [1, 1, N_a2, level1n, N_a2, N_semiz]
        midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

        % Refine between level1 points using maxgap
        maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                aprimeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_jj(1,1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,1,:,ii,:,:);
                midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        % Now have midpoints for all a1. Layer 2 (fine GI).
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
    end

    % Take max over d2_c
    [V_jj,d2_max]=max(V_ford2,[],3);
    V(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(d2_max,-1);
    d2_max_lin=reshape(d2_max,[N_a*N_semiz,1]);
    idx=(1:N_a*N_semiz)'+(N_a*N_semiz)*(d2_max_lin-1);
    Policy(2,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz]);
    PolicyL2flag(1,:,:,N_j)=reshape(L2flag_ford2(idx),[1,N_a,N_semiz]);
else
    % Using V_Jplus1
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j));
    V_next=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_semiz=pi_semiz_J(:,:,d2_c,N_j);
        midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,'gpuArray');

        % EV given d2_c
        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);

        % Layer 1 sparse
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                aprimeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3, 1);
                aprime=aprimeindexes+N_a1*a2ind+N_a1*N_a2*semizBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[(maxgap(ii)+1)*1,N_a2,1,N_a2,N_semiz]));
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
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
    end

    [V_jj,d2_max]=max(V_ford2,[],3);
    V(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(d2_max,-1);
    d2_max_lin=reshape(d2_max,[N_a*N_semiz,1]);
    idx=(1:N_a*N_semiz)'+(N_a*N_semiz)*(d2_max_lin-1);
    Policy(2,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz]);
    PolicyL2flag(1,:,:,N_j)=reshape(L2flag_ford2(idx),[1,N_a,N_semiz]);
end

%% Backward iteration
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));

    V_next=V(:,:,jj+1);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_semiz=pi_semiz_J(:,:,d2_c,jj);
        midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,'gpuArray');

        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);

        % Layer 1 sparse
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1, 0);
        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                aprimeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1_grid(aprimeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3, 1);
                aprime=aprimeindexes+N_a1*a2ind+N_a1*N_a2*semizBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[(maxgap(ii)+1)*1,N_a2,1,N_a2,N_semiz]));
                [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
            else
                loweredge=maxindex1(1,1,:,ii,:,:);
                midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d2, n_semiz, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
    end

    [V_jj,d2_max]=max(V_ford2,[],3);
    V(:,:,jj)=V_jj;
    Policy(1,:,:,jj)=shiftdim(d2_max,-1);
    d2_max_lin=reshape(d2_max,[N_a*N_semiz,1]);
    idx=(1:N_a*N_semiz)'+(N_a*N_semiz)*(d2_max_lin-1);
    Policy(2,:,:,jj)=reshape(mid_ford2(idx), [1,N_a,N_semiz]);
    Policy(3,:,:,jj)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz]);
    Policy(4,:,:,jj)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz]);
    PolicyL2flag(1,:,:,jj)=reshape(L2flag_ford2(idx),[1,N_a,N_semiz]);
end


%% Convert Policy(2) from midpoint to lower grid point, Policy(4) from -n2short-1:1+n2short to 1:n2short+2
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];

% Policy=Policy(1,:,:,:)+N_d2*(Policy(2,:,:,:)-1)+N_d2*N_a1*(Policy(3,:,:,:)-1)+N_d2*N_a1*N_a2*(Policy(4,:,:,:)-1)+N_d2*N_a1*N_a2*(n2short+2)*(PolicyL2flag-1);


end
