function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC2A_GI2A_nod1_noz_e_raw(n_d2, n_a, n_semiz, n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated QH + SemiExo + DC2A_GI2A: two-endo, divide-and-conquer on first endo + grid interpolation layer.
% Sophisticated: ONE maximisation, with the beta0*beta-discounted continuation built from Vunderbar_{j+1}.
%   Vhat_j      = max u + beta_0*beta*E[Vunderbar_{j+1}]   (the agent's actual choice)
%   Vunderbar_j = Vhat_j + (beta - beta_0*beta)*E[Vunderbar_{j+1}] evaluated at that choice

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

Vunderbar=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Vhat=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
% Policy: 4 channels [d2, a1prime midpoint, a2prime, a1prime L2]
Policy=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%% Split a
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

level1ii=round(linspace(1,N_a1,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

%% Indexing helpers
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);
semizind =shiftdim(gpuArray(0:1:N_semiz-1),-1);
eind =shiftdim(gpuArray(0:1:N_e-1),-2);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-4);
a12ind=gpuArray(0:1:N_a1*N_a2-1);

special_n_d2=ones(1,length(n_d2));

% lowmemory: which shocks are looped vs vectorised (spec: =1 loop e, vectorise semiz; =2 outer semiz/inner e)
if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_semiz=ones(1,length(n_semiz));
    special_n_e=ones(1,length(n_e));
end

pi_e_J=shiftdim(pi_e_J,-2); % Move e probabilities to third dimension

%% Preallocate
V_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Vunderbar_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
mid_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
L2a1_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
L2a2_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
L2flag_ford2=2*ones(N_a,N_semiz,N_e,N_d2,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')

  if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,N_e,'gpuArray');

        % Layer 1 sparse
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoints_jj(1,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(maxindex1(1,1,:,2:end,:,:,:)-maxindex1(1,1,:,1:end-1,:,:,:),[],3),[],5),[],6),[],7));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3, 0);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_jj(1,1,:,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,1,:,ii,:,:,:);
                midpoints_jj(1,1,:,curraindex,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1,1);
            end
        end

        % Layer 2 fine GI
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1);
        mid_ford2(:,:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind+N_a2*N_a*N_semiz*eind);
        L2a1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind + n2long*N_a2*N_a*N_semiz*eind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind + n2long*N_a2*N_a*N_semiz*eind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2(:,:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
    end

  elseif vfoptions.lowmemory==1 % loop e, vectorise semiz
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,'gpuArray');

            % Layer 1 sparse
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1, 0);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3, 0);
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
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2, 0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2(:,:,e_c,d2_c)=shiftdim(Vtempii,1);
            mid_ford2(:,:,e_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,e (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end
    end

  elseif vfoptions.lowmemory==2 % outer semiz / inner e
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        for semiz_c=1:N_semiz
            semiz_val=semiz_gridvals_J(semiz_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,'gpuArray');

                % Layer 1 sparse
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_semiz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_val, e_val, ReturnFnParamsVec, 1, 0);
                [~,maxindex1]=max(ReturnMatrix_ii,[],2);
                midpoints_jj(1,1,:,level1ii,:)=maxindex1;

                maxgap=squeeze(max(max(maxindex1(1,1,:,2:end,:)-maxindex1(1,1,:,1:end-1,:),[],3),[],5));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,1,:,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_semiz, special_n_e, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_val, e_val, ReturnFnParamsVec, 3, 0);
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
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_semiz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, e_val, ReturnFnParamsVec, 2, 0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                maxindexL2a1=rem(maxindexL2-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2/n2long);

                V_ford2(:,semiz_c,e_c,d2_c)=shiftdim(Vtempii,1);
                mid_ford2(:,semiz_c,e_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind);
                L2a1_ford2(:,semiz_c,e_c,d2_c)=shiftdim(maxindexL2a1,1);
                L2a2_ford2(:,semiz_c,e_c,d2_c)=shiftdim(maxindexL2a2,1);

                % L2 flag for this d2,semiz,e (no d1)
                linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                L2flag_ford2(:,semiz_c,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end
    end
  end

    [V_jj,d2_max]=max(V_ford2,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(1,:,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_semiz*N_e;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(2,:,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(L2flag_ford2(idx),[1,N_a,N_semiz,N_e]);

    % Terminal period: QH agent and exponential discounter coincide
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;
    V_next=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*pi_e_J(1,1,:,N_j+1),3);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

      if vfoptions.lowmemory==0
        midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,N_e,'gpuArray');

        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EVund=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        EVundinterp=interp1(a1_grid,EVund,a1prime_grid);

        % Layer 1 sparse
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);

        %% Single maximisation: beta0*beta-discounted continuation (the QH agent's own choice)
        DiscountedEV=beta0beta*EVund;
        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(1,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(maxindex1(1,1,:,2:end,:,:,:)-maxindex1(1,1,:,1:end-1,:,:,:),[],3),[],5),[],6),[],7));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3, 1);
                aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*semizBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2,N_semiz,N_e]));
                [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                midpoints_jj(1,1,:,curraindex,:,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
            else
                loweredge=maxindex1(1,1,:,ii,:,:,:);
                midpoints_jj(1,1,:,curraindex,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        EVfine=reshape(EVundinterp(aprime),[n2long*N_a2,N_a,N_semiz,N_e]);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1);
        linidx=reshape(maxindexL2,[1,N_a*N_semiz*N_e])+size(EVfine,1)*(0:N_a*N_semiz*N_e-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz,N_e]);
        Vunderbar_ford2(:,:,:,d2_c)=V_ford2(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy;
        mid_ford2(:,:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind+N_a2*N_a*N_semiz*eind);
        L2a1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind + n2long*N_a2*N_a*N_semiz*eind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind + n2long*N_a2*N_a*N_semiz*eind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2(:,:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

      elseif vfoptions.lowmemory==1 % loop e, vectorise semiz
        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EVund=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        EVundinterp=interp1(a1_grid,EVund,a1prime_grid);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1, 0);

            %% Single maximisation: beta0*beta-discounted continuation (the QH agent's own choice)
            DiscountedEV=beta0beta*EVund;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3, 1);
                    aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2,N_semiz]));
                    [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                    midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:,:);
                    midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
                end
            end

            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
            EVfine=reshape(EVundinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2(:,:,e_c,d2_c)=shiftdim(Vtempii,1);
            linidx=reshape(maxindexL2,[1,N_a*N_semiz])+size(EVfine,1)*(0:N_a*N_semiz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
            Vunderbar_ford2(:,:,e_c,d2_c)=V_ford2(:,:,e_c,d2_c)+(beta-beta0beta)*EV_at_policy;
            mid_ford2(:,:,e_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,e (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end

      elseif vfoptions.lowmemory==2 % outer semiz / inner e
        for semiz_c=1:N_semiz
            semiz_val=semiz_gridvals_J(semiz_c,:,N_j);
            EV_d2z=V_next.*shiftdim(pi_semiz(semiz_c,:)',-1);
            EV_d2z(isnan(EV_d2z))=0;
            EV_d2z=sum(EV_d2z,2);
            EV_d2zund=reshape(EV_d2z,[N_a1,N_a2]);
            EV_d2zundinterp=interp1(a1_grid,EV_d2zund,a1prime_grid);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,'gpuArray');

                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_semiz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_val, e_val, ReturnFnParamsVec, 1, 0);

                %% Single maximisation: beta0*beta-discounted continuation (the QH agent's own choice)
                DiscountedEV_d2z=beta0beta*EV_d2zund;
                entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV_d2z,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoints_jj(1,1,:,level1ii,:)=maxindex1;

                maxgap=squeeze(max(max(maxindex1(1,1,:,2:end,:)-maxindex1(1,1,:,1:end-1,:),[],3),[],5));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,1,:,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_semiz, special_n_e, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_val, e_val, ReturnFnParamsVec, 3, 1);
                        aprime=a1primeindexes+N_a1*a2ind;
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2]));
                        [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                        midpoints_jj(1,1,:,curraindex,:)=shiftdim(maxindex,-1)+(loweredge-1);
                    else
                        loweredge=maxindex1(1,1,:,ii,:);
                        midpoints_jj(1,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                    end
                end

                midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
                a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_semiz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, e_val, ReturnFnParamsVec, 2, 0);
                aprime=a1primeindexes+N_a1fine*a2ind;
                EVfine=reshape(EV_d2zundinterp(aprime),[n2long*N_a2,N_a]);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                maxindexL2a1=rem(maxindexL2-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2/n2long);

                V_ford2(:,semiz_c,e_c,d2_c)=shiftdim(Vtempii,1);
                linidx=reshape(maxindexL2,[1,N_a])+size(EVfine,1)*(0:N_a-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
                Vunderbar_ford2(:,semiz_c,e_c,d2_c)=V_ford2(:,semiz_c,e_c,d2_c)+(beta-beta0beta)*EV_at_policy;
                mid_ford2(:,semiz_c,e_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind);
                L2a1_ford2(:,semiz_c,e_c,d2_c)=shiftdim(maxindexL2a1,1);
                L2a2_ford2(:,semiz_c,e_c,d2_c)=shiftdim(maxindexL2a2,1);

                % L2 flag for this d2,semiz,e (no d1)
                linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                L2flag_ford2(:,semiz_c,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end
      end
    end

    M=N_a*N_semiz*N_e;

    [V_jj,d2_max]=max(V_ford2,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(1,:,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_semiz*N_e;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Vunderbar(:,:,:,N_j)=reshape(Vunderbar_ford2(idx),[N_a,N_semiz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(mid_ford2(idx), [1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(L2flag_ford2(idx),[1,N_a,N_semiz,N_e]);
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

    V_next=sum(Vunderbar(:,:,:,jj+1).*pi_e_J(1,1,:,jj+1),3);

    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        pi_semiz=pi_semiz_J(:,:,d2_c,jj);

      if vfoptions.lowmemory==0
        midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,N_e,'gpuArray');

        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EVund=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        EVundinterp=interp1(a1_grid,EVund,a1prime_grid);

        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1, 0);

        %% Single maximisation: beta0*beta-discounted continuation (the QH agent's own choice)
        DiscountedEV=beta0beta*EVund;
        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(1,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(maxindex1(1,1,:,2:end,:,:,:)-maxindex1(1,1,:,1:end-1,:,:,:),[],3),[],5),[],6),[],7));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3, 1);
                aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*semizBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2,N_semiz,N_e]));
                [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                midpoints_jj(1,1,:,curraindex,:,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
            else
                loweredge=maxindex1(1,1,:,ii,:,:,:);
                midpoints_jj(1,1,:,curraindex,:,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
        EVfine=reshape(EVundinterp(aprime),[n2long*N_a2,N_a,N_semiz,N_e]);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);

        V_ford2(:,:,:,d2_c)=shiftdim(Vtempii,1);
        linidx=reshape(maxindexL2,[1,N_a*N_semiz*N_e])+size(EVfine,1)*(0:N_a*N_semiz*N_e-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz,N_e]);
        Vunderbar_ford2(:,:,:,d2_c)=V_ford2(:,:,:,d2_c)+(beta-beta0beta)*EV_at_policy;
        mid_ford2(:,:,:,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind+N_a2*N_a*N_semiz*eind);
        L2a1_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,:,d2_c)=shiftdim(maxindexL2a2,1);

        % L2 flag for this d2 (no d1)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind + n2long*N_a2*N_a*N_semiz*eind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind + n2long*N_a2*N_a*N_semiz*eind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        L2flag_ford2(:,:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

      elseif vfoptions.lowmemory==1 % loop e, vectorise semiz
        EV=V_next.*shiftdim(pi_semiz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        EVund=reshape(EV,[N_a1,N_a2,1,1,N_semiz]);
        EVundinterp=interp1(a1_grid,EVund,a1prime_grid);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,N_semiz,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1, 0);

            %% Single maximisation: beta0*beta-discounted continuation (the QH agent's own choice)
            DiscountedEV=beta0beta*EVund;
            entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoints_jj(1,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(maxindex1(1,1,:,2:end,:,:)-maxindex1(1,1,:,1:end-1,:,:),[],3),[],5),[],6));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3, 1);
                    aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*semizBind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2,N_semiz]));
                    [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                    midpoints_jj(1,1,:,curraindex,:,:)=shiftdim(maxindex,-1)+(loweredge-1);
                else
                    loweredge=maxindex1(1,1,:,ii,:,:);
                    midpoints_jj(1,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
                end
            end

            midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 2, 0);
            aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*semizBind;
            EVfine=reshape(EVundinterp(aprime),[n2long*N_a2,N_a,N_semiz]);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);

            V_ford2(:,:,e_c,d2_c)=shiftdim(Vtempii,1);
            linidx=reshape(maxindexL2,[1,N_a*N_semiz])+size(EVfine,1)*(0:N_a*N_semiz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
            Vunderbar_ford2(:,:,e_c,d2_c)=V_ford2(:,:,e_c,d2_c)+(beta-beta0beta)*EV_at_policy;
            mid_ford2(:,:,e_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*semizind);
            L2a1_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a1,1);
            L2a2_ford2(:,:,e_c,d2_c)=shiftdim(maxindexL2a2,1);

            % L2 flag for this d2,e (no d1)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*semizind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            L2flag_ford2(:,:,e_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
        end

      elseif vfoptions.lowmemory==2 % outer semiz / inner e
        for semiz_c=1:N_semiz
            semiz_val=semiz_gridvals_J(semiz_c,:,jj);
            EV_d2z=V_next.*shiftdim(pi_semiz(semiz_c,:)',-1);
            EV_d2z(isnan(EV_d2z))=0;
            EV_d2z=sum(EV_d2z,2);
            EV_d2zund=reshape(EV_d2z,[N_a1,N_a2]);
            EV_d2zundinterp=interp1(a1_grid,EV_d2zund,a1prime_grid);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                midpoints_jj=zeros(1,1,N_a2,N_a1,N_a2,'gpuArray');

                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_semiz, special_n_e, d2_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, semiz_val, e_val, ReturnFnParamsVec, 1, 0);

                %% Single maximisation: beta0*beta-discounted continuation (the QH agent's own choice)
                DiscountedEV_d2z=beta0beta*EV_d2zund;
                entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV_d2z,-1);
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoints_jj(1,1,:,level1ii,:)=maxindex1;

                maxgap=squeeze(max(max(maxindex1(1,1,:,2:end,:)-maxindex1(1,1,:,1:end-1,:),[],3),[],5));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,1,:,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+shiftdim((0:1:maxgap(ii))',-1);
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_semiz, special_n_e, d2_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, semiz_val, e_val, ReturnFnParamsVec, 3, 1);
                        aprime=a1primeindexes+N_a1*a2ind;
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_d2z(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2]));
                        [~,maxindex]=max(entireRHS_ii,[],1); % max over a1prime
                        midpoints_jj(1,1,:,curraindex,:)=shiftdim(maxindex,-1)+(loweredge-1);
                    else
                        loweredge=maxindex1(1,1,:,ii,:);
                        midpoints_jj(1,1,:,curraindex,:)=repelem(loweredge,1,1,1,length(curraindex),1);
                    end
                end

                midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
                a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, special_n_d2, special_n_semiz, special_n_e, d2_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, semiz_val, e_val, ReturnFnParamsVec, 2, 0);
                aprime=a1primeindexes+N_a1fine*a2ind;
                EVfine=reshape(EV_d2zundinterp(aprime),[n2long*N_a2,N_a]);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                maxindexL2a1=rem(maxindexL2-1,n2long)+1;
                maxindexL2a2=ceil(maxindexL2/n2long);

                V_ford2(:,semiz_c,e_c,d2_c)=shiftdim(Vtempii,1);
                linidx=reshape(maxindexL2,[1,N_a])+size(EVfine,1)*(0:N_a-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
                Vunderbar_ford2(:,semiz_c,e_c,d2_c)=V_ford2(:,semiz_c,e_c,d2_c)+(beta-beta0beta)*EV_at_policy;
                mid_ford2(:,semiz_c,e_c,d2_c)=midpoints_jj(maxindexL2a2+N_a2*a12ind);
                L2a1_ford2(:,semiz_c,e_c,d2_c)=shiftdim(maxindexL2a1,1);
                L2a2_ford2(:,semiz_c,e_c,d2_c)=shiftdim(maxindexL2a2,1);

                % L2 flag for this d2,semiz,e (no d1)
                linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
                isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                L2flag_ford2(:,semiz_c,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end
      end
    end

    M=N_a*N_semiz*N_e;

    [V_jj,d2_max]=max(V_ford2,[],4);
    Vhat(:,:,:,jj)=V_jj;
    Policy(1,:,:,:,jj)=shiftdim(d2_max,-1);
    M=N_a*N_semiz*N_e;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Vunderbar(:,:,:,jj)=reshape(Vunderbar_ford2(idx),[N_a,N_semiz,N_e]);
    Policy(2,:,:,:,jj)=reshape(mid_ford2(idx), [1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,jj)=reshape(L2a2_ford2(idx),[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,jj)=reshape(L2a1_ford2(idx),[1,N_a,N_semiz,N_e]);
    PolicyL2flag(1,:,:,:,jj)=reshape(L2flag_ford2(idx),[1,N_a,N_semiz,N_e]);
end


%% Convert Policy(2) from midpoint to lower grid point, Policy(4) from -n2short-1:1+n2short to 1:n2short+2
adjust=(Policy(4,:,:,:,:)<1+n2short+1);
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust;
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];



% Policy=Policy(1,:,:,:,:)+N_d2*(Policy(2,:,:,:,:)-1)+N_d2*N_a1*(Policy(3,:,:,:,:)-1)+N_d2*N_a1*N_a2*(Policy(4,:,:,:,:)-1)+N_d2*N_a1*N_a2*(n2short+2)*(PolicyL2flag-1);


end
