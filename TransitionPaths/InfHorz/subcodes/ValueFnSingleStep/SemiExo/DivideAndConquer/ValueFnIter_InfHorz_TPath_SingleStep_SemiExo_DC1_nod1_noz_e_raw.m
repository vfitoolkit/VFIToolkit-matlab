function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_DC1_nod1_noz_e_raw(Vnext,n_d2,n_a,n_semiz,n_e, d2_gridvals, a_grid, semiz_gridvals, e_gridvals, pi_semiz, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, divide-and-conquer (one endogenous state), with semi-exogenous state semiz, no d1, with d2 (d2 determines the transition of semiz), no z, with iid e.
% Vnext is [N_a,N_semiz,N_e] (next period value fn)
% pi_semiz is [N_semiz,N_semizprime,N_d2] (this period's transition); pi_e is [N_e,1], the distribution of next period e.
% Policy is returned as kron rows [d2; aprime], [2,N_a,N_semiz,N_e]
% lowmemory: 0 is everything at once; 1 loops over e.

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % [1,1,N_semiz], already includes -1

% Preallocate
V_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the condl expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
% First e', then semiz' (given d2)
EV=reshape(Vnext,[N_a,N_semiz,N_e]).*reshape(pi_e,[1,1,N_e]); % [N_a,N_semiz,N_e], aprime-by-semizprime-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_semiz]
EV=EV.*permute(pi_semiz,[4,2,1,3]); % [N_a,N_semizprime,N_semiz,N_d2]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over semiz', [N_a,1,N_semiz,N_d2]
DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a,N_semiz,N_d2]); % aprime-by-semiz-by-d2

%%
if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        DiscountedEV_d2=reshape(DiscountedEV(:,:,d2_c),[N_a,1,N_semiz]); % broadcasts over level1n and e; linear index of (aprime,1,semiz) is aprime+N_a*(semiz-1)

        % n-Monotonicity
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals, e_gridvals, ReturnFnParamsVec,4); % [N_a,level1n,N_semiz,N_e], aprime-by-a-by-semiz-by-e

        entireRHS_ii=ReturnMatrix_d2ii+DiscountedEV_d2;

        % First, we want aprime conditional on (1,a,semiz,e)
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1); % [1,level1n,N_semiz,N_e]

        % Store
        V_ford2(level1ii,:,:,d2_c)=reshape(Vtempii,[vfoptions.level1n,N_semiz,N_e]);
        Policy_ford2(level1ii,:,:,d2_c)=reshape(maxindex1,[vfoptions.level1n,N_semiz,N_e]); % aprime

        % Second level based on monotonicity
        maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3)); % max over semiz,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            npts=length(curraindex);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1-by-n_semiz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, reshape(a_grid(aprimeindexes),[maxgap(ii)+1,1,N_semiz,N_e]), a_grid(curraindex), semiz_gridvals, e_gridvals, ReturnFnParamsVec,5); % [maxgap(ii)+1,npts,N_semiz,N_e]
                aprimez=aprimeindexes+N_a*semizind; % [maxgap(ii)+1,1,N_semiz,N_e], index into DiscountedEV_d2
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_d2(aprimez),[maxgap(ii)+1,1,N_semiz,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1); % [1,npts,N_semiz,N_e]
                V_ford2(curraindex,:,:,d2_c)=reshape(Vtempii,[npts,N_semiz,N_e]);
                Policy_ford2(curraindex,:,:,d2_c)=reshape(maxindex+(loweredge-1),[npts,N_semiz,N_e]); % no d1
            else
                loweredge=maxindex1(1,ii,:,:);
                % loweredge is 1-by-1-by-n_semiz-by-n_e
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, reshape(a_grid(loweredge),[1,1,N_semiz,N_e]), a_grid(curraindex), semiz_gridvals, e_gridvals, ReturnFnParamsVec,5); % [1,npts,N_semiz,N_e]
                aprimez=loweredge+N_a*semizind; % [1,1,N_semiz,N_e], index into DiscountedEV_d2
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_d2(aprimez),[1,1,N_semiz,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                V_ford2(curraindex,:,:,d2_c)=reshape(entireRHS_ii,[npts,N_semiz,N_e]);
                Policy_ford2(curraindex,:,:,d2_c)=repmat(reshape(loweredge,[1,N_semiz,N_e]),npts,1,1); % no d1
            end
        end
    end

elseif vfoptions.lowmemory==1
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        DiscountedEV_d2=reshape(DiscountedEV(:,:,d2_c),[N_a,1,N_semiz]); % broadcasts over level1n; linear index of (aprime,1,semiz) is aprime+N_a*(semiz-1)

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);

            % n-Monotonicity
            ReturnMatrix_d2iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals, e_val, ReturnFnParamsVec,4); % [N_a,level1n,N_semiz], aprime-by-a-by-semiz

            entireRHS_ii=ReturnMatrix_d2iie+DiscountedEV_d2;

            % First, we want aprime conditional on (1,a,semiz)
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1); % [1,level1n,N_semiz]

            % Store
            V_ford2(level1ii,:,e_c,d2_c)=reshape(Vtempii,[vfoptions.level1n,N_semiz]);
            Policy_ford2(level1ii,:,e_c,d2_c)=reshape(maxindex1,[vfoptions.level1n,N_semiz]); % aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3)); % max over semiz
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                npts=length(curraindex);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, reshape(a_grid(aprimeindexes),[maxgap(ii)+1,1,N_semiz]), a_grid(curraindex), semiz_gridvals, e_val, ReturnFnParamsVec,5); % [maxgap(ii)+1,npts,N_semiz]
                    aprimez=aprimeindexes+N_a*semizind; % [maxgap(ii)+1,1,N_semiz], index into DiscountedEV_d2
                    entireRHS_ii=ReturnMatrix_iie+reshape(DiscountedEV_d2(aprimez),[maxgap(ii)+1,1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1); % [1,npts,N_semiz]
                    V_ford2(curraindex,:,e_c,d2_c)=reshape(Vtempii,[npts,N_semiz]);
                    Policy_ford2(curraindex,:,e_c,d2_c)=reshape(maxindex+(loweredge-1),[npts,N_semiz]); % no d1
                else
                    loweredge=maxindex1(1,ii,:);
                    % loweredge is 1-by-1-by-n_semiz
                    % Just use aprime(ii) for everything
                    ReturnMatrix_iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, reshape(a_grid(loweredge),[1,1,N_semiz]), a_grid(curraindex), semiz_gridvals, e_val, ReturnFnParamsVec,5); % [1,npts,N_semiz]
                    aprimez=loweredge+N_a*semizind; % [1,1,N_semiz], index into DiscountedEV_d2
                    entireRHS_ii=ReturnMatrix_iie+reshape(DiscountedEV_d2(aprimez),[1,1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    V_ford2(curraindex,:,e_c,d2_c)=reshape(entireRHS_ii,[npts,N_semiz]);
                    Policy_ford2(curraindex,:,e_c,d2_c)=repmat(reshape(loweredge,[1,N_semiz]),npts,1); % no d1
                end
            end
        end
    end
end

% Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
[V,maxindex]=max(V_ford2,[],4); % max over d2, [N_a,N_semiz,N_e]
aprime_ind=reshape(Policy_ford2((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(reshape(maxindex,[N_a*N_semiz*N_e,1])-1)),[1,N_a,N_semiz,N_e]);
Policy=[reshape(maxindex,[1,N_a,N_semiz,N_e]); aprime_ind]; % [d2; aprime]


end
