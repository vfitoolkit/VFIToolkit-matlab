function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC2A_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_a3,n_e,n_u,N_j, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, e_gridvals_J, u_grid, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAsset_DC1_nod1_noz_e_raw.
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No d1, no z, with e.
%
% a1: standard endogenous state, this is the one divide-and-conquer is applied to
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% With no z, e is the only shock, so it occupies the single shock slot of the
% CreateReturnFnMatrix_ExpAsset_Disc_DC2A builder (same as the DC1 version does).
%
% The EV pipeline is unchanged from the DC1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_e=prod(n_e);
N_u=prod(n_u);

N_a12=N_a1*N_a2; % the two standard assets, carried forward directly

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_e,N_j,'gpuArray'); % (1)=d2, (2)=d3, (3)=a1prime, (4)=a2prime
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d3_gridvals=CreateGridvals(n_d3,d3_grid,1);

if vfoptions.lowmemory==0
    eBind=shiftdim(gpuArray(0:1:N_e-1),-1); % [1,1,N_e]
elseif vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
end

% Setup for DC (over a1 only)
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Precompute
a2ind=gpuArray(0:N_a2-1)';
a3ind=gpuArray(0:N_a3-1)';
d3col=(1:1:N_d3)';                     % [N_d3,1]
a2pcol=reshape(0:1:N_a2-1,[1,1,N_a2]); % [1,1,N_a2prime]

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        % [N_d3, N_a1prime, N_a2prime, level1n, N_a2, N_a3, N_e]
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d3*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        d3part     =rem(maxindex2-1,N_d3)+1;
        a1primepart=rem(floor((maxindex2-1)/N_d3),N_a1)+1;
        a2primepart=floor((maxindex2-1)/(N_d3*N_a1))+1;
        V(curraindex,:,N_j)       =shiftdim(Vtempii,1);
        Policy(2,curraindex,:,N_j)=d3part;
        Policy(3,curraindex,:,N_j)=a1primepart;
        Policy(4,curraindex,:,N_j)=a2primepart;

        % Divide-and-conquer layer 2
        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % [N_d3,1,N_a2prime,1,N_a2,N_a3,N_e]
                a1primeindexes=loweredge+(0:1:maxgap(ii));                % [N_d3,maxgap+1,N_a2prime,1,N_a2,N_a3,N_e]
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, n_e, d3_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d3part     =rem(maxindex-1,N_d3)+1;
                a1localind =rem(floor((maxindex-1)/N_d3),maxgap(ii)+1)+1;
                a2primepart=floor((maxindex-1)/(N_d3*(maxgap(ii)+1)))+1;
                loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat + N_d3*N_a2*N_a2*N_a3*eBind;
                a1primepart=a1localind+loweredge(loweredge_idx)-1;
                Policy(2,curraindex,:,N_j)=d3part;
                Policy(3,curraindex,:,N_j)=a1primepart;
                Policy(4,curraindex,:,N_j)=a2primepart;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, n_e, d3_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d3part     =rem(maxindex-1,N_d3)+1;
                a2primepart=floor((maxindex-1)/N_d3)+1;
                loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat + N_d3*N_a2*N_a2*N_a3*eBind;
                Policy(2,curraindex,:,N_j)=d3part;
                Policy(3,curraindex,:,N_j)=loweredge(loweredge_idx);
                Policy(4,curraindex,:,N_j)=a2primepart;
            end
        end

    elseif vfoptions.lowmemory>=1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d3*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);

            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            d3part     =rem(maxindex2-1,N_d3)+1;
            a1primepart=rem(floor((maxindex2-1)/N_d3),N_a1)+1;
            a2primepart=floor((maxindex2-1)/(N_d3*N_a1))+1;
            V(curraindex,e_c,N_j)       =shiftdim(Vtempii,1);
            Policy(2,curraindex,e_c,N_j)=d3part;
            Policy(3,curraindex,e_c,N_j)=a1primepart;
            Policy(4,curraindex,e_c,N_j)=a2primepart;

            % Divide-and-conquer layer 2
            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % [N_d3,1,N_a2prime,1,N_a2,N_a3]
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_e, d3_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    d3part     =rem(maxindex-1,N_d3)+1;
                    a1localind =rem(floor((maxindex-1)/N_d3),maxgap(ii)+1)+1;
                    a2primepart=floor((maxindex-1)/(N_d3*(maxgap(ii)+1)))+1;
                    loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat;
                    a1primepart=a1localind+loweredge(loweredge_idx)-1;
                    Policy(2,curraindex,e_c,N_j)=d3part;
                    Policy(3,curraindex,e_c,N_j)=a1primepart;
                    Policy(4,curraindex,e_c,N_j)=a2primepart;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_e, d3_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    d3part     =rem(maxindex-1,N_d3)+1;
                    a2primepart=floor((maxindex-1)/N_d3)+1;
                    loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat;
                    Policy(2,curraindex,e_c,N_j)=d3part;
                    Policy(3,curraindex,e_c,N_j)=loweredge(loweredge_idx);
                    Policy(4,curraindex,e_c,N_j)=a2primepart;
                end
            end
        end
    end

    % d2, which was not in ReturnFn
    Policy(1,:,:,N_j)=ones(1,N_a,N_e,'gpuArray'); % d2 (terminal: d2 doesn't matter since it's only in the expectations term)

else % V_Jplus1

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);
    EV=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j+1),-1),2); % [N_a,1]
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a3primeProbs,N_a12,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a12,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a12,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a12]),[],1);
    EV=reshape(EV,[N_d3*N_a12,1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2]);

    % DiscountedEV: (d3,a1prime,a2prime), broadcast against the (a1,a2,a3,e) dims of ReturnMatrix_ii
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,N_a2]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d3*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        V(curraindex,:,N_j)=shiftdim(Vtempii,1);
        d3part     =rem(maxindex2-1,N_d3)+1;
        a1primepart=rem(floor((maxindex2-1)/N_d3),N_a1)+1;
        a2primepart=floor((maxindex2-1)/(N_d3*N_a1))+1;
        Policy(2,curraindex,:,N_j)=d3part;
        Policy(3,curraindex,:,N_j)=a1primepart;
        Policy(4,curraindex,:,N_j)=a2primepart;
        % Get the d2Policy
        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
        Policy(1,curraindex,:,N_j)=d2index_resh(lin);

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, n_e, d3_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d3aprime=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprime),[N_d3*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d3part     =rem(maxindex-1,N_d3)+1;
                a1localind =rem(floor((maxindex-1)/N_d3),maxgap(ii)+1)+1;
                a2primepart=floor((maxindex-1)/(N_d3*(maxgap(ii)+1)))+1;
                loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat + N_d3*N_a2*N_a2*N_a3*eBind;
                a1primepart=a1localind+loweredge(loweredge_idx)-1;
                Policy(2,curraindex,:,N_j)=d3part;
                Policy(3,curraindex,:,N_j)=a1primepart;
                Policy(4,curraindex,:,N_j)=a2primepart;
                % Get the d2Policy
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                Policy(1,curraindex,:,N_j)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, n_e, d3_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d3aprime=d3col + N_d3*(loweredge-1) + N_d3*N_a1*a2pcol;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprime),[N_d3*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d3part     =rem(maxindex-1,N_d3)+1;
                a2primepart=floor((maxindex-1)/N_d3)+1;
                loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat + N_d3*N_a2*N_a2*N_a3*eBind;
                a1primepart=loweredge(loweredge_idx);
                Policy(2,curraindex,:,N_j)=d3part;
                Policy(3,curraindex,:,N_j)=a1primepart;
                Policy(4,curraindex,:,N_j)=a2primepart;
                % Get the d2Policy
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                Policy(1,curraindex,:,N_j)=d2index_resh(lin);
            end
        end

    elseif vfoptions.lowmemory>=1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV;

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d3*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
            d3part     =rem(maxindex2-1,N_d3)+1;
            a1primepart=rem(floor((maxindex2-1)/N_d3),N_a1)+1;
            a2primepart=floor((maxindex2-1)/(N_d3*N_a1))+1;
            Policy(2,curraindex,e_c,N_j)=d3part;
            Policy(3,curraindex,e_c,N_j)=a1primepart;
            Policy(4,curraindex,e_c,N_j)=a2primepart;
            % Get the d2Policy
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
            Policy(1,curraindex,e_c,N_j)=d2index_resh(lin);

            % Divide and conquer layer 2
            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_e, d3_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d3aprime=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEV(d3aprime),[N_d3*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    d3part     =rem(maxindex-1,N_d3)+1;
                    a1localind =rem(floor((maxindex-1)/N_d3),maxgap(ii)+1)+1;
                    a2primepart=floor((maxindex-1)/(N_d3*(maxgap(ii)+1)))+1;
                    loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat;
                    a1primepart=a1localind+loweredge(loweredge_idx)-1;
                    Policy(2,curraindex,e_c,N_j)=d3part;
                    Policy(3,curraindex,e_c,N_j)=a1primepart;
                    Policy(4,curraindex,e_c,N_j)=a2primepart;
                    % Get the d2Policy
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                    Policy(1,curraindex,e_c,N_j)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_e, d3_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d3aprime=d3col + N_d3*(loweredge-1) + N_d3*N_a1*a2pcol;
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEV(d3aprime),[N_d3*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    d3part     =rem(maxindex-1,N_d3)+1;
                    a2primepart=floor((maxindex-1)/N_d3)+1;
                    loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat;
                    a1primepart=loweredge(loweredge_idx);
                    Policy(2,curraindex,e_c,N_j)=d3part;
                    Policy(3,curraindex,e_c,N_j)=a1primepart;
                    Policy(4,curraindex,e_c,N_j)=a2primepart;
                    % Get the d2Policy
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                    Policy(1,curraindex,e_c,N_j)=d2index_resh(lin);
                end
            end
        end
    end
end

%% Iterate backwards
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj));

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states
    EV=sum(V(:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-1),2);
    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:))==EV(aprimeplus1Index(:)));
    aprimeProbs=repmat(a3primeProbs,N_a12,1);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)),[N_d23*N_a12,N_u]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)),[N_d23*N_a12,N_u]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);

    % Refine d2 out of EV before combining with ReturnFn
    [EV,d2index]=max(reshape(EV,[N_d2,N_d3*N_a12]),[],1);
    EV=reshape(EV,[N_d3*N_a12,1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d3,N_a1,N_a2]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d3*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        V(curraindex,:,jj)=shiftdim(Vtempii,1);
        d3part     =rem(maxindex2-1,N_d3)+1;
        a1primepart=rem(floor((maxindex2-1)/N_d3),N_a1)+1;
        a2primepart=floor((maxindex2-1)/(N_d3*N_a1))+1;
        Policy(2,curraindex,:,jj)=d3part;
        Policy(3,curraindex,:,jj)=a1primepart;
        Policy(4,curraindex,:,jj)=a2primepart;
        % Get the d2Policy
        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
        Policy(1,curraindex,:,jj)=d2index_resh(lin);

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, n_e, d3_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d3aprime=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprime),[N_d3*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                d3part     =rem(maxindex-1,N_d3)+1;
                a1localind =rem(floor((maxindex-1)/N_d3),maxgap(ii)+1)+1;
                a2primepart=floor((maxindex-1)/(N_d3*(maxgap(ii)+1)))+1;
                loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat + N_d3*N_a2*N_a2*N_a3*eBind;
                a1primepart=a1localind+loweredge(loweredge_idx)-1;
                Policy(2,curraindex,:,jj)=d3part;
                Policy(3,curraindex,:,jj)=a1primepart;
                Policy(4,curraindex,:,jj)=a2primepart;
                % Get the d2Policy
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                Policy(1,curraindex,:,jj)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, n_e, d3_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d3aprime=d3col + N_d3*(loweredge-1) + N_d3*N_a1*a2pcol;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprime),[N_d3*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                d3part     =rem(maxindex-1,N_d3)+1;
                a2primepart=floor((maxindex-1)/N_d3)+1;
                loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat + N_d3*N_a2*N_a2*N_a3*eBind;
                a1primepart=loweredge(loweredge_idx);
                Policy(2,curraindex,:,jj)=d3part;
                Policy(3,curraindex,:,jj)=a1primepart;
                Policy(4,curraindex,:,jj)=a2primepart;
                % Get the d2Policy
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                Policy(1,curraindex,:,jj)=d2index_resh(lin);
            end
        end

    elseif vfoptions.lowmemory>=1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_e, d3_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV;

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d3*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
            d3part     =rem(maxindex2-1,N_d3)+1;
            a1primepart=rem(floor((maxindex2-1)/N_d3),N_a1)+1;
            a2primepart=floor((maxindex2-1)/(N_d3*N_a1))+1;
            Policy(2,curraindex,e_c,jj)=d3part;
            Policy(3,curraindex,e_c,jj)=a1primepart;
            Policy(4,curraindex,e_c,jj)=a2primepart;
            % Get the d2Policy
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
            Policy(1,curraindex,e_c,jj)=d2index_resh(lin);

            % Divide and conquer layer 2
            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_e, d3_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d3aprime=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEV(d3aprime),[N_d3*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    d3part     =rem(maxindex-1,N_d3)+1;
                    a1localind =rem(floor((maxindex-1)/N_d3),maxgap(ii)+1)+1;
                    a2primepart=floor((maxindex-1)/(N_d3*(maxgap(ii)+1)))+1;
                    loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat;
                    a1primepart=a1localind+loweredge(loweredge_idx)-1;
                    Policy(2,curraindex,e_c,jj)=d3part;
                    Policy(3,curraindex,e_c,jj)=a1primepart;
                    Policy(4,curraindex,e_c,jj)=a2primepart;
                    % Get the d2Policy
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                    Policy(1,curraindex,e_c,jj)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d3, n_a2, n_a3, special_n_e, d3_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d3aprime=d3col + N_d3*(loweredge-1) + N_d3*N_a1*a2pcol;
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEV(d3aprime),[N_d3*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    d3part     =rem(maxindex-1,N_d3)+1;
                    a2primepart=floor((maxindex-1)/N_d3)+1;
                    loweredge_idx=d3part + N_d3*(a2primepart-1) + N_d3*N_a2*a2ind_flat + N_d3*N_a2*N_a2*a3ind_flat;
                    a1primepart=loweredge(loweredge_idx);
                    Policy(2,curraindex,e_c,jj)=d3part;
                    Policy(3,curraindex,e_c,jj)=a1primepart;
                    Policy(4,curraindex,e_c,jj)=a2primepart;
                    % Get the d2Policy
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                    Policy(1,curraindex,e_c,jj)=d2index_resh(lin);
                end
            end
        end
    end
end


end
