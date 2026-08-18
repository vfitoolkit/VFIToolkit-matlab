function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC2A_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, e_gridvals_J, u_grid, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAsset_DC1_e_raw.
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% With d1, with z and e.
%
% a1: standard endogenous state, this is the one divide-and-conquer is applied to
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% d1 and d3 enter the ReturnFn, so the return matrix is over the joint d13 (d1 fastest).
% DiscountedEV only depends on d3, so it is repelem-ed up to d13 before being added.
%
% The EV pipeline is unchanged from the DC1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);
N_e=prod(n_e);
N_u=prod(n_u);

N_a12=N_a1*N_a2; % the two standard assets, carried forward directly

% For ReturnFn (d1 and d3)
n_d13=[n_d1,n_d3];
N_d13=N_d1*N_d3;
d13_grid=[d1_grid; d3_grid];
% For aprimeFn (d2 and d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_z,N_e,N_j,'gpuArray'); % (1)=d1, (2)=d2, (3)=d3, (4)=a1prime, (5)=a2prime
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

if vfoptions.lowmemory==0
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1); % [1,1,N_z]
    eBind=shiftdim(gpuArray(0:1:N_e-1),-2); % [1,1,1,N_e]
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1);
elseif vfoptions.lowmemory==2
    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
end

% Setup for DC (over a1 only)
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Precompute
a2ind=gpuArray(0:N_a2-1)';
a3ind=gpuArray(0:N_a3-1)';
d3col=repelem((1:1:N_d3)',N_d1,1);     % [N_d13,1]; maps full d13-index to d3-component
a2pcol=reshape(0:1:N_a2-1,[1,1,N_a2]); % [1,1,N_a2prime]

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        % [N_d13, N_a1prime, N_a2prime, level1n, N_a2, N_a3, N_z, N_e]
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d13*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        dind       =rem(maxindex2-1,N_d13)+1;
        a1primepart=rem(floor((maxindex2-1)/N_d13),N_a1)+1;
        a2primepart=floor((maxindex2-1)/(N_d13*N_a1))+1;
        d1part     =rem(dind-1,N_d1)+1;
        d3part     =ceil(dind/N_d1);
        V(curraindex,:,:,N_j)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,:,N_j)=d1part;
        Policy(3,curraindex,:,:,N_j)=d3part;
        Policy(4,curraindex,:,:,N_j)=a1primepart;
        Policy(5,curraindex,:,:,N_j)=a2primepart;

        % Divide-and-conquer layer 2
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii)); % [N_d13,1,N_a2prime,1,N_a2,N_a3,N_z,N_e]
                a1primeindexes=loweredge+(0:1:maxgap(ii));                  % [N_d13,maxgap+1,N_a2prime,1,N_a2,N_a3,N_z,N_e]
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, n_e, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind       =rem(maxindex-1,N_d13)+1;
                a1localind =rem(floor((maxindex-1)/N_d13),maxgap(ii)+1)+1;
                a2primepart=floor((maxindex-1)/(N_d13*(maxgap(ii)+1)))+1;
                loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind + N_d13*N_a2*N_a2*N_a3*N_z*eBind;
                a1primepart=a1localind+loweredge(loweredge_idx)-1;
                d1part     =rem(dind-1,N_d1)+1;
                d3part     =ceil(dind/N_d1);
                Policy(1,curraindex,:,:,N_j)=d1part;
                Policy(3,curraindex,:,:,N_j)=d3part;
                Policy(4,curraindex,:,:,N_j)=a1primepart;
                Policy(5,curraindex,:,:,N_j)=a2primepart;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, n_e, d13_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind       =rem(maxindex-1,N_d13)+1;
                a2primepart=floor((maxindex-1)/N_d13)+1;
                loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind + N_d13*N_a2*N_a2*N_a3*N_z*eBind;
                d1part     =rem(dind-1,N_d1)+1;
                d3part     =ceil(dind/N_d1);
                Policy(1,curraindex,:,:,N_j)=d1part;
                Policy(3,curraindex,:,:,N_j)=d3part;
                Policy(4,curraindex,:,:,N_j)=loweredge(loweredge_idx);
                Policy(5,curraindex,:,:,N_j)=a2primepart;
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d13*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z]),[],1);

            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            dind       =rem(maxindex2-1,N_d13)+1;
            a1primepart=rem(floor((maxindex2-1)/N_d13),N_a1)+1;
            a2primepart=floor((maxindex2-1)/(N_d13*N_a1))+1;
            d1part     =rem(dind-1,N_d1)+1;
            d3part     =ceil(dind/N_d1);
            V(curraindex,:,e_c,N_j)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,:,e_c,N_j)=d1part;
            Policy(3,curraindex,:,e_c,N_j)=d3part;
            Policy(4,curraindex,:,e_c,N_j)=a1primepart;
            Policy(5,curraindex,:,e_c,N_j)=a2primepart;

            % Divide-and-conquer layer 2
            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % [N_d13,1,N_a2prime,1,N_a2,N_a3,N_z]
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, special_n_e, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind       =rem(maxindex-1,N_d13)+1;
                    a1localind =rem(floor((maxindex-1)/N_d13),maxgap(ii)+1)+1;
                    a2primepart=floor((maxindex-1)/(N_d13*(maxgap(ii)+1)))+1;
                    loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind;
                    a1primepart=a1localind+loweredge(loweredge_idx)-1;
                    d1part     =rem(dind-1,N_d1)+1;
                    d3part     =ceil(dind/N_d1);
                    Policy(1,curraindex,:,e_c,N_j)=d1part;
                    Policy(3,curraindex,:,e_c,N_j)=d3part;
                    Policy(4,curraindex,:,e_c,N_j)=a1primepart;
                    Policy(5,curraindex,:,e_c,N_j)=a2primepart;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, special_n_e, d13_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind       =rem(maxindex-1,N_d13)+1;
                    a2primepart=floor((maxindex-1)/N_d13)+1;
                    loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind;
                    d1part     =rem(dind-1,N_d1)+1;
                    d3part     =ceil(dind/N_d1);
                    Policy(1,curraindex,:,e_c,N_j)=d1part;
                    Policy(3,curraindex,:,e_c,N_j)=d3part;
                    Policy(4,curraindex,:,e_c,N_j)=loweredge(loweredge_idx);
                    Policy(5,curraindex,:,e_c,N_j)=a2primepart;
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % Layer 1
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d13*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);

                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                dind       =rem(maxindex2-1,N_d13)+1;
                a1primepart=rem(floor((maxindex2-1)/N_d13),N_a1)+1;
                a2primepart=floor((maxindex2-1)/(N_d13*N_a1))+1;
                d1part     =rem(dind-1,N_d1)+1;
                d3part     =ceil(dind/N_d1);
                V(curraindex,z_c,e_c,N_j)       =shiftdim(Vtempii,1);
                Policy(1,curraindex,z_c,e_c,N_j)=d1part;
                Policy(3,curraindex,z_c,e_c,N_j)=d3part;
                Policy(4,curraindex,z_c,e_c,N_j)=a1primepart;
                Policy(5,curraindex,z_c,e_c,N_j)=a2primepart;

                % Divide-and-conquer layer 2
                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % [N_d13,1,N_a2prime,1,N_a2,N_a3]
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind       =rem(maxindex-1,N_d13)+1;
                        a1localind =rem(floor((maxindex-1)/N_d13),maxgap(ii)+1)+1;
                        a2primepart=floor((maxindex-1)/(N_d13*(maxgap(ii)+1)))+1;
                        loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat;
                        a1primepart=a1localind+loweredge(loweredge_idx)-1;
                        d1part     =rem(dind-1,N_d1)+1;
                        d3part     =ceil(dind/N_d1);
                        Policy(1,curraindex,z_c,e_c,N_j)=d1part;
                        Policy(3,curraindex,z_c,e_c,N_j)=d3part;
                        Policy(4,curraindex,z_c,e_c,N_j)=a1primepart;
                        Policy(5,curraindex,z_c,e_c,N_j)=a2primepart;
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind       =rem(maxindex-1,N_d13)+1;
                        a2primepart=floor((maxindex-1)/N_d13)+1;
                        loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat;
                        d1part     =rem(dind-1,N_d1)+1;
                        d3part     =ceil(dind/N_d1);
                        Policy(1,curraindex,z_c,e_c,N_j)=d1part;
                        Policy(3,curraindex,z_c,e_c,N_j)=d3part;
                        Policy(4,curraindex,z_c,e_c,N_j)=loweredge(loweredge_idx);
                        Policy(5,curraindex,z_c,e_c,N_j)=a2primepart;
                    end
                end
            end
        end
    end

    % d2, which was not in ReturnFn
    Policy(2,:,:,:,N_j)=ones(1,N_a,N_z,N_e,'gpuArray'); % d2 (terminal: d2 doesn't matter since it's only in the expectations term)

else % V_Jplus1

    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));

    % Build a3primeIndex and a3primeProbs for RiskyAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    % Get EV in terms of next period endogenous states
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);
    EVnext=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j+1),-2),3);
    EV=EVnext.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);
    EV=reshape(EV,[N_a,N_z]);

    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
    aprimeProbs=repmat(a3primeProbs,N_a12,N_z);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_z]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a12,N_u,N_z]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a12,N_u,N_z]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    EV=reshape(EV,[N_d23*N_a12,N_z]);

    % Refine d2 out of EV before combining with ReturnFn
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a12,N_z]),[],1);
    EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_z]);

    % DiscountedEV: (d3,a1prime,a2prime,-,-,-,z), broadcast against the (a1,a2,a3,e) dims of ReturnMatrix_ii
    DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_z]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        dind       =rem(maxindex2-1,N_d13)+1;
        a1primepart=rem(floor((maxindex2-1)/N_d13),N_a1)+1;
        a2primepart=floor((maxindex2-1)/(N_d13*N_a1))+1;
        d1part     =rem(dind-1,N_d1)+1;
        d3part     =ceil(dind/N_d1);
        Policy(1,curraindex,:,:,N_j)=d1part;
        Policy(3,curraindex,:,:,N_j)=d3part;
        Policy(4,curraindex,:,:,N_j)=a1primepart;
        Policy(5,curraindex,:,:,N_j)=a2primepart;
        % Get the d2Policy
        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
        Policy(2,curraindex,:,:,N_j)=d2index_resh(lin);

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, n_e, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d3aprimez=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind       =rem(maxindex-1,N_d13)+1;
                a1localind =rem(floor((maxindex-1)/N_d13),maxgap(ii)+1)+1;
                a2primepart=floor((maxindex-1)/(N_d13*(maxgap(ii)+1)))+1;
                loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind + N_d13*N_a2*N_a2*N_a3*N_z*eBind;
                a1primepart=a1localind+loweredge(loweredge_idx)-1;
                d1part     =rem(dind-1,N_d1)+1;
                d3part     =ceil(dind/N_d1);
                Policy(1,curraindex,:,:,N_j)=d1part;
                Policy(3,curraindex,:,:,N_j)=d3part;
                Policy(4,curraindex,:,:,N_j)=a1primepart;
                Policy(5,curraindex,:,:,N_j)=a2primepart;
                % Get the d2Policy
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
                Policy(2,curraindex,:,:,N_j)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, n_e, d13_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d3aprimez=d3col + N_d3*(loweredge-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind       =rem(maxindex-1,N_d13)+1;
                a2primepart=floor((maxindex-1)/N_d13)+1;
                loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind + N_d13*N_a2*N_a2*N_a3*N_z*eBind;
                a1primepart=loweredge(loweredge_idx);
                d1part     =rem(dind-1,N_d1)+1;
                d3part     =ceil(dind/N_d1);
                Policy(1,curraindex,:,:,N_j)=d1part;
                Policy(3,curraindex,:,:,N_j)=d3part;
                Policy(4,curraindex,:,:,N_j)=a1primepart;
                Policy(5,curraindex,:,:,N_j)=a2primepart;
                % Get the d2Policy
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
                Policy(2,curraindex,:,:,N_j)=d2index_resh(lin);
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d13*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            dind       =rem(maxindex2-1,N_d13)+1;
            a1primepart=rem(floor((maxindex2-1)/N_d13),N_a1)+1;
            a2primepart=floor((maxindex2-1)/(N_d13*N_a1))+1;
            d1part     =rem(dind-1,N_d1)+1;
            d3part     =ceil(dind/N_d1);
            Policy(1,curraindex,:,e_c,N_j)=d1part;
            Policy(3,curraindex,:,e_c,N_j)=d3part;
            Policy(4,curraindex,:,e_c,N_j)=a1primepart;
            Policy(5,curraindex,:,e_c,N_j)=a2primepart;
            % Get the d2Policy
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
            Policy(2,curraindex,:,e_c,N_j)=d2index_resh(lin);

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
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, special_n_e, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                    d3aprimez=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind       =rem(maxindex-1,N_d13)+1;
                    a1localind =rem(floor((maxindex-1)/N_d13),maxgap(ii)+1)+1;
                    a2primepart=floor((maxindex-1)/(N_d13*(maxgap(ii)+1)))+1;
                    loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind;
                    a1primepart=a1localind+loweredge(loweredge_idx)-1;
                    d1part     =rem(dind-1,N_d1)+1;
                    d3part     =ceil(dind/N_d1);
                    Policy(1,curraindex,:,e_c,N_j)=d1part;
                    Policy(3,curraindex,:,e_c,N_j)=d3part;
                    Policy(4,curraindex,:,e_c,N_j)=a1primepart;
                    Policy(5,curraindex,:,e_c,N_j)=a2primepart;
                    % Get the d2Policy
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
                    Policy(2,curraindex,:,e_c,N_j)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, special_n_e, d13_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                    d3aprimez=d3col + N_d3*(loweredge-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEV(d3aprimez),[N_d13*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind       =rem(maxindex-1,N_d13)+1;
                    a2primepart=floor((maxindex-1)/N_d13)+1;
                    loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind;
                    a1primepart=loweredge(loweredge_idx);
                    d1part     =rem(dind-1,N_d1)+1;
                    d3part     =ceil(dind/N_d1);
                    Policy(1,curraindex,:,e_c,N_j)=d1part;
                    Policy(3,curraindex,:,e_c,N_j)=d3part;
                    Policy(4,curraindex,:,e_c,N_j)=a1primepart;
                    Policy(5,curraindex,:,e_c,N_j)=a2primepart;
                    % Get the d2Policy
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
                    Policy(2,curraindex,:,e_c,N_j)=d2index_resh(lin);
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c); % [N_d3,N_a1,N_a2]
            d2index_z=d2index_resh(:,:,:,z_c);            % [N_d3,N_a1,N_a2]
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % Layer 1
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_z,N_d1,1,1);

                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d13*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                dind       =rem(maxindex2-1,N_d13)+1;
                a1primepart=rem(floor((maxindex2-1)/N_d13),N_a1)+1;
                a2primepart=floor((maxindex2-1)/(N_d13*N_a1))+1;
                d1part     =rem(dind-1,N_d1)+1;
                d3part     =ceil(dind/N_d1);
                Policy(1,curraindex,z_c,e_c,N_j)=d1part;
                Policy(3,curraindex,z_c,e_c,N_j)=d3part;
                Policy(4,curraindex,z_c,e_c,N_j)=a1primepart;
                Policy(5,curraindex,z_c,e_c,N_j)=a2primepart;
                % Get the d2Policy
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                Policy(2,curraindex,z_c,e_c,N_j)=d2index_z(lin);

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
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                        d3aprime=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind       =rem(maxindex-1,N_d13)+1;
                        a1localind =rem(floor((maxindex-1)/N_d13),maxgap(ii)+1)+1;
                        a2primepart=floor((maxindex-1)/(N_d13*(maxgap(ii)+1)))+1;
                        loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat;
                        a1primepart=a1localind+loweredge(loweredge_idx)-1;
                        d1part     =rem(dind-1,N_d1)+1;
                        d3part     =ceil(dind/N_d1);
                        Policy(1,curraindex,z_c,e_c,N_j)=d1part;
                        Policy(3,curraindex,z_c,e_c,N_j)=d3part;
                        Policy(4,curraindex,z_c,e_c,N_j)=a1primepart;
                        Policy(5,curraindex,z_c,e_c,N_j)=a2primepart;
                        % Get the d2Policy
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                        Policy(2,curraindex,z_c,e_c,N_j)=d2index_z(lin);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                        d3aprime=d3col + N_d3*(loweredge-1) + N_d3*N_a1*a2pcol;
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_z(d3aprime),[N_d13*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind       =rem(maxindex-1,N_d13)+1;
                        a2primepart=floor((maxindex-1)/N_d13)+1;
                        loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat;
                        a1primepart=loweredge(loweredge_idx);
                        d1part     =rem(dind-1,N_d1)+1;
                        d3part     =ceil(dind/N_d1);
                        Policy(1,curraindex,z_c,e_c,N_j)=d1part;
                        Policy(3,curraindex,z_c,e_c,N_j)=d3part;
                        Policy(4,curraindex,z_c,e_c,N_j)=a1primepart;
                        Policy(5,curraindex,z_c,e_c,N_j)=a2primepart;
                        % Get the d2Policy
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                        Policy(2,curraindex,z_c,e_c,N_j)=d2index_z(lin);
                    end
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
    EVnext=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);
    EV=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);
    EV=reshape(EV,[N_a,N_z]);

    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
    aprimeProbs=repmat(a3primeProbs,N_a12,N_z);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_z]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a12,N_u,N_z]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a12,N_u,N_z]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    EV=reshape(EV,[N_d23*N_a12,N_z]);

    % Refine d2 out of EV before combining with ReturnFn
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a12,N_z]),[],1);
    EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_z]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
        dind       =rem(maxindex2-1,N_d13)+1;
        a1primepart=rem(floor((maxindex2-1)/N_d13),N_a1)+1;
        a2primepart=floor((maxindex2-1)/(N_d13*N_a1))+1;
        d1part     =rem(dind-1,N_d1)+1;
        d3part     =ceil(dind/N_d1);
        Policy(1,curraindex,:,:,jj)=d1part;
        Policy(3,curraindex,:,:,jj)=d3part;
        Policy(4,curraindex,:,:,jj)=a1primepart;
        Policy(5,curraindex,:,:,jj)=a2primepart;
        % Get the d2Policy
        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
        Policy(2,curraindex,:,:,jj)=d2index_resh(lin);

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            a2ind_flat=repmat(repelem((0:1:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:1:N_a3-1),1,level1iidiff(ii)*N_a2);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, n_e, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d3aprimez=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind       =rem(maxindex-1,N_d13)+1;
                a1localind =rem(floor((maxindex-1)/N_d13),maxgap(ii)+1)+1;
                a2primepart=floor((maxindex-1)/(N_d13*(maxgap(ii)+1)))+1;
                loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind + N_d13*N_a2*N_a2*N_a3*N_z*eBind;
                a1primepart=a1localind+loweredge(loweredge_idx)-1;
                d1part     =rem(dind-1,N_d1)+1;
                d3part     =ceil(dind/N_d1);
                Policy(1,curraindex,:,:,jj)=d1part;
                Policy(3,curraindex,:,:,jj)=d3part;
                Policy(4,curraindex,:,:,jj)=a1primepart;
                Policy(5,curraindex,:,:,jj)=a2primepart;
                % Get the d2Policy
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
                Policy(2,curraindex,:,:,jj)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, n_e, d13_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d3aprimez=d3col + N_d3*(loweredge-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d13*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind       =rem(maxindex-1,N_d13)+1;
                a2primepart=floor((maxindex-1)/N_d13)+1;
                loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind + N_d13*N_a2*N_a2*N_a3*N_z*eBind;
                a1primepart=loweredge(loweredge_idx);
                d1part     =rem(dind-1,N_d1)+1;
                d3part     =ceil(dind/N_d1);
                Policy(1,curraindex,:,:,jj)=d1part;
                Policy(3,curraindex,:,:,jj)=d3part;
                Policy(4,curraindex,:,:,jj)=a1primepart;
                Policy(5,curraindex,:,:,jj)=a2primepart;
                % Get the d2Policy
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
                Policy(2,curraindex,:,:,jj)=d2index_resh(lin);
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
            entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d13*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
            dind       =rem(maxindex2-1,N_d13)+1;
            a1primepart=rem(floor((maxindex2-1)/N_d13),N_a1)+1;
            a2primepart=floor((maxindex2-1)/(N_d13*N_a1))+1;
            d1part     =rem(dind-1,N_d1)+1;
            d3part     =ceil(dind/N_d1);
            Policy(1,curraindex,:,e_c,jj)=d1part;
            Policy(3,curraindex,:,e_c,jj)=d3part;
            Policy(4,curraindex,:,e_c,jj)=a1primepart;
            Policy(5,curraindex,:,e_c,jj)=a2primepart;
            % Get the d2Policy
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
            Policy(2,curraindex,:,e_c,jj)=d2index_resh(lin);

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
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, special_n_e, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                    d3aprimez=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    dind       =rem(maxindex-1,N_d13)+1;
                    a1localind =rem(floor((maxindex-1)/N_d13),maxgap(ii)+1)+1;
                    a2primepart=floor((maxindex-1)/(N_d13*(maxgap(ii)+1)))+1;
                    loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind;
                    a1primepart=a1localind+loweredge(loweredge_idx)-1;
                    d1part     =rem(dind-1,N_d1)+1;
                    d3part     =ceil(dind/N_d1);
                    Policy(1,curraindex,:,e_c,jj)=d1part;
                    Policy(3,curraindex,:,e_c,jj)=d3part;
                    Policy(4,curraindex,:,e_c,jj)=a1primepart;
                    Policy(5,curraindex,:,e_c,jj)=a2primepart;
                    % Get the d2Policy
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
                    Policy(2,curraindex,:,e_c,jj)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, n_z, special_n_e, d13_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                    d3aprimez=d3col + N_d3*(loweredge-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEV(d3aprimez),[N_d13*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    dind       =rem(maxindex-1,N_d13)+1;
                    a2primepart=floor((maxindex-1)/N_d13)+1;
                    loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat + N_d13*N_a2*N_a2*N_a3*zBind;
                    a1primepart=loweredge(loweredge_idx);
                    d1part     =rem(dind-1,N_d1)+1;
                    d3part     =ceil(dind/N_d1);
                    Policy(1,curraindex,:,e_c,jj)=d1part;
                    Policy(3,curraindex,:,e_c,jj)=d3part;
                    Policy(4,curraindex,:,e_c,jj)=a1primepart;
                    Policy(5,curraindex,:,e_c,jj)=a2primepart;
                    % Get the d2Policy
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1)+N_d3*N_a1*N_a2*zBind;
                    Policy(2,curraindex,:,e_c,jj)=d2index_resh(lin);
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
            d2index_z=d2index_resh(:,:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                % Layer 1
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_z,N_d1,1,1);

                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d13*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                dind       =rem(maxindex2-1,N_d13)+1;
                a1primepart=rem(floor((maxindex2-1)/N_d13),N_a1)+1;
                a2primepart=floor((maxindex2-1)/(N_d13*N_a1))+1;
                d1part     =rem(dind-1,N_d1)+1;
                d3part     =ceil(dind/N_d1);
                Policy(1,curraindex,z_c,e_c,jj)=d1part;
                Policy(3,curraindex,z_c,e_c,jj)=d3part;
                Policy(4,curraindex,z_c,e_c,jj)=a1primepart;
                Policy(5,curraindex,z_c,e_c,jj)=a2primepart;
                % Get the d2Policy
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                Policy(2,curraindex,z_c,e_c,jj)=d2index_z(lin);

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
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                        d3aprime=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        dind       =rem(maxindex-1,N_d13)+1;
                        a1localind =rem(floor((maxindex-1)/N_d13),maxgap(ii)+1)+1;
                        a2primepart=floor((maxindex-1)/(N_d13*(maxgap(ii)+1)))+1;
                        loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat;
                        a1primepart=a1localind+loweredge(loweredge_idx)-1;
                        d1part     =rem(dind-1,N_d1)+1;
                        d3part     =ceil(dind/N_d1);
                        Policy(1,curraindex,z_c,e_c,jj)=d1part;
                        Policy(3,curraindex,z_c,e_c,jj)=d3part;
                        Policy(4,curraindex,z_c,e_c,jj)=a1primepart;
                        Policy(5,curraindex,z_c,e_c,jj)=a2primepart;
                        % Get the d2Policy
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                        Policy(2,curraindex,z_c,e_c,jj)=d2index_z(lin);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                        d3aprime=d3col + N_d3*(loweredge-1) + N_d3*N_a1*a2pcol;
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_z(d3aprime),[N_d13*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        dind       =rem(maxindex-1,N_d13)+1;
                        a2primepart=floor((maxindex-1)/N_d13)+1;
                        loweredge_idx=dind + N_d13*(a2primepart-1) + N_d13*N_a2*a2ind_flat + N_d13*N_a2*N_a2*a3ind_flat;
                        a1primepart=loweredge(loweredge_idx);
                        d1part     =rem(dind-1,N_d1)+1;
                        d3part     =ceil(dind/N_d1);
                        Policy(1,curraindex,z_c,e_c,jj)=d1part;
                        Policy(3,curraindex,z_c,e_c,jj)=d3part;
                        Policy(4,curraindex,z_c,e_c,jj)=a1primepart;
                        Policy(5,curraindex,z_c,e_c,jj)=a2primepart;
                        % Get the d2Policy
                        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(a2primepart-1);
                        Policy(2,curraindex,z_c,e_c,jj)=d2index_z(lin);
                    end
                end
            end
        end
    end
end


end
