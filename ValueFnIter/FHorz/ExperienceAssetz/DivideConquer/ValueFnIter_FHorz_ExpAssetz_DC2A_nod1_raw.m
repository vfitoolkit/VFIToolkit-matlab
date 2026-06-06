function [V,Policy]=ValueFnIter_FHorz_ExpAssetz_DC2A_nod1_raw(n_d2, n_a1, n_a2, n_a3, n_z, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% _z analog of ValueFnIter_FHorz_ExpAsset_DC2A_nod1_raw: aprimeFn depends on z.
% Reuses CreateReturnFnMatrix_ExpAsset_Disc_DC2A (ReturnFn matrix shape unchanged).
% Policy: 1=d2, 2=a1prime, 3=a2prime. a3prime via aprimeFn downstream.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(3,N_a,N_z,N_j,'gpuArray'); % channels: 1=d2, 2=a1prime, 3=a2prime

%%
if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
else
    % precompute decode index for narrow-band Level 2 (lowmemory==0)
    zind=shiftdim((0:1:N_z-1),-1); % places N_z at dim 3 of decode index
end

% n-Monotonicity over a1 (the DC dim)
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % --- Level 1 sweep ---
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        % [N_d2, N_a1, N_a2, level1n, N_a2, N_a3, N_z]

        % Best a1prime per (d2, a2prime, a1=level1, a2, a3, z) — used for narrow band
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        % [N_d2, 1, N_a2, level1n, N_a2, N_a3, N_z]

        % Joint (d2, a1prime, a2prime) max at level1 nodes
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z]),[],1);

        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                 + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        V(curraindex,:,N_j)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,N_j)=d2ind;
        Policy(2,curraindex,:,N_j)=a1pind;
        Policy(3,curraindex,:,N_j)=a2pind;

        % --- Level 2 (narrow band) ---
        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii)); % [N_d2,1,N_a2,1,N_a2,N_a3,N_z]
                a1primeindexes=loweredge+(0:1:maxgap(ii)); % [N_d2,maxgap+1,N_a2,1,N_a2,N_a3,N_z]
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                % [N_d2*(maxgap+1)*N_a2, level1iidiff*N_a2*N_a3, N_z]
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % decode (d2, a1localind, a2prime)
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                % recover a1prime from loweredge
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,N_j)=d2ind;
                Policy(2,curraindex,:,N_j)=a1prime_rec;
                Policy(3,curraindex,:,N_j)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:); % [N_d2,1,N_a2,1,N_a2,N_a3,N_z]
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                % [N_d2*1*N_a2, level1iidiff*N_a2*N_a3, N_z]
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind;
                Policy(1,curraindex,:,N_j)=d2ind;
                Policy(2,curraindex,:,N_j)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,N_j)=a2pind;
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            % --- Level 1 sweep ---
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            V(curraindex,z_c,N_j)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,z_c,N_j)=d2ind;
            Policy(2,curraindex,z_c,N_j)=a1pind;
            Policy(3,curraindex,z_c,N_j)=a2pind;

            % --- Level 2 ---
            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii)); % [N_d2,1,N_a2,1,N_a2,N_a3]
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,z_c,N_j)=d2ind;
                    Policy(2,curraindex,z_c,N_j)=a1prime_rec;
                    Policy(3,curraindex,z_c,N_j)=a2pind;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    Policy(1,curraindex,z_c,N_j)=d2ind;
                    Policy(2,curraindex,z_c,N_j)=loweredge(loweredge_idx);
                    Policy(3,curraindex,z_c,N_j)=a2pind;
                end
            end
        end
    end

else
    % vfoptions.V_Jplus1 provided
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % [N_d2, N_a3, N_z] — z-dependent

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1); % [N_d2*N_a1*N_a2, N_a3, N_z]
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_z); % [N_d2*N_a1*N_a2, N_a3, N_zcur, N_zprime]

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_z]); % (d2a1a2, a3, zcur, zprime)
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,N_j),-2); % pi[zcur,zprime] shaped [1,1,zcur,zprime]
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4)); % sum over zprime → (d2a1a2, a3, zcur)

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]);

    if vfoptions.lowmemory==0
        % --- Level 1 ---
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                 + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        V(curraindex,:,N_j)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,N_j)=d2ind;
        Policy(2,curraindex,:,N_j)=a1pind;
        Policy(3,curraindex,:,N_j)=a2pind;

        % --- Level 2 ---
        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,N_j)=d2ind;
                Policy(2,curraindex,:,N_j)=a1prime_rec;
                Policy(3,curraindex,:,N_j)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind;
                Policy(1,curraindex,:,N_j)=d2ind;
                Policy(2,curraindex,:,N_j)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,N_j)=a2pind;
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);

            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z;
            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            V(curraindex,z_c,N_j)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,z_c,N_j)=d2ind;
            Policy(2,curraindex,z_c,N_j)=a1pind;
            Policy(3,curraindex,z_c,N_j)=a2pind;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    d2aprime_z=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_z(d2aprime_z),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,z_c,N_j)=d2ind;
                    Policy(2,curraindex,z_c,N_j)=a1prime_rec;
                    Policy(3,curraindex,z_c,N_j)=a2pind;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    d2aprime_z=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_z(d2aprime_z),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    Policy(1,curraindex,z_c,N_j)=d2ind;
                    Policy(2,curraindex,z_c,N_j)=loweredge(loweredge_idx);
                    Policy(3,curraindex,z_c,N_j)=a2pind;
                end
            end
        end
    end
end


%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_z);

    Vlower=reshape(V(aprimeIndex(:),:,jj+1),    [N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    Vupper=reshape(V(aprimeplus1Index(:),:,jj+1),[N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,jj),-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                 + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        V(curraindex,:,jj)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,jj)=d2ind;
        Policy(2,curraindex,:,jj)=a1pind;
        Policy(3,curraindex,:,jj)=a2pind;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,jj)=d2ind;
                Policy(2,curraindex,:,jj)=a1prime_rec;
                Policy(3,curraindex,:,jj)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_z, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind;
                Policy(1,curraindex,:,jj)=d2ind;
                Policy(2,curraindex,:,jj)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,jj)=a2pind;
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);

            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z;
            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            V(curraindex,z_c,jj)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,z_c,jj)=d2ind;
            Policy(2,curraindex,z_c,jj)=a1pind;
            Policy(3,curraindex,z_c,jj)=a2pind;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    d2aprime_z=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_z(d2aprime_z),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,z_c,jj)=d2ind;
                    Policy(2,curraindex,z_c,jj)=a1prime_rec;
                    Policy(3,curraindex,z_c,jj)=a2pind;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, special_n_z, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    d2aprime_z=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_z(d2aprime_z),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    Policy(1,curraindex,z_c,jj)=d2ind;
                    Policy(2,curraindex,z_c,jj)=loweredge(loweredge_idx);
                    Policy(3,curraindex,z_c,jj)=a2pind;
                end
            end
        end
    end

end


end
