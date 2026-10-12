function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_raw(Vnext,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported for ValueFnIter_InfHorz_TPath_SingleStep_DC1_raw')
end

V=zeros(N_a,N_z,'gpuArray');
Policy=zeros(N_a,N_z,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
% The THIRD OUTPUT, for vfoptions.localsearch: one aprime index per (d,a,z), the window centre a
% restricted sweep needs. It has to exist for every d and not only the one this sweep chose, which no
% single max over a folded (d,aprime) dimension can give -- hence the splits below. Callers taking
% two outputs are unaffected.
aprimeReferencePolicyNew=zeros(N_d,N_a,N_z,'gpuArray');

%%
% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;


%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over z', leaving a singular second dimension
DiscountedEV=DiscountFactorParamsVec*shiftdim(EV,-1); % [1,aprime,1,z] — pre-discounted; broadcasts over d (and level1n) at every use site

if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid, a_grid(level1ii), z_gridvals, ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want aprime conditional on (d,1,a,z)
    [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    % maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2(:)+N_d*repmat((0:1:vfoptions.level1n-1)',N_z,1)+N_d*vfoptions.level1n*repelem((0:1:N_z-1)',vfoptions.level1n,1)); % aprime

    % Store
    V(level1ii,:)=shiftdim(Vtempii,2);
    Policy(level1ii,:)=shiftdim(maxindex2,2)+N_d*(reshape(maxindex1d,[vfoptions.level1n,N_z])-1); % d,aprime
    % maxindex1 is [N_d,1,level1n,N_z]: already the aprime argmax PER d. Only the max over d that
    % follows discards it, so at the level-1 points the reference is free.
    aprimeReferencePolicyNew(:,level1ii,:)=reshape(maxindex1,[N_d,vfoptions.level1n,N_z]);

    % Attempt for improved version
    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals, ReturnFnParamsVec,2);
            aprimez=aprimeindexes+N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimez,[N_d*(maxgap(ii)+1),1,N_z])); % autofill level1iidiff(ii) in 2nd dimension
            % We need aprimeReferencePolicyNew, so split the max over d&aprime into one over aprime and then a later one over d
            npts=level1ii(ii+1)-level1ii(ii)-1;
            [Vtempii_intermediate,maxindex_intermediate]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,npts,N_z]),[],2); % over aprime, per d
            [Vtempii,maxindex_final]=max(Vtempii_intermediate,[],1);                                                  % then over d
            aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:)=reshape(loweredge,[N_d,1,N_z])+reshape(maxindex_intermediate,[N_d,npts,N_z])-1;
            % Policy at the CHOSEN d: maxindex_final gives d, and the aprime offset is
            % maxindex_intermediate read AT that d, which is the one lookup to get wrong here.
            dstar=reshape(maxindex_final,[npts,N_z]);
            offsetstar=reshape(maxindex_intermediate,[N_d,npts,N_z]);
            offsetstar=offsetstar(dstar+N_d*(0:1:npts-1)'+N_d*npts*(0:1:N_z-1));
            lel=reshape(loweredge,[N_d,N_z]);
            aprimestar=lel(dstar+N_d*(0:1:N_z-1))+offsetstar-1;
            V(level1ii(ii)+1:level1ii(ii+1)-1,:)=reshape(Vtempii,[npts,N_z]);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=dstar+N_d*(aprimestar-1);
        else
            loweredge=maxindex1(:,1,ii,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals, ReturnFnParamsVec,2);
            aprimez=loweredge+N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimez,[N_d,1,N_z])); % autofill level1iidiff(ii) in 2nd dimension
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*shiftdim((0:1:N_z-1),-1))-1),1); % loweredge(given the d and z)
            % maxgap(ii)==0: the bracket is a single aprime per d, so there is nothing to max over and
            % the reference is loweredge itself, the same for every a in this interval.
            aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:)=repmat(reshape(loweredge,[N_d,1,N_z]),[1,level1ii(ii+1)-level1ii(ii)-1,1]);
        end

    end

elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV(:,:,:,z_c); % [1,aprime,1]

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

        % First, we want aprime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

        % Store
        V(level1ii,z_c)=shiftdim(Vtempii,1);
        Policy(level1ii,z_c)=shiftdim(maxindex2,1); % d,aprime
        % maxindex1 is [N_d,1,level1n] here: the aprime argmax per d at the level-1 points, free.
        aprimeReferencePolicyNew(:,level1ii,z_c)=reshape(maxindex1,[N_d,vfoptions.level1n]);

        % Attempt for improved version
        maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_z(aprimeindexes),[N_d*(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd dimension
                % We need aprimeReferencePolicyNew, so split the max over d&aprime into one over aprime and then a later one over d
                npts=length(curraindex);
                [Vtempii_intermediate,maxindex_intermediate]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,npts]),[],2); % over aprime, per d
                [Vtempii,maxindex_final]=max(Vtempii_intermediate,[],1);                                              % then over d
                aprimeReferencePolicyNew(:,curraindex,z_c)=reshape(loweredge,[N_d,1])+reshape(maxindex_intermediate,[N_d,npts])-1;
                % Policy at the CHOSEN d, reading the offset at that d
                dstar=reshape(maxindex_final,[npts,1]);
                offsetstar=reshape(maxindex_intermediate,[N_d,npts]);
                offsetstar=offsetstar(dstar+N_d*(0:1:npts-1)');
                aprimestar=loweredge(dstar)+offsetstar-1;
                V(curraindex,z_c)=reshape(Vtempii,[npts,1]);
                Policy(curraindex,z_c)=dstar+N_d*(aprimestar-1);
            else
                loweredge=maxindex1(:,1,ii);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV_z(loweredge),[N_d,1]); % autoexpand level1iidiff(ii) in 2nd dimension
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,z_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                Policy(curraindex,z_c)=shiftdim(maxindex+N_d*(loweredge(dind)'-1),1); % loweredge(given the d)
                % One aprime per d, so the reference is loweredge itself across this interval.
                aprimeReferencePolicyNew(:,curraindex,z_c)=repmat(reshape(loweredge,[N_d,1]),[1,length(curraindex)]);
            end
        end
    end
end


%% Policy in transition paths
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],Policy(:))',[length(n_d)+length(n_a),N_a,N_z]);


end
