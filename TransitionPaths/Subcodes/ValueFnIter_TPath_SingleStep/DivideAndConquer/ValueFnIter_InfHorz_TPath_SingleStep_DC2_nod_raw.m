function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_DC2_nod_raw(Vnext,n_a,n_z, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% DC2: two endogenous states, divide-and-conquer both endogenous states

N_a=prod(n_a);
N_z=prod(n_z);

N_a1=n_a(1);
N_a2=n_a(2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
% vfoptions.level1n=[7,5];
level11ii=round(linspace(1,N_a1,vfoptions.level1n(1)));
level11iidiff=level11ii(2:end)-level11ii(1:end-1)+1; % Note: For 2D this includes the end-points
level12jj=round(linspace(1,N_a2,vfoptions.level1n(2)));
level12jjdiff=level12jj(2:end)-level12jj(1:end-1)+1; % Note: For 2D this includes the end-points

V=zeros(N_a1,N_a2,N_z,'gpuArray');
Policy=zeros(N_a1,N_a2,N_z,'gpuArray'); % first dim indexes the optimal choice for aprime rest of dimensions a,z

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
EV=sum(EV,2); % sum over z', leaving a singular second dimension

DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1*N_a2,1,1,N_z]);

% n-Monotonicity
ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_nod_Par2(ReturnFn, n_z, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12jj), z_gridvals, ReturnFnParamsVec,1);

entireRHS_ii=ReturnMatrix_iijj+DiscountedEV; % autoexpand (a,z)

% Calc the max and it's index
[~,maxindex1]=max(entireRHS_ii,[],1); % Vtempii
% maxindex1 is optimal a1a2prime(a,z)

% In 2D, this all gets overwritten as each layer2-ii-jj includes the edges, so I can skip it to save time
% % Store
% V(level11ii,level12jj,:)=shiftdim(Vtempii,1);
% Policy(level11ii,level12jj,:)=shiftdim(maxindex1,1);

% Split maxindex1 into a1prime and a2prime
maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);
maxindex12=reshape(ceil(maxindex1/N_a1),[1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);

% Attempt for improved version
maxgap1=squeeze(max(maxindex11(1,2:end,2:end,:)-maxindex11(1,1:end-1,1:end-1,:),[],4));
maxgap2=squeeze(max(maxindex12(1,2:end,2:end,:)-maxindex12(1,1:end-1,1:end-1,:),[],4));
for ii=1:(vfoptions.level1n(1)-1)
    for jj=1:(vfoptions.level1n(2)-1)
        curra1index=(level11ii(ii):1:level11ii(ii+1)); % Note: in 2D we need to include the edges
        curra2index=(level12jj(jj):1:level12jj(jj+1)); % Note: in 2D we need to include the edges

        if maxgap1(ii,jj)>0 && maxgap2(ii,jj)>0
            loweredge1=min(maxindex11(1,ii,jj,:),N_a1-maxgap1(ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            loweredge2=min(maxindex12(1,ii,jj,:),N_a2-maxgap2(ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-1-by-n_z
            a1primeindexes=loweredge1+repmat((0:1:maxgap1(ii,jj))',maxgap2(ii,jj)+1,1);
            a2primeindexes=loweredge2+repelem((0:1:maxgap2(ii,jj))',maxgap1(ii,jj)+1,1);
            % aprime possibilities are maxgap(ii)+1-by-1-by-1-by-n_z
            ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_nod_Par2(ReturnFn, n_z, a1_grid(a1primeindexes), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
            aprimez=a1primeindexes+N_a1*(a2primeindexes-1)+N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_iijj+DiscountedEV(reshape(aprimez,[(maxgap1(ii,jj)+1)*(maxgap2(ii,jj)+1),1,1,N_z]));  % autofill level11iidiff(ii),level12jjdiff(jj) in 2nd and 3rd dimensions
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            a1primeind=rem(maxindex-1,maxgap1(ii,jj)+1)+1;
            a2primeind=ceil(maxindex/(maxgap1(ii,jj)+1));
            maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge1, need to 'add' the loweredge1
            %  the a2prime is relative to loweredge2, need to 'add' the loweredge2
            Policy(curra1index,curra2index,:)=shiftdim(maxindexfix+(loweredge1-1)+N_a1*(loweredge2-1),1);

        elseif maxgap1(ii,jj)>0 && maxgap2(ii,jj)==0
            loweredge1=min(maxindex11(1,ii,jj,:),N_a1-maxgap1(ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            loweredge2=maxindex12(1,ii,jj,:);
            % loweredge is 1-by-1-by-1-by-n_z
            a1primeindexes=loweredge1+(0:1:maxgap1(ii,jj))';
            ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_nod_Par2(ReturnFn, n_z, a1_grid(a1primeindexes), reshape(a2_grid(loweredge2),[1,1,1,N_z]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
            aprimez=a1primeindexes+N_a1*(loweredge2-1)+N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_iijj+DiscountedEV(reshape(aprimez,[(maxgap1(ii,jj)+1),1,1,N_z]));  % autofill level11iidiff(ii),level12jjdiff(jj) in 2nd and 3rd dimensions
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
            % no need to rework maxindex in this case
            %  the a1prime is relative to loweredge1, need to 'add' the loweredge1
            %  the a2prime is relative to loweredge2, need to 'add' the loweredge2
            Policy(curra1index,curra2index,:)=shiftdim(maxindex+(loweredge1-1)+N_a1*(loweredge2-1),1);
        elseif maxgap1(ii,jj)==0 && maxgap2(ii,jj)>0
            loweredge1=maxindex11(1,ii,jj,:);
            loweredge2=min(maxindex12(1,ii,jj,:),N_a2-maxgap2(ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
            a2primeindexes=loweredge2+(0:1:maxgap2(ii,jj))';
            ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_nod_Par2(ReturnFn, n_z, reshape(a1_grid(loweredge1),[1,1,1,N_z]), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
            aprimez=loweredge1+N_a1*(a2primeindexes-1)+N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_iijj+DiscountedEV(reshape(aprimez,[(maxgap2(ii,jj)+1),1,1,N_z]));  % autofill level11iidiff(ii),level12jjdiff(jj) in 2nd and 3rd dimensions
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            % a1primeind=1;
            % a2primeind=maxindex;
            maxindexfix=1+N_a1*(maxindex-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge1, need to 'add' the loweredge1
            %  the a2prime is relative to loweredge2, need to 'add' the loweredge2
            Policy(curra1index,curra2index,:)=shiftdim(maxindexfix+(loweredge1-1)+N_a1*(loweredge2-1),1);

        else % maxgap1(ii,jj)==0 && maxgap2(ii,jj)==0
            loweredge1=maxindex11(1,ii,jj,:);
            loweredge2=maxindex12(1,ii,jj,:);
            % loweredge is 1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
            ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_nod_Par2(ReturnFn, n_z, reshape(a1_grid(loweredge1),[1,1,1,N_z]), reshape(a2_grid(loweredge2),[1,1,1,N_z]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
            aprimez=loweredge1+N_a1*(loweredge2-1)+N_a*shiftdim((0:1:N_z-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_iijj+DiscountedEV(reshape(aprimez,[1,1,1,N_z]));  % autofill level11iidiff(ii),level12jjdiff(jj) in 2nd and 3rd dimensions
            % Can skip max() as first dimension is singular
            V(curra1index,curra2index,:)=shiftdim(entireRHS_ii,1);
            maxindexfix=ones(level11iidiff(ii),level12jjdiff(jj),N_z,'gpuArray');
            %  the a1prime is relative to loweredge1, need to 'add' the loweredge1
            %  the a2prime is relative to loweredge2, need to 'add' the loweredge2
            Policy(curra1index,curra2index,:)=maxindexfix+shiftdim((loweredge1-1)+N_a1*(loweredge2-1),1);

        end
    end
end


%% Reshape output
V=reshape(V,[N_a1*N_a2,N_z]);
Policy=reshape(Policy,[N_a1*N_a2,N_z]);

end
