function [V,Policy2]=ValueFnIter_Case1_TPath_SingleStep_DC2_raw(Vnext,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% DC2B: two endogenous states, divide-and-conquer on the first endo state, but not on the second endo state

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

N_a1=n_a(1);
N_a2=n_a(2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

V=zeros(N_a1,N_a2,N_z,'gpuArray');
Policy=zeros(N_a1,N_a2,N_z,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_gridvals=CreateGridvals(n_d,d_grid,1);

% n-Monotonicity
% vfoptions.level1n=[7,5];
level11ii=round(linspace(1,N_a1,vfoptions.level1n(1)));
level11iidiff=level11ii(2:end)-level11ii(1:end-1)+1; % Note: For 2D this includes the end-points
level12jj=round(linspace(1,N_a2,vfoptions.level1n(2)));
level12jjdiff=level12jj(2:end)-level12jj(1:end-1)+1; % Note: For 2D this includes the end-points

%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
EV=sum(EV,2); % sum over z', leaving a singular second dimension
entireEV=repmat(shiftdim(EV,-1),N_d,1,1,1); % [d,aprime,1,z]

% disp('size entireEV')
% size(entireEV)

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12jj), z_gridvals, ReturnFnParamsVec,1);

entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV,[N_d,N_a1*N_a2,1,1,N_z]); % move a2prime into same dimension as (a1,a2), so second dimension is solely a1prime

% First, we want a1a2prime conditional on (d,1,a,z)
[~,maxindex1]=max(entireRHS_ii,[],2);

% Now, get and store the full (d,aprime)
[Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n(1),vfoptions.level1n(2),N_z]),[],1);
% Store
V(level11ii,level12jj,:)=shiftdim(Vtempii,1);
Policy(level11ii,level12jj,:)=shiftdim(maxindex2,1);

% Split maxindex1 into a1prime and a2prime
maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);
maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);

% Attempt for improved version
maxgap1=max(max(maxindex11(:,1,2:end,2:end,:)-maxindex11(:,1,1:end-1,1:end-1,:),[],5),[],1);
maxgap2=max(max(maxindex12(:,1,2:end,2:end,:)-maxindex12(:,1,1:end-1,1:end-1,:),[],5),[],1);
for ii=1:(vfoptions.level1n(1)-1)
    for jj=1:(vfoptions.level1n(2)-1)
        curra1index=(level11ii(ii):1:level11ii(ii+1)); % Note: in 2D we need to include the edges
        curra2index=(level12jj(jj):1:level12jj(jj+1)); % Note: in 2D we need to include the edges

        if maxgap1(1,1,ii,jj)>0 && maxgap2(1,1,ii,jj)>0
            loweredge1=min(maxindex11(:,1,ii,jj,:),N_a1-maxgap1(1,1,ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            loweredge2=min(maxindex12(:,1,ii,jj,:),N_a2-maxgap2(1,1,ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-1-by-n_z
            a1primeindexes=loweredge1+repmat((0:1:maxgap1(1,1,ii,jj)),1,maxgap2(1,1,ii,jj)+1);
            a2primeindexes=loweredge2+repelem((0:1:maxgap2(1,1,ii,jj)),1,maxgap1(1,1,ii,jj)+1);
            % aprime possibilities are maxgap(ii)+1-n_a2-by-1-by-n_a2-by-n_z
            ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(a1primeindexes), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repmat(a1primeindexes-1,1,1,level11iidiff(ii),1,1)+N_d*N_a1*repelem(a2primeindexes-1,1,1,1,level12jjdiff(jj),1)+N_d*N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_iijj+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*(maxgap1(1,1,ii,jj)+1)*(maxgap2(1,1,ii,jj)+1),level11iidiff(ii),level12jjdiff(jj),N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=rem(maxindex-1,N_d)+1;
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap1(1,1,ii,jj)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap1(1,1,ii,jj)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
            %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
            zind=shiftdim(0:1:N_z-1,-2); % already includes -1
            Policy(curra1index,curra2index,:)=shiftdim(maxindexfix+N_d*(loweredge1(dind+N_d*zind)-1)+N_d*N_a1*(loweredge2(dind+N_d*zind)-1),1);

        elseif maxgap1(1,1,ii,jj)>0 && maxgap2(1,1,ii,jj)==0
            loweredge1=min(maxindex11(:,1,ii,jj,:),N_a1-maxgap1(1,1,ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            loweredge2=maxindex12(:,1,ii,jj,:);
            % loweredge is 1-by-1-by-1-by-n_z
            a1primeindexes=loweredge1+(0:1:maxgap1(1,1,ii,jj));
            ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(a1primeindexes), reshape(a2_grid(loweredge2),[N_d,1,1,1,N_z]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repmat(a1primeindexes-1,1,1,level11iidiff(ii),1,1)+N_d*N_a1*repelem(loweredge2-1,1,1,1,level12jjdiff(jj),1)+N_d*N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_iijj+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*(maxgap1(1,1,ii,jj)+1)*1,level11iidiff(ii),level12jjdiff(jj),N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
            % no need to rework maxindex in this case
            dind=rem(maxindex-1,N_d)+1;
            %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
            %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
            zind=shiftdim(0:1:N_z-1,-2); % already includes -1
            Policy(curra1index,curra2index,:)=shiftdim(maxindex+N_d*(loweredge1(dind+N_d*zind)-1)+N_d*N_a1*(loweredge2(dind+N_d*zind)-1),1);

        elseif maxgap1(1,1,ii,jj)==0 && maxgap2(1,1,ii,jj)>0
            loweredge1=maxindex11(:,1,ii,jj,:);
            loweredge2=min(maxindex12(:,1,ii,jj,:),N_a2-maxgap2(1,1,ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
            a2primeindexes=loweredge2+(0:1:maxgap2(1,1,ii,jj));
            ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, reshape(a1_grid(loweredge1),[N_d,1,1,1,N_z]), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repmat(a1primeindexes-1,1,1,level11iidiff(ii),1,1)+N_d*N_a1*repelem(a2primeindexes-1,1,1,1,level12jjdiff(jj),1)+N_d*N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_iijj+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*(maxgap1(1,1,ii,jj)+1)*(maxgap2(1,1,ii,jj)+1),level11iidiff(ii),level12jjdiff(jj),N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            dind=rem(maxindex-1,N_d)+1;
            a1primeind=0;  % already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
            %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
            zind=shiftdim(0:1:N_z-1,-2); % already includes -1
            Policy(curra1index,curra2index,:)=shiftdim(maxindexfix+N_d*(loweredge1(dind+N_d*zind)-1)+N_d*N_a1*(loweredge2(dind+N_d*zind)-1),1);

        else % maxgap1(1,1,ii,jj)==0 && maxgap2(1,1,ii,jj)==0
            loweredge1=maxindex11(:,1,ii,jj,:);
            loweredge2=maxindex12(:,1,ii,jj,:);
            % loweredge is 1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
            ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, reshape(a1_grid(loweredge1),[N_d,1,1,1,N_z]), reshape(a2_grid(loweredge2),[N_d,1,1,1,N_z]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*repmat(loweredge1-1,1,1,level11iidiff(ii),1,1)+N_d*N_a1*repelem(loweredge2-1,1,1,1,level12jjdiff(jj),1)+N_d*N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_iijj+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*1*1,level11iidiff(ii),level12jjdiff(jj),N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
            % maxindex is anyway just the index for d, so
            dind=maxindex;
            %  the a1prime is relative to loweredge1(zind), need to 'add'  the loweredge1
            %  the a2prime is relative to loweredge2(zind), need to 'add'  the loweredge2
            zind=shiftdim(0:1:N_z-1,-2); % already includes -1
            Policy(curra1index,curra2index,:)=shiftdim(maxindex+N_d*(loweredge1(dind+N_d*zind)-1)+N_d*N_a1*(loweredge2(dind+N_d*zind)-1),1);

        end
    end
end

% error('STOP!')


%% Reshape output
V=reshape(V,[N_a1*N_a2,N_z]);
Policy=reshape(Policy,[N_a1*N_a2,N_z]);


% for ii=1:(vfoptions.level1n-1)
%     curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level11iidiff(ii),1);
%     if maxgap(ii)>0
%         loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(:,1,:,ii,:,:)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
%         % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
%         a1primeindexes=loweredge+(0:1:maxgap(ii));
%         % aprime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-n_z
%         ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals, ReturnFnParamsVec,2);
%         daprimez=(1:1:N_d)'+N_d*repelem(a1primeindexes-1,1,1,1,level11iidiff(ii),1,1)+N_d*N_a1*shiftdim((0:1:N_a2-1),-1)+N_d*N_a*shiftdim((0:1:N_z-1),-4); % the current aprimeii(ii):aprimeii(ii+1)
%         entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*(maxgap(ii)+1)*N_a2,level11iidiff(ii)*N_a2,N_z]));
%         [Vtempii,maxindex]=max(entireRHS_ii,[],1);
%         V(curraindex,:)=shiftdim(Vtempii,1);
%         % maxindex needs to be reworked:
%         %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
%         dind=(rem(maxindex-1,N_d)+1);
%         a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
%         a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
%         maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
%         %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
%         a2ind=repelem((0:1:N_a2-1),1,level11iidiff(ii)); % already includes -1
%         zind=shiftdim((0:1:N_z-1),-1); % already includes -1
%         allind=dind+N_d*a2primeind+N_d*N_a2*a2ind+N_d*N_a2*N_a2*zind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
%         Policy(curraindex,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
%     else
%         loweredge=maxindex1(:,1,:,ii,:,:);
%         % Just use aprime(ii) for everything
%         ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals, ReturnFnParamsVec,2);
%         daprimez=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,1,level11iidiff(ii),1,1)+N_d*1*shiftdim((0:1:N_a2-1),-1)+N_d*N_a*shiftdim((0:1:N_z-1),-4); % the current aprimeii(ii):aprimeii(ii+1)
%         entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d*1*N_a2,level11iidiff(ii)*N_a2,N_z]));
%         [Vtempii,maxindex]=max(entireRHS_ii,[],1);
%         V(curraindex,:)=shiftdim(Vtempii,1);
%         % maxindex needs to be reworked:
%         %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
%         dind=(rem(maxindex-1,N_d)+1);
%         a1primeind=0; %1-1; % already includes -1
%         a2primeind=ceil(maxindex/N_d)-1; % already includes -1 % divide by (N_d*1)
%         maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
%         %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
%         a2ind=repelem((0:1:N_a2-1),1,level11iidiff(ii)); % already includes -1
%         zind=shiftdim((0:1:N_z-1),-1); % already includes -1
%         allind=dind+N_d*a2primeind+N_d*N_a2*a2ind+N_d*N_a2*N_a2*zind; % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
%         Policy(curraindex,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
%     end
% 
% end



%%
Policy2=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1); % d
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1); % aprime

end