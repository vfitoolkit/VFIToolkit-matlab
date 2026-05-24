function [V,Policy]=ValueFnIter_FHorz_SemiExo_DC2A_GI2A_raw(n_d1, n_d2, n_a, n_z, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% SemiExo + DC2A_GI2A (two-endo grid interpolation + divide-and-conquer on first endo), with d1, with z, no e. bothz=[semiz,z].

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;

V=zeros(N_a,N_bothz,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_j,'gpuArray');

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

%% Combine d1 and d2 grids
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

%% Indexing helpers
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-1);
bothzind=gpuArray(0:1:N_bothz-1);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-4);
a12ind=gpuArray(0:1:N_a1*N_a2-1)';

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

%% Preallocate
V_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
d1_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
mid_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2a1_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
L2a2_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        midpoints_jj=zeros(N_d1,1,N_a2,N_a1,N_a2,N_bothz,'gpuArray');

        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3, 0);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:);
                midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
        maxindexL2a=ceil(maxindexL2/N_d1);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        d1_ford2(:,:,d2_c)=shiftdim(maxindexL2d1,1);
        mid_ford2(:,:,d2_c)=midpoints_jj(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind+N_d1*N_a2*N_a*bothzind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

    [V_jj,d2_max]=max(V_ford2,[],3);
    V(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_bothz;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(1,:,:,N_j)=reshape(d1_ford2(idx), [1,N_a,N_bothz]);
    Policy(3,:,:,N_j)=reshape(mid_ford2(idx),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz]);
    Policy(5,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz]);
else
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j));
    V_next=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
        midpoints_jj=zeros(N_d1,1,N_a2,N_a1,N_a2,N_bothz,'gpuArray');

        EV=V_next.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_bothz]);
        DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);

        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1, 0);
        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3, 0);
                aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*bothzBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d1,(maxgap(ii)+1),N_a2,1,N_a2,N_bothz]));
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:);
                midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*bothzBind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[N_d1*n2long*N_a2,N_a,N_bothz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
        maxindexL2a=ceil(maxindexL2/N_d1);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        d1_ford2(:,:,d2_c)=shiftdim(maxindexL2d1,1);
        mid_ford2(:,:,d2_c)=midpoints_jj(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind+N_d1*N_a2*N_a*bothzind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

    [V_jj,d2_max]=max(V_ford2,[],3);
    V(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(d2_max,-1);
    M=N_a*N_bothz;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(1,:,:,N_j)=reshape(d1_ford2(idx), [1,N_a,N_bothz]);
    Policy(3,:,:,N_j)=reshape(mid_ford2(idx),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz]);
    Policy(5,:,:,N_j)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz]);
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
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
        midpoints_jj=zeros(N_d1,1,N_a2,N_a1,N_a2,N_bothz,'gpuArray');

        EV=V_next.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0;
        EV=sum(EV,2);
        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_bothz]);
        DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);

        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1, 0);
        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1);
        [~,maxindex1]=max(entireRHS_ii,[],2);
        midpoints_jj(:,1,:,level1ii,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3, 0);
                aprime=a1primeindexes+N_a1*a2ind+N_a1*N_a2*bothzBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprime,[N_d1,(maxgap(ii)+1),N_a2,1,N_a2,N_bothz]));
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:);
                midpoints_jj(:,1,:,curraindex,:,:)=repelem(loweredge,1,1,1,length(curraindex),1,1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a1-1),2);
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
        aprime=a1primeindexes+N_a1fine*a2ind+N_a1fine*N_a2*bothzBind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(aprime),[N_d1*n2long*N_a2,N_a,N_bothz]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        maxindexL2d1=rem(maxindexL2-1,N_d1)+1;
        maxindexL2a=ceil(maxindexL2/N_d1);
        maxindexL2a1=rem(maxindexL2a-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2a/n2long);

        V_ford2(:,:,d2_c)=shiftdim(Vtempii,1);
        d1_ford2(:,:,d2_c)=shiftdim(maxindexL2d1,1);
        mid_ford2(:,:,d2_c)=midpoints_jj(shiftdim(maxindexL2d1,1)+N_d1*(shiftdim(maxindexL2a2,1)-1)+N_d1*N_a2*a12ind+N_d1*N_a2*N_a*bothzind);
        L2a1_ford2(:,:,d2_c)=shiftdim(maxindexL2a1,1);
        L2a2_ford2(:,:,d2_c)=shiftdim(maxindexL2a2,1);
    end

    [V_jj,d2_max]=max(V_ford2,[],3);
    V(:,:,jj)=V_jj;
    Policy(2,:,:,jj)=shiftdim(d2_max,-1);
    M=N_a*N_bothz;
    d2_max_lin=reshape(d2_max,[M,1]);
    idx=(1:M)'+M*(d2_max_lin-1);
    Policy(1,:,:,jj)=reshape(d1_ford2(idx), [1,N_a,N_bothz]);
    Policy(3,:,:,jj)=reshape(mid_ford2(idx),[1,N_a,N_bothz]);
    Policy(4,:,:,jj)=reshape(L2a2_ford2(idx),[1,N_a,N_bothz]);
    Policy(5,:,:,jj)=reshape(L2a1_ford2(idx),[1,N_a,N_bothz]);
end


%% Convert Policy(3) from midpoint to lower grid point, Policy(5) from -n2short-1:1+n2short to 1:n2short+2
adjust=(Policy(5,:,:,:)<1+n2short+1);
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust;
Policy(5,:,:,:)=adjust.*Policy(5,:,:,:)+(1-adjust).*(Policy(5,:,:,:)-n2short-1);

Policy=Policy(1,:,:,:)+N_d1*(Policy(2,:,:,:)-1)+N_d*(Policy(3,:,:,:)-1)+N_d*N_a1*(Policy(4,:,:,:)-1)+N_d*N_a1*N_a2*(Policy(5,:,:,:)-1);


end
