function Fmatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec, Parallel)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

d_gridvals=CreateGridvals(n_d,d_grid,1); % 1 is matrix (rather than cell)
a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 is matrix (rather than cell)
z_gridvals=CreateGridvals(n_z,z_grid,1); % 1 is matrix (rather than cell)

if Parallel==0
    Fmatrix=zeros(N_d,N_a,N_z);
    for i1=1:N_d
        for i2=1:N_a
            for i3=1:N_z
                temp=num2cell([d_gridvals(i1,:),a_gridvals(i2,:),z_gridvals(i3,:),ReturnFnParamsVec]);
                Fmatrix(i1,i2,i3)=ReturnFn(temp{:});
            end
        end
    end
    
elseif Parallel==1
    Fmatrix=zeros(N_d,N_a,N_z);
    parfor i3=1:N_z
        z_gridvals_temp=z_gridvals(i3,:);
        Fmatrix_z=zeros(N_d,N_a);
        for i1=1:N_d
            for i2=1:N_a
                Fmatrix_z(i1,i2)=ReturnFn(d_gridvals(i1,:),a_gridvals(i2,:),z_gridvals_temp,ReturnFnParamsVec);
            end
        end
        Fmatrix(:,:,i3)=Fmatrix_z;
    end
        
% elseif Parallel==2 % FOR Parallel==2 should instead be using
% CreateReturnFnMatrix_Case2_Disc_Par2
%     disp('WARNING: CreateReturnFnMatrix_Case1_Disc does not really suppport Parallel=2 yet')
%     if n_d==0
%         aprime_dim=gpuArray.ones(prod(n_a),1,1); aprime_dim(:,1,1)=1:1:prod(n_a);
%         a_dim=gpuArray.ones(1,prod(n_a),1); a_dim(1,:,1)=1:1:prod(n_a);
%         z_dim=gpuArray.ones(1,1,prod(n_z)); z_dim(1,1,:)=1:1:prod(n_z);
%         
%         z_gridvals=zeros(N_z,length(n_z));
%         for i1=1:N_z
%             sub=zeros(1,length(n_z));
%             sub(1)=rem(i1-1,n_z(1))+1;
%             for ii=2:length(n_z)-1
%                 sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
%             end
%             sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
%             
%             if length(n_z)>1
%                 sub=sub+[0,cumsum(n_z(1:end-1))];
%             end
%             z_gridvals(i1,:)=z_grid(sub);
%         end
%         a_gridvals=zeros(N_a,length(n_a));
%         for i2=1:N_a
%             sub=zeros(1,length(n_a));
%             sub(1)=rem(i2-1,n_a(1)+1);
%             for ii=2:length(n_a)-1
%                 sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
%             end
%             sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
%             
%             if length(n_a)>1
%                 sub=sub+[0,cumsum(n_a(1:end-1))];
%             end
%             a_gridvals(i2,:)=a_grid(sub);
%         end
%         
%         ReturnFn_Par=@(i1,i2,i3) ReturnFn(a_gridvals(i1), a_gridvals(i2), z_gridvals(i3));
%         Fmatrix=arrayfun(ReturnFn_Par, aprime_dim, a_dim, z_dim);
%     else
%         d_dim=gpuArray.ones(prod(n_d),1,1,1); d_dim(:,1,1,1)=1:1:prod(n_d);
%         aprime_dim=gpuArray.ones(1,prod(n_a),1,1); aprime_dim(1,:,1,1)=1:1:prod(n_a);
%         a_dim=gpuArray.ones(1,1,prod(n_a),1); a_dim(1,1,:,1)=1:1:prod(n_a);
%         z_dim=gpuArray.ones(1,1,1,prod(n_z)); z_dim(1,1,1,:)=1:1:prod(n_z);
%         ReturnFn_Par=@(i1,i2,i3,i4) ReturnFn(ind2grid_homemade(n_d,i1,d_grid),ind2grid_homemade(n_a,i2,a_grid), ind2grid_homemade(n_a,i3,a_grid),ind2grid_homemade(n_z,i4,z_grid));
%         Fmatrix=arrayfun(ReturnFn_Par, d_dim, aprime_dim, a_dim, z_dim);
%     end
end


end


