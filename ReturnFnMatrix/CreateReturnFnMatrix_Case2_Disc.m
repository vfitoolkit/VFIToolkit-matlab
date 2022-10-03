function Fmatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec, Parallel)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

d_gridvals=CreateGridvals(n_d,d_grid,1); % 1 is matrix (rather than cell)
a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 is matrix (rather than cell)
if all(size(z_grid)==[sum(n_z),1])
    z_gridvals=CreateGridvals(n_z,z_grid,1); % 1 is matrix (rather than cell)
elseif all(size(z_grid)==[prod(n_z),length(n_z)])
    z_gridvals=z_grid;
end

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
                temp=num2cell([d_gridvals(i1,:),a_gridvals(i2,:),z_gridvals_temp,ReturnFnParamsVec]);
                Fmatrix_z(i1,i2)=ReturnFn(temp{:});
            end
        end
        Fmatrix(:,:,i3)=Fmatrix_z;
    end
end


end


