function Fmatrix=CreateReturnToExitFnMatrix_Case1_Disc(ReturnFn, n_a, n_z, a_grid, z_grid,Parallel,ReturnFnParamsVec)
% Is same as 'CreateReturnFnMatrix' codes, but only on (a,z)

ParamCell=cell(length(ReturnFnParamsVec),1);
for ii=1:length(ReturnFnParamsVec)
    ParamCell(ii,1)={ReturnFnParamsVec(ii)};
end

N_a=prod(n_a);
N_z=prod(n_z);

l_a=length(n_a);
l_z=length(n_z);

a_gridvals=CreateGridvals(n_a,a_grid,1);
if all(size(z_grid)==[sum(n_z),1])
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
elseif all(size(z_grid)==[prod(n_z),l_z])
    z_gridvals=z_grid;
end

Fmatrix=zeros(N_a,N_z);

if Parallel==0
    for i1=1:N_a
        for i2=1:N_z
            tempcell=num2cell([a_gridvals(i1,:),z_gridvals(i2,:)]);
            Fmatrix(i1,i2)=ReturnFn(tempcell{:},ParamCell{:});
        end
    end      
elseif Parallel==1
    parfor i2=1:N_z
        z_gridvals_i2=z_gridvals(i2,:);
        Fmatrix_z=zeros(N_a,1);
        for i1=1:N_a
            tempcell=num2cell([a_gridvals(i1,:),z_gridvals_i2]);
            Fmatrix_z(i1)=ReturnFn(tempcell{:},ParamCell{:});
        end
        Fmatrix(:,i2)=Fmatrix_z;
    end
end

end


