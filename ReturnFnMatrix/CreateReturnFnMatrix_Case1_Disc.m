function Fmatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,Parallel,ReturnFnParamsVec,Refine) % Refine is an optional input
%If there is no d variable, just input n_d=0 and d_grid=0

if ~exist('Refine','var')
    Refine=0;
end

ParamCell=cell(length(ReturnFnParamsVec),1);
for ii=1:length(ReturnFnParamsVec)
    ParamCell(ii,1)={ReturnFnParamsVec(ii)};
end


N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a=length(n_a); 
l_z=length(n_z);
if l_d>4
    error('ERROR: Using CPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a>4
    error('ERROR: Using CPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_z>5
    error('ERROR: Using CPU for the return fn does not allow for more than four of z variable (you have length(n_z)>5)')
end

a_gridvals=CreateGridvals(n_a,a_grid,1);
if all(size(z_grid)==[sum(n_z),1])
    z_gridvals=CreateGridvals(n_z,z_grid,1);
elseif all(size(z_grid)==[prod(n_z),l_z])
    z_gridvals=z_grid;
end
if N_d~=0
    d_gridvals=CreateGridvals(n_d,d_grid,1);
end

if Parallel==0
        
    if N_d==0
        Fmatrix=zeros(N_a,N_a,N_z);
        for i1=1:N_a
            for i2=1:N_a
                for i3=1:N_z
                    tempcell=num2cell([a_gridvals(i1,:),a_gridvals(i2,:),z_gridvals(i3,:)]);
                    Fmatrix(i1,i2,i3)=ReturnFn(tempcell{:},ParamCell{:});
                end
            end
        end
            
    else
        Fmatrix=zeros(N_d*N_a,N_a,N_z);
                   
        for i1=1:N_d
            for i2=1:N_a
                i1i2=i1+(i2-1)*N_d;
                for i3=1:N_a
                    for i4=1:N_z
                        tempcell=num2cell([d_gridvals(i1,:),a_gridvals(i2,:),a_gridvals(i3,:),z_gridvals(i4,:)]);
                        Fmatrix(i1i2,i3,i4)=ReturnFn(tempcell{:},ParamCell{:});
                    end
                end
            end
        end
    end
    
elseif Parallel==1
    
    if N_d==0
        Fmatrix=zeros(N_a,N_a,N_z);
        parfor i3=1:N_z
            z_gridvals_c=z_gridvals(i3,:);
            
            Fmatrix_z=zeros(N_a,N_a);
            for i1=1:N_a
                for i2=1:N_a
                    tempcell=num2cell([a_gridvals(i1,:),a_gridvals(i2,:),z_gridvals_c]);
                    Fmatrix_z(i1,i2)=ReturnFn(tempcell{:},ParamCell{:});
                end
            end
            Fmatrix(:,:,i3)=Fmatrix_z;
        end
    else        
        Fmatrix=zeros(N_d*N_a,N_a,N_z);
        parfor i4=1:N_z
            z_gridvals_c=z_gridvals(i4,:);
            
            Fmatrix_z=zeros(N_d*N_a,N_a);
            for i1=1:N_d
                for i2=1:N_a
                    for i3=1:N_a
                        tempcell=num2cell([d_gridvals(i1,:),a_gridvals(i2,:),a_gridvals(i3,:),z_gridvals_c]);
                        Fmatrix_z(i1+(i2-1)*N_d,i3)=ReturnFn(tempcell{:},ParamCell{:});
                    end
                end
            end
            Fmatrix(:,:,i4)=Fmatrix_z;
        end
    end
end

if N_d>0
    if Refine==1
        Fmatrix=reshape(Fmatrix,[N_d,N_a,N_a,N_z]); % This is the difference when using Refine
    end
end

end


