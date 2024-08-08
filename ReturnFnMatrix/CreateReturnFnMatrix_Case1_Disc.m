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

    if isscalar(n_d) && isscalar(n_a) && isscalar(n_z)
        % hardcode the 1D for speed
        if N_d==0
            Fmatrix=zeros(N_a,N_a,N_z);
            for i3=1:N_z
                for i2=1:N_a
                    for i1=1:N_a
                        Fmatrix(i1,i2,i3)=ReturnFn(a_gridvals(i1),a_gridvals(i2),z_gridvals(i3),ParamCell{:});
                    end
                end
            end

        else
            % N_d
            Fmatrix=zeros(N_d*N_a,N_a,N_z);
            for i4=1:N_z
                for i3=1:N_a
                    for i1i2=1:N_d*N_a
                        i1=rem(i1i2-1,N_d)+1;
                        i2=ceil(i1i2/N_d);
                        Fmatrix(i1i2,i3,i4)=ReturnFn(d_gridvals(i1),a_gridvals(i2),a_gridvals(i3),z_gridvals(i4),ParamCell{:});
                    end
                end
            end
        end

    elseif isscalar(n_d) && isscalar(n_a) && length(n_z)==2
        % d and a one-dimensional, z is two-dimensional
        if N_d==0
            Fmatrix=zeros(N_a,N_a,N_z);
            for i3i4=1:N_z
                for i2=1:N_a
                    for i1=1:N_a
                        Fmatrix(i1,i2,i3i4)=ReturnFn(a_gridvals(i1),a_gridvals(i2),z_gridvals(i3i4,1),z_gridvals(i3i4,2),ParamCell{:});
                    end
                end
            end

        else
            % N_d
            Fmatrix=zeros(N_d*N_a,N_a,N_z);
            for i3i4=1:N_z
                for i3=1:N_a % a today
                    for i1i2=1:N_d*N_a % (d,a')
                        i1=rem(i1i2-1,N_d)+1;
                        i2=ceil(i1i2/N_d);
                        Fmatrix(i1i2,i3,i3i4)=ReturnFn(d_gridvals(i1),a_gridvals(i2),a_gridvals(i3),z_gridvals(i3i4,1),z_gridvals(i3i4,2),ParamCell{:});
                    end
                end
            end
        end %end N_d


    else % more than one-dimensional

        if N_d==0
            Fmatrix=zeros(N_a,N_a,N_z);
            for i3=1:N_z
                tempcell3 = num2cell(z_gridvals(i3,:));
                for i2=1:N_a
                    tempcell2 = num2cell(a_gridvals(i2,:));
                    for i1=1:N_a
                        tempcell1 = num2cell(a_gridvals(i1,:));
                        Fmatrix(i1,i2,i3)=ReturnFn(tempcell1{:},tempcell2{:},tempcell3{:},ParamCell{:});
                    end
                end
            end

        else
            % N_d
            Fmatrix=zeros(N_d*N_a,N_a,N_z);
            for i4=1:N_z % z today
                tempcell4 = num2cell(z_gridvals(i4,:));
                for i3=1:N_a % a today
                    tempcell3 = num2cell(a_gridvals(i3,:));
                    for i2=1:N_a % a tomorrow choice
                        tempcell2 = num2cell(a_gridvals(i2,:));
                        for i1=1:N_d % d choice
                            tempcell1 = num2cell(d_gridvals(i1,:));
                            Fmatrix(i1+(i2-1)*N_d,i3,i4)=ReturnFn(tempcell1{:},tempcell2{:},tempcell3{:},tempcell4{:},ParamCell{:});
                        end
                    end %end i2
                end %end i3
            end %end i4
        end % end if

    end %end 1D trick

elseif Parallel==1

    if isscalar(n_d) && isscalar(n_a) && isscalar(n_z)
        % hardcode the 1D for speed

        if N_d==0
            Fmatrix=zeros(N_a,N_a,N_z);
            parfor i3=1:N_z
                Fmatrix_z=zeros(N_a,N_a);
                for i2=1:N_a % a today
                    for i1=1:N_a % a' tomorrow
                        Fmatrix_z(i1,i2)=ReturnFn(a_gridvals(i1),a_gridvals(i2),z_gridvals(i3),ParamCell{:});
                    end
                end
                Fmatrix(:,:,i3)=Fmatrix_z;
            end
        else
            Fmatrix=zeros(N_d*N_a,N_a,N_z);
            % ALE
            parfor i4=1:N_z
                Fmatrix_z=zeros(N_d*N_a,N_a);
                for i3=1:N_a % a today
                    for i2=1:N_a % a' tomorrow
                        for i1=1:N_d % d choice
                            Fmatrix_z(i1+(i2-1)*N_d,i3)=ReturnFn(d_gridvals(i1),a_gridvals(i2),a_gridvals(i3),z_gridvals(i4),ParamCell{:});
                        end
                    end
                end
                Fmatrix(:,:,i4)=Fmatrix_z;
            end
        end

    elseif isscalar(n_d) && isscalar(n_a) && length(n_z)==2
        % d and a are one-dimensional, z is two-dimensional
        if N_d==0
            Fmatrix=zeros(N_a,N_a,N_z);
            parfor i3i4=1:N_z
                Fmatrix_z=zeros(N_a,N_a);
                for i2=1:N_a % a today
                    for i1=1:N_a % a' tomorrow
                        Fmatrix_z(i1,i2)=ReturnFn(a_gridvals(i1),a_gridvals(i2),z_gridvals(i3i4,1),z_gridvals(i3i4,2),ParamCell{:});
                    end
                end
                Fmatrix(:,:,i3i4)=Fmatrix_z;
            end
        else
            Fmatrix=zeros(N_d*N_a,N_a,N_z);
            % ALE
            parfor i4i5=1:N_z
                Fmatrix_z=zeros(N_d*N_a,N_a);
                for i3=1:N_a % a today
                    for i2=1:N_a % a' tomorrow
                        for i1=1:N_d % d choice
                            Fmatrix_z(i1+(i2-1)*N_d,i3)=ReturnFn(d_gridvals(i1),a_gridvals(i2),a_gridvals(i3),z_gridvals(i4i5,1),z_gridvals(i4i5,2),ParamCell{:});
                        end
                    end
                end
                Fmatrix(:,:,i4i5)=Fmatrix_z;
            end
        end

    else % more than one-dimensional
        if N_d==0
            Fmatrix=zeros(N_a,N_a,N_z);
            parfor i3=1:N_z
                tempcell3 = num2cell(z_gridvals(i3,:));
                Fmatrix_z=zeros(N_a,N_a);
                for i2=1:N_a
                    tempcell2 = num2cell(a_gridvals(i2,:));
                    for i1=1:N_a
                        tempcell1 = num2cell(a_gridvals(i1,:));
                        Fmatrix_z(i1,i2)=ReturnFn(tempcell1{:},tempcell2{:},tempcell3{:},ParamCell{:});
                    end
                end
                Fmatrix(:,:,i3)=Fmatrix_z;
            end
        else
            Fmatrix=zeros(N_d*N_a,N_a,N_z);
            % ALE
            parfor i4=1:N_z
                tempcell4 = num2cell(z_gridvals(i4,:));
                Fmatrix_z=zeros(N_d*N_a,N_a);
                for i3=1:N_a % a today
                    tempcell3 = num2cell(a_gridvals(i3,:));
                    for i2=1:N_a % a tomorrow
                        tempcell2 = num2cell(a_gridvals(i2,:));
                        for i1=1:N_d % d
                            tempcell1 = num2cell(d_gridvals(i1,:));
                            Fmatrix_z(i1+(i2-1)*N_d,i3)=ReturnFn(tempcell1{:},tempcell2{:},tempcell3{:},tempcell4{:},ParamCell{:});
                        end
                    end
                end
                Fmatrix(:,:,i4)=Fmatrix_z;
            end % end parfor i4
        end
    end
end

if N_d>0
    if Refine==1
        Fmatrix=reshape(Fmatrix,[N_d,N_a,N_a,N_z]); % This is the difference when using Refine
    end
end

end


