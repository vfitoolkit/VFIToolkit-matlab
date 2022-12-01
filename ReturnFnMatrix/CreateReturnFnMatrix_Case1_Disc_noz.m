function Fmatrix=CreateReturnFnMatrix_Case1_Disc_noz(ReturnFn, n_d, n_a, d_grid, a_grid,Parallel,ReturnFnParamsVec,Refine) % Refine is an optional input
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

a_gridvals=CreateGridvals(n_a,a_grid,1);

if N_d~=0
    d_gridvals=CreateGridvals(n_d,d_grid,1);
end

if Parallel==0
        
    if N_d==0
        Fmatrix=zeros(N_a,N_a);
        for i1=1:N_a
            for i2=1:N_a
                tempcell=num2cell([a_gridvals(i1,:),a_gridvals(i2,:)]);
                Fmatrix(i1,i2)=ReturnFn(tempcell{:},ParamCell{:});
            end
        end
            
    else
        Fmatrix=zeros(N_d*N_a,N_a);
        
        for i1=1:N_d
            for i2=1:N_a
                i1i2=i1+(i2-1)*N_d;
                for i3=1:N_a
                    tempcell=num2cell([d_gridvals(i1,:),a_gridvals(i2,:),a_gridvals(i3,:)]);
                    Fmatrix(i1i2,i3)=ReturnFn(tempcell{:},ParamCell{:});
                end
            end
        end
    end
    
elseif Parallel==1
    
    if N_d==0
        Fmatrix=zeros(N_a,N_a);
        parfor i2=1:N_a
            Fmatrix_a=zeros(N_a,1);
            a_vals=a_gridvals(i2,:);
            for i1=1:N_a
                tempcell=num2cell([a_gridvals(i1,:),a_vals]);
                Fmatrix_a(i1)=ReturnFn(tempcell{:},ParamCell{:});
            end
            Fmatrix(:,i2)=Fmatrix_a;
        end
    else        
        Fmatrix=zeros(N_d*N_a,N_a);
        parfor i3=1:N_a
            Fmatrix_a=zeros(N_d*N_a,1);
            a_vals=a_gridvals(i3,:);
            for i1=1:N_d
                for i2=1:N_a
                    tempcell=num2cell([d_gridvals(i1,:),a_gridvals(i2,:),a_vals]);
                    Fmatrix_a(i1+(i2-1)*N_d)=ReturnFn(tempcell{:},ParamCell{:});
                end
            end
            Fmatrix(:,i3)=Fmatrix_a;
        end
    end
end

if N_d>0
    if Refine==1
        Fmatrix=reshape(Fmatrix,[N_d,N_a,N_a]); % This is the difference when using Refine
    end
end


end


