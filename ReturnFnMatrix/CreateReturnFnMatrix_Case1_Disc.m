function Fmatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel)
%If there is no d variable, just input n_d=0 and d_grid=0

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);


if Parallel==0
    
    z_gridvals=zeros(N_z,length(n_z));
    for i1=1:N_z
        sub=zeros(1,length(n_z));
        sub(1)=rem(i1-1,n_z(1))+1;
        for ii=2:length(n_z)-1
            sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
        end
        sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
        
        if length(n_z)>1
            sub=sub+[0,cumsum(n_z(1:end-1))];
        end
        z_gridvals(i1,:)=z_grid(sub);
    end
    a_gridvals=zeros(N_a,length(n_a));
    for i2=1:N_a
        sub=zeros(1,length(n_a));
        sub(1)=rem(i2-1,n_a(1)+1);
        for ii=2:length(n_a)-1
            sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
        end
        sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
        
        if length(n_a)>1
            sub=sub+[0,cumsum(n_a(1:end-1))];
        end
        a_gridvals(i2,:)=a_grid(sub);
    end    
    
    if N_d==0
        Fmatrix=zeros(N_a,N_a,N_z);
        for i1=1:N_a
            %aprime_gridvals=ind2grid_homemade(n_a,i1,a_grid);
            for i2=1:N_a
                %a_gridvals=ind2grid_homemade(n_a,i2,a_grid);
                for i3=1:N_z
                    %z_gridvals=ind2grid_homemade(n_z,i3,z_grid);
                    Fmatrix(i1,i2,i3)=ReturnFn(a_gridvals(i1,:),a_gridvals(i2,:),z_gridvals(i3,:));
                end
            end
        end
            
    else
        Fmatrix=zeros(N_d*N_a,N_a,N_z);
    
        for i1=1:N_d
            %d_gridvals=ind2grid_homemade(n_d,i1,d_grid);
            sub=zeros(1,length(n_d));
            sub(1)=rem(i1-1,n_d(1))+1;
            for ii=2:length(n_d)-1
                sub(ii)=rem(ceil(i1/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
            end
            sub(length(n_d))=ceil(i1/prod(n_d(1:length(n_d)-1)));
            
            if length(n_d)>1
                sub=sub+[0,cumsum(n_d(1:end-1))];
            end
            d_gridvals=d_grid(sub);
        end
           
        for i1=1:N_d
            for i2=1:N_a
                %aprime_gridvals=ind2grid_homemade(n_a,i2,a_grid);
                %i1i2=sub2ind_homemade([N_d,N_a],[i1,i2]);
                i1i2=i1+(i2-1)*N_d;
                for i3=1:N_a
                    %a_gridvals=ind2grid_homemade(n_a,i3,a_grid);
                    for i4=1:N_z
                        %z_gridvals=ind2grid_homemade(n_z,i4,z_grid);
%                        Fmatrix(i1i2,i3,i4)=ReturnFn(d_gridvals(i1,:),a_gridvals(i2,:),a_gridvals(i3,:),z_gridvals(i4,:));
                        Fmatrix(i1i2,i3,i4)=ReturnFn(d_gridvals,a_gridvals(i2,:),a_gridvals(i3,:),z_gridvals(i4,:));
                    end
                end
            end
        end
    end
    
elseif Parallel==1
    
    a_gridvals=zeros(N_a,length(n_a));
    for i2=1:N_a
        sub=zeros(1,length(n_a));
        sub(1)=rem(i2-1,n_a(1)+1);
        for ii=2:length(n_a)-1
            sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
        end
        sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
        
        if length(n_a)>1
            sub=sub+[0,cumsum(n_a(1:end-1))];
        end
        a_gridvals(i2,:)=a_grid(sub);
    end

    if N_d==0
        Fmatrix=zeros(N_a,N_a,N_z);
        parfor i3=1:N_z
            sub=zeros(1,length(n_z));
            sub(1)=rem(i3-1,n_z(1))+1;
            for ii=2:length(n_z)-1
                sub(ii)=rem(ceil(i3/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
            end
            sub(length(n_z))=ceil(i3/prod(n_z(1:length(n_z)-1)));
            
            if length(n_z)>1
                sub=sub+[0,cumsum(n_z(1:end-1))];
            end
            z_gridvals=z_grid(sub);
            
            Fmatrix_z=zeros(N_a,N_a);
            for i1=1:N_a
                for i2=1:N_a
                    Fmatrix_z(i1,i2)=ReturnFn(a_gridvals(i1,:),a_gridvals(i2,:),z_gridvals);
                end
            end
            Fmatrix(:,:,i3)=Fmatrix_z;
        end
    else
        d_gridvals=zeros(N_d,length(n_d));
        for i2=1:N_d
            sub=zeros(1,length(n_d));
            sub(1)=rem(i2-1,n_d(1)+1);
            for ii=2:length(n_d)-1
                sub(ii)=rem(ceil(i2/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
            end
            sub(length(n_d))=ceil(i2/prod(n_d(1:length(n_d)-1)));
            
            if length(n_d)>1
                sub=sub+[0,cumsum(n_d(1:end-1))];
            end
            d_gridvals(i2,:)=d_grid(sub);
        end
        
        Fmatrix=zeros(N_d*N_a,N_a,N_z);
        parfor i4=1:N_z
            sub=zeros(1,length(n_z));
            sub(1)=rem(i4-1,n_z(1))+1;
            for ii=2:length(n_z)-1
                sub(ii)=rem(ceil(i4/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
            end
            sub(length(n_z))=ceil(i4/prod(n_z(1:length(n_z)-1)));
            
            if length(n_z)>1
                sub=sub+[0,cumsum(n_z(1:end-1))];
            end
            z_gridvals=z_grid(sub);
            
            Fmatrix_z=zeros(N_d*N_a,N_a);
            for i1=1:N_d
                for i2=1:N_a
                    for i3=1:N_a
                        Fmatrix_z(i1+(i2-1)*N_d,i3)=ReturnFn(d_gridvals(i1,:),a_gridvals(i2,:),a_gridvals(i3,:),z_gridvals);
                    end
                end
            end
            Fmatrix(:,:,i4)=Fmatrix_z;
        end
    end
end


end


