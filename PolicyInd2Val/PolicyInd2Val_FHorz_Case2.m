function PolicyValues=PolicyInd2Val_FHorz_Case2(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,Parallel)
% Parallel is an optional input

if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
end

l_d=length(n_d);
% l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);

cumsum_n_a=cumsum(n_a);
cumsum_n_d=cumsum(n_d);

if Parallel==2
    PolicyIndexes=reshape(PolicyIndexes,[l_d,N_a*N_z*N_j]);
    PolicyValues=zeros(l_d,N_a*N_z*N_j,'gpuArray');
    
    temp_d_grid=d_grid(1:cumsum_n_d(1));
    PolicyValues(1,:)=temp_d_grid(PolicyIndexes(1,:));
    if l_d>1
        if l_d>2
            for ii=2:l_d
                temp_d_grid=d_grid(1+cumsum_n_d(ii-1):cumsum_n_d(ii));
                PolicyValues(ii,:)=temp_d_grid(PolicyIndexes(ii,:));
            end
        end
        temp_d_grid=d_grid(cumsum_n_d(end-1)+1:end);
        PolicyValues(end,:)=temp_d_grid(PolicyIndexes(end,:));
    end
    
    PolicyValues=reshape(PolicyValues,[l_d,n_a,n_z,N_j]);
end

if Parallel~=2
    PolicyValues=zeros(l_d,N_a,N_z,N_j);
    for a_c=1:N_a
        for z_c=1:N_z
            for jj=1:N_j
                temp_d=ind2grid_homemade(n_d,PolicyIndexes(a_c,z_c,jj),d_grid);
                for ii=1:length(n_d)
                    PolicyValues(ii,a_c,z_c,jj)=temp_d(ii);
                end
            end
        end
    end
end


end
