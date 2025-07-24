function PolicyValues=PolicyInd2Val_Case2(Policy,n_d,n_a,n_z,d_grid)
% Can use simoptions or vfoptions. If user is calling it, will probably be
% vfoptions. But internally it gets used with simoptions. The options that
% it checks are all things that will be common to both.

if isgpuarray(Policy)
    Parallel=2;
else
    Parallel=1;
end

l_d=length(n_d);

N_a=prod(n_a);
N_z=prod(n_z);

if Parallel==2
    Policy=reshape(Policy,[l_d,N_a*N_z]);
    PolicyValues=zeros(l_d,N_a*N_z,'gpuArray');
    
    cumsum_n_d=cumsum(n_d);
    
    temp_d_grid=d_grid(1:cumsum_n_d(1));
    PolicyValues(1,:)=temp_d_grid(Policy(1,:));
    if l_d>1
        if l_d>2
            for ii=2:l_d
                temp_d_grid=d_grid(1+cumsum_n_d(ii-1):cumsum_n_d(ii));
                PolicyValues(ii,:)=temp_d_grid(Policy(ii,:));
            end
        end
        temp_d_grid=d_grid(cumsum_n_d(end-1)+1:end);
        PolicyValues(end,:)=temp_d_grid(Policy(end,:));
    end
    
    PolicyValues=reshape(PolicyValues,[l_d,n_a,n_z]);
else
    Policy=reshape(Policy,[l_d,N_a*N_z]);
    PolicyValues=zeros(l_d,N_a*N_z);
    
    cumsum_n_d=cumsum(n_d);
    
    temp_d_grid=d_grid(1:cumsum_n_d(1));
    PolicyValues(1,:)=temp_d_grid(Policy(1,:));
    if l_d>1
        if l_d>2
            for ii=2:l_d
                temp_d_grid=d_grid(1+cumsum_n_d(ii-1):cumsum_n_d(ii));
                PolicyValues(ii,:)=temp_d_grid(Policy(ii,:));
            end
        end
        temp_d_grid=d_grid(cumsum_n_d(end-1)+1:end);
        PolicyValues(end,:)=temp_d_grid(Policy(end,:));
    end
    
    PolicyValues=reshape(PolicyValues,[l_d,n_a,n_z]);
end


end