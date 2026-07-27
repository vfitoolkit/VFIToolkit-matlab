function PolicyValues=PolicyInd2Val_InfHorz_CPU(Policy,n_d,n_a,n_z,d_grid,a_grid,vfoptions,outputkron)
% On cpu, limited to just the most basic setup (cannot handle no z)

if l_d==0
    Policy=reshape(Policy,[l_a,N_a*N_z]);
    PolicyValues=zeros(l_a,N_a*N_z);

    temp_a_grid=a_grid(1:n_a(1));
    PolicyValues(1,:)=temp_a_grid(Policy(1,:));
    if l_a>1
        if l_a>2
            for ii=2:l_a-1
                temp_a_grid=a_grid((1+cumsum_n_a(ii-1)):cumsum_n_a(ii));
                PolicyValues(ii,:)=temp_a_grid(Policy(ii,:));
            end
        end
        temp_a_grid=a_grid((1+cumsum_n_a(end-1)):end);
        PolicyValues(end,:)=temp_a_grid(Policy(end,:));
    end
    PolicyValues=reshape(PolicyValues,[l_a,n_a,n_z]);

else
    Policy=reshape(Policy,[l_d+l_a,N_a*N_z]);
    PolicyValues=zeros(l_d+l_a,N_a*N_z);

    temp_d_grid=d_grid(1:n_d(1));
    PolicyValues(1,:)=temp_d_grid(Policy(1,:));
    if l_d>1
        if l_d>2
            for ii=2:l_d-1
                temp_d_grid=d_grid((1+cumsum_n_d(ii-1)):cumsum_n_d(ii));
                PolicyValues(ii,:)=temp_d_grid(Policy(ii,:));
            end
        end
        temp_d_grid=d_grid((1+cumsum_n_d(l_d-1)):end);
        PolicyValues(l_d,:)=temp_d_grid(Policy(l_d,:));
    end

    temp_a_grid=a_grid(1:n_a(1));
    PolicyValues(l_d+1,:)=temp_a_grid(Policy(l_d+1,:));
    if l_a>1
        if l_a>2
            for ii=2:l_a-1
                temp_a_grid=a_grid((1+cumsum_n_a(ii-1)):cumsum_n_a(ii));
                PolicyValues(l_d+ii,:)=temp_a_grid(Policy(l_d+ii,:));
            end
        end
        temp_a_grid=a_grid((1+cumsum_n_a(l_a-1)):end);
        PolicyValues(end,:)=temp_a_grid(Policy(end,:));
    end

    PolicyValues=reshape(PolicyValues,[l_d+l_a,n_a,n_z]);
end


end
