function PolicyValues=PolicyInd2Val_FHorz_CPU(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions)
% CPU fallback for PolicyInd2Val_FHorz. Only supports scalar n_d and scalar n_a.

N_a=prod(n_a);
N_z=prod(n_z);

if N_z==0
    if n_d(1)==0
        Policy=reshape(Policy,[1,N_a*N_j]);
        PolicyValues=zeros(1,N_a*N_j);
        PolicyValues(1,:)=a_grid(Policy(1,:));
        PolicyValues=reshape(PolicyValues,[1,n_a,N_j]);
    else
        Policy=reshape(Policy,[2,N_a*N_j]);
        PolicyValues=zeros(2,N_a*N_j);
        PolicyValues(1,:)=d_grid(Policy(1,:));
        PolicyValues(2,:)=a_grid(Policy(2,:));
        PolicyValues=reshape(PolicyValues,[2,n_a,N_j]);
    end
else
    if n_d(1)==0
        Policy=reshape(Policy,[1,N_a*N_z*N_j]);
        PolicyValues=zeros(1,N_a*N_z*N_j);
        PolicyValues(1,:)=a_grid(Policy(1,:));
        PolicyValues=reshape(PolicyValues,[1,n_a,n_z,N_j]);
    else
        Policy=reshape(Policy,[2,N_a*N_z*N_j]);
        PolicyValues=zeros(2,N_a*N_z*N_j);
        PolicyValues(1,:)=d_grid(Policy(1,:));
        PolicyValues(2,:)=a_grid(Policy(2,:));
        PolicyValues=reshape(PolicyValues,[2,n_a,n_z,N_j]);
    end
end

end
