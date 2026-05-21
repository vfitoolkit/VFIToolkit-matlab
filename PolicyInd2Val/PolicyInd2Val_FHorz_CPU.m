function PolicyValues=PolicyInd2Val_FHorz_CPU(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid)
% CPU version, just the basics



if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_aprime=l_a;
aprime_grid=a_grid;

N_a=prod(n_a);
N_z=prod(n_z);


% This CPU implementation could be vectorized to be much faster
Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j, vfoptions);
if n_d(1)==0
    PolicyValues=zeros(l_aprime,N_a,N_z,N_j);
    for jj=1:N_j
        for a_c=1:N_a
            for z_c=1:N_z
                temp_a=ind2grid_homemade(Policy(1,a_c,z_c,jj),n_a,aprime_grid);
                for ii=1:l_aprime
                    PolicyValues(ii,a_c,z_c,jj)=temp_a(ii);
                end
            end
        end
    end
    PolicyValues=reshape(PolicyValues,[l_aprime,n_a,n_z,N_j]);
else
    PolicyValues=zeros(l_d+l_aprime,N_a,N_z,N_j);
    for jj=1:N_j
        for a_c=1:N_a
            for z_c=1:N_z
                temp_d=ind2grid_homemade(n_d,Policy(1,a_c,z_c,jj),d_grid);
                for ii=1:l_d
                    PolicyValues(ii,a_c,z_c,jj)=temp_d(ii);
                end
                temp_a=ind2grid_homemade(n_a,Policy(2,a_c,z_c,jj),aprime_grid);
                for ii=1:l_aprime
                    PolicyValues(l_d+ii,a_c,z_c,jj)=temp_a(ii);
                end
            end
        end
    end
    PolicyValues=reshape(PolicyValues,[l_d+l_aprime,n_a,n_z,N_j]);
end


end
