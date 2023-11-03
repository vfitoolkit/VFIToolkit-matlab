function PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions)

if isgpuarray(PolicyIndexes)
    Parallel=2;
else
    Parallel=1;
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);

if N_z==0
    % codes still work if we prentend z is a single point (as the sizes are not really affected)
    n_z=1; 
    N_z=1;
end

if ~exist('simoptions','var')
    l_aprime=l_a;
    aprime_grid=a_grid;
    n_aprime=n_a;
else
    % If using a specific asset type, then remove from aprime
    if isfield(simoptions,'experienceasset')
        l_aprime=l_a-1;
        aprime_grid=a_grid(1:sum(n_a(1:end-1)));
        n_aprime=n_a(1:end-1);
    elseif isfield(simoptions,'riskyasset')
        l_aprime=l_a-1;
        aprime_grid=a_grid(1:sum(n_a(1:end-1)));
        n_aprime=n_a(1:end-1);
    elseif isfield(simoptions,'residualasset')
        l_aprime=l_a-1;
        aprime_grid=a_grid(1:sum(n_a(1:end-1)));
        n_aprime=n_a(1:end-1);
    else % not using any specific asset type
        l_aprime=l_a;
        aprime_grid=a_grid;
        n_aprime=n_a;
    end
end

cumsum_n_aprime=cumsum(n_aprime);
cumsum_n_d=cumsum(n_d);

if Parallel==2
    if l_d==0
        PolicyIndexes=reshape(PolicyIndexes,[l_aprime,N_a*N_z*N_j]);
        PolicyValues=zeros(l_aprime,N_a*N_z*N_j,'gpuArray');

        temp_aprime_grid=aprime_grid(1:cumsum_n_aprime(1));
        PolicyValues(1,:)=temp_aprime_grid(PolicyIndexes(1,:));
        if l_aprime>1
            if l_aprime>2
                for ii=2:(l_aprime-1)
                    temp_aprime_grid=aprime_grid(1+cumsum_n_aprime(ii-1):cumsum_n_aprime(ii));
                    PolicyValues(ii,:)=temp_aprime_grid(PolicyIndexes(ii,:));
                end
            end
            temp_aprime_grid=aprime_grid(cumsum_n_aprime(end-1)+1:end);
            PolicyValues(end,:)=temp_aprime_grid(PolicyIndexes(end,:));
        end
        PolicyValues=reshape(PolicyValues,[l_aprime,n_a,n_z,N_j]);
    else
        PolicyIndexes=reshape(PolicyIndexes,[l_d+l_aprime,N_a*N_z*N_j]);
        PolicyValues=zeros(l_d+l_aprime,N_a*N_z*N_j,'gpuArray');

        temp_d_grid=d_grid(1:cumsum_n_d(1));
        PolicyValues(1,:)=temp_d_grid(PolicyIndexes(1,:));
        if l_d>1
            if l_d>2
                for ii=2:(l_d-1)
                    temp_d_grid=d_grid(1+cumsum_n_d(ii-1):cumsum_n_d(ii));
                    PolicyValues(ii,:)=temp_d_grid(PolicyIndexes(ii,:));
                end
            end
            temp_d_grid=d_grid(cumsum_n_d(end-1)+1:end);
            PolicyValues(l_d,:)=temp_d_grid(PolicyIndexes(l_d,:));
        end
        
        if l_aprime>0
            temp_aprime_grid=aprime_grid(1:cumsum_n_aprime(1));
            PolicyValues(l_d+1,:)=temp_aprime_grid(PolicyIndexes(l_d+1,:));
            if l_aprime>1
                if l_aprime>2
                    for ii=2:(l_aprime-1)
                        temp_aprime_grid=aprime_grid(1+cumsum_n_aprime(ii-1):cumsum_n_aprime(ii));
                        PolicyValues(l_d+ii,:)=temp_aprime_grid(PolicyIndexes(l_d+ii,:));
                    end
                end
                temp_aprime_grid=aprime_grid(cumsum_n_aprime(end-1)+1:end);
                PolicyValues(l_d+l_aprime,:)=temp_aprime_grid(PolicyIndexes(l_d+l_aprime,:));
            end
        end

        PolicyValues=reshape(PolicyValues,[l_d+l_aprime,n_a,n_z,N_j]);
    end
end

% This CPU implementation could be vectorized to be much faster
if Parallel~=2
    PolicyIndexes=KronPolicyIndexes_FHorz_Case1(PolicyIndexes, n_d, n_a, n_z, N_j);
    if n_d(1)==0
        PolicyValues=zeros(l_aprime,N_a,N_z,N_j);
        for jj=1:N_j
            for a_c=1:N_a
                for z_c=1:N_z
                    temp_a=ind2grid_homemade(PolicyIndexes(a_c,z_c,jj),n_a,aprime_grid);
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
                    temp_d=ind2grid_homemade(n_d,PolicyIndexes(1,a_c,z_c,jj),d_grid);
                    for ii=1:l_d
                        PolicyValues(ii,a_c,z_c,jj)=temp_d(ii);
                    end
                    temp_a=ind2grid_homemade(n_a,PolicyIndexes(2,a_c,z_c,jj),aprime_grid);
                    for ii=1:l_aprime
                        PolicyValues(l_d+ii,a_c,z_c,jj)=temp_a(ii);
                    end
                end
            end
        end
        PolicyValues=reshape(PolicyValues,[l_d+l_aprime,n_a,n_z,N_j]);
    end
end


end