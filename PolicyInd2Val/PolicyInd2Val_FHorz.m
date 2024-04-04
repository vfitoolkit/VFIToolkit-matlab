function PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,outputkron)

if ~exist('outputkron','var')
    outputkron=0; % outputkron=1 is just for internal use
end

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

if ~exist('simoptions','var')
    l_aprime=l_a;
    aprime_grid=a_grid;
    n_aprime=n_a;
else
    ordinary=1;
    % If using a specific asset type, then remove from aprime
    if isfield(simoptions,'experienceasset')
        if simoptions.experienceasset>0
            l_aprime=l_a-1;
            aprime_grid=a_grid(1:sum(n_a(1:end-1)));
            n_aprime=n_a(1:end-1);
            ordinary=0;
        end
    elseif isfield(simoptions,'experienceassetu')
        if simoptions.experienceassetu>0
            l_aprime=l_a-1;
            aprime_grid=a_grid(1:sum(n_a(1:end-1)));
            n_aprime=n_a(1:end-1);
            ordinary=0;
        end
    elseif isfield(simoptions,'riskyasset')
        if simoptions.riskyasset>0
            l_aprime=l_a-1;
            aprime_grid=a_grid(1:sum(n_a(1:end-1)));
            n_aprime=n_a(1:end-1);
            ordinary=0;
        end
    elseif isfield(simoptions,'residualasset')
        if simoptions.residualasset>0
            l_aprime=l_a-1;
            aprime_grid=a_grid(1:sum(n_a(1:end-1)));
            n_aprime=n_a(1:end-1);
            ordinary=0;
        end
    end
    if ordinary==1 % not using any specific asset type
        l_aprime=l_a;
        aprime_grid=a_grid;
        n_aprime=n_a;
    end
end

cumsum_n_aprime=cumsum(n_aprime);
cumsum_n_d=cumsum(n_d);

% When there is an e or semiz variable, can just pretend it is a z for
% current purposes
if isfield(simoptions,'n_e')
    if prod(simoptions.n_e)==0
        % do nothing
    else
        if N_z==0
            n_z=simoptions.n_e;
        else
            n_z=[n_z,simoptions.n_e];
        end
        N_z=prod(n_z);
    end
end
if isfield(simoptions,'n_semiz')
    if prod(simoptions.n_semiz)==0
        % do nothing
    else
        if N_z==0
            n_z=simoptions.n_semiz;
        else
            n_z=[n_z,simoptions.n_semiz];
        end
        N_z=prod(n_z);
    end
end


%%
if Parallel==2
    if N_z==0
        if l_d==0
            PolicyIndexes=reshape(PolicyIndexes,[l_aprime,N_a*N_j]);
            PolicyValues=zeros(l_aprime,N_a*N_j,'gpuArray');

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

            if outputkron==0
                PolicyValues=reshape(PolicyValues,[l_aprime,n_a,N_j]);
            else
                PolicyValues=reshape(PolicyValues,[l_aprime,N_a,N_j]);
            end
        else
            PolicyIndexes=reshape(PolicyIndexes,[l_d+l_aprime,N_a*N_j]);
            PolicyValues=zeros(l_d+l_aprime,N_a*N_j,'gpuArray');

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

            if outputkron==0
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,n_a,N_j]);
            else
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,N_a,N_j]);
            end
        end

    else % N_z
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

            if outputkron==0
                PolicyValues=reshape(PolicyValues,[l_aprime,n_a,n_z,N_j]);
            else
                PolicyValues=reshape(PolicyValues,[l_aprime,N_a,N_z,N_j]);
            end
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

            if outputkron==0
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,n_a,n_z,N_j]);
            else
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,N_a,N_z,N_j]);
            end
        end
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