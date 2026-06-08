function PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,outputkron)
% Can use simoptions or vfoptions. If user is calling it, will probably be
% vfoptions. But internally it gets used with simoptions. The options that
% it checks are all things that will be common to both.

if isUnderlyingType(a_grid,'single')
    precision='single';
    precision_cast=@(x) single(x);
else
    precision='double';
    precision_cast=@(x) double(x);
end

if ~exist('outputkron','var')
    outputkron=0; % outputkron=1 is just for internal use
end

if ~isgpuarray(Policy)
    if gpuDeviceCount==0
        % CPU version, much more limited functionality
        PolicyValues=PolicyInd2Val_FHorz_CPU(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions);
        return
    else
        Policy=gpuArray(Policy);
    end
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);

if ~exist('vfoptions','var')
    l_aprime=l_a;
    aprime_grid=a_grid;
    n_aprime=n_a;
else
    l_aprime=l_a;
    aprime_grid=a_grid;
    n_aprime=n_a;
    % If using a specific asset type, then remove from aprime
    if isfield(vfoptions,'experienceasset') && vfoptions.experienceasset>0
        l_aprime=l_a-vfoptions.experienceasset;
        aprime_grid=a_grid(1:sum(n_a(1:end-vfoptions.experienceasset)));
        n_aprime=n_a(1:end-vfoptions.experienceasset);
    end
    if isfield(vfoptions,'experienceassetu') && vfoptions.experienceassetu>0
        l_aprime=l_a-vfoptions.experienceassetu;
        aprime_grid=a_grid(1:sum(n_a(1:end-vfoptions.experienceassetu)));
        n_aprime=n_a(1:end-vfoptions.experienceassetu);
    end
    if isfield(vfoptions,'experienceassetz') && vfoptions.experienceassetz>0
        l_aprime=l_a-vfoptions.experienceassetz;
        aprime_grid=a_grid(1:sum(n_a(1:end-vfoptions.experienceassetz)));
        n_aprime=n_a(1:end-vfoptions.experienceassetz);
    end
    if isfield(vfoptions,'experienceassete') && vfoptions.experienceassete>0
        l_aprime=l_a-vfoptions.experienceassete;
        aprime_grid=a_grid(1:sum(n_a(1:end-vfoptions.experienceassete)));
        n_aprime=n_a(1:end-vfoptions.experienceassete);
    end
    if isfield(vfoptions,'experienceassetze') && vfoptions.experienceassetze>0
        l_aprime=l_a-vfoptions.experienceassetze;
        aprime_grid=a_grid(1:sum(n_a(1:end-vfoptions.experienceassetze)));
        n_aprime=n_a(1:end-vfoptions.experienceassetze);
    end
    if isfield(vfoptions,'riskyasset') && vfoptions.riskyasset>0
        l_aprime=l_a-1;
        aprime_grid=a_grid(1:sum(n_a(1:end-1)));
        n_aprime=n_a(1:end-1);
    end
    if isfield(vfoptions,'residualasset') && vfoptions.residualasset>0
        l_aprime=l_a-1;
        aprime_grid=a_grid(1:sum(n_a(1:end-1)));
        n_aprime=n_a(1:end-1);
    end

    if isfield(vfoptions,'gridinterplayer')
        if vfoptions.gridinterplayer==1
            % Strip trailing PolicyL2flag channel (PolicyInd2Val doesn't need it; only sim does)
            tempsize=size(Policy);
            Policy=reshape(Policy,[tempsize(1),prod(tempsize)/tempsize(1)]);
            Policy=reshape(Policy(1:end-1,:), [tempsize(1)-1, tempsize(2:end)]);
            a1prime_grid=interp1(gpuArray(precision_cast(1):n_aprime(1))',aprime_grid(1:n_aprime(1)),linspace(precision_cast(1),n_aprime(1),n_aprime(1)+(n_aprime(1)-1)*vfoptions.ngridinterp))';
            if isscalar(n_aprime)
                aprime_grid=a1prime_grid;
            else
                aprime_grid=[a1prime_grid; aprime_grid(n_aprime(1)+1:end)];
            end
            n_aprime(1)=n_aprime(1)+(n_aprime(1)-1)*vfoptions.ngridinterp; % =length(a1prime_grid)
            % Put the last two parts of Policy together to get the aprime index
            tempsize=size(Policy);
            Policy=reshape(Policy,[tempsize(1),prod(tempsize)/tempsize(1)]); % note: prod(tempsize) is just a presumably faster way to numel(tempsize)
            Policy(end-l_aprime,:)=(vfoptions.ngridinterp+1)*(Policy(end-l_aprime,:)-1)+Policy(end,:); % combine (lower grid point and 2nd layer point) to get aprime index [lower grid point is in the first n_a, it is NOT end-l_a+1 because we then have another -1 for the 2nd layer index]
            tempsize(1)=tempsize(1)-1; % put last two policies together (lower grid point, and the second layer grid index; get aprime grid index)
            Policy=reshape(Policy(1:end-1,:),tempsize); % get rid of last policy entry
        end
    end
end

cumsum_n_aprime=cumsum(n_aprime);
cumsum_n_d=cumsum(n_d);

% When there is an e or semiz variable, can just pretend it is a z for
% current purposes
if isfield(vfoptions,'n_e')
    if prod(vfoptions.n_e)==0
        % do nothing
    else
        if N_z==0
            n_z=vfoptions.n_e;
        else
            n_z=[n_z,vfoptions.n_e];
        end
        N_z=prod(n_z);
    end
end
if isfield(vfoptions,'n_semiz')
    if prod(vfoptions.n_semiz)==0
        % do nothing
    else
        if N_z==0
            n_z=vfoptions.n_semiz;
        else
            n_z=[vfoptions.n_semiz,n_z];
        end
        N_z=prod(n_z);
    end
end




%%
if N_z==0
    if l_d==0
        Policy=reshape(Policy,[l_aprime,N_a*N_j]);
        PolicyValues=zeros(l_aprime,N_a*N_j,precision,'gpuArray');

        temp_aprime_grid=aprime_grid(1:cumsum_n_aprime(1));
        PolicyValues(1,:)=temp_aprime_grid(Policy(1,:));
        if l_aprime>1
            if l_aprime>2
                for ii=2:(l_aprime-1)
                    temp_aprime_grid=aprime_grid(1+cumsum_n_aprime(ii-1):cumsum_n_aprime(ii));
                    PolicyValues(ii,:)=temp_aprime_grid(Policy(ii,:));
                end
            end
            temp_aprime_grid=aprime_grid(cumsum_n_aprime(end-1)+1:end);
            PolicyValues(end,:)=temp_aprime_grid(Policy(end,:));
        end

        if outputkron==0
            PolicyValues=reshape(PolicyValues,[l_aprime,n_a,N_j]);
        else
            PolicyValues=reshape(PolicyValues,[l_aprime,N_a,N_j]);
        end
    else
        Policy=reshape(Policy,[l_d+l_aprime,N_a*N_j]);
        PolicyValues=zeros(l_d+l_aprime,N_a*N_j,precision,'gpuArray');
        
        temp_d_grid=d_grid(1:cumsum_n_d(1));
        PolicyValues(1,:)=temp_d_grid(Policy(1,:));
        if l_d>1
            if l_d>2
                for ii=2:(l_d-1)
                    temp_d_grid=d_grid(1+cumsum_n_d(ii-1):cumsum_n_d(ii));
                    PolicyValues(ii,:)=temp_d_grid(Policy(ii,:));
                end
            end
            temp_d_grid=d_grid(cumsum_n_d(end-1)+1:end);
            PolicyValues(l_d,:)=temp_d_grid(Policy(l_d,:));
        end

        if l_aprime>0
            temp_aprime_grid=aprime_grid(1:cumsum_n_aprime(1));
            PolicyValues(l_d+1,:)=temp_aprime_grid(Policy(l_d+1,:));
            if l_aprime>1
                if l_aprime>2
                    for ii=2:(l_aprime-1)
                        temp_aprime_grid=aprime_grid(1+cumsum_n_aprime(ii-1):cumsum_n_aprime(ii));
                        PolicyValues(l_d+ii,:)=temp_aprime_grid(Policy(l_d+ii,:));
                    end
                end
                temp_aprime_grid=aprime_grid(cumsum_n_aprime(end-1)+1:end);
                PolicyValues(l_d+l_aprime,:)=temp_aprime_grid(Policy(l_d+l_aprime,:));
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

        Policy=reshape(Policy,[l_aprime,N_a*N_z*N_j]);
        PolicyValues=zeros(l_aprime,N_a*N_z*N_j,precision,'gpuArray');

        temp_aprime_grid=aprime_grid(1:cumsum_n_aprime(1));
        PolicyValues(1,:)=temp_aprime_grid(Policy(1,:));
        if l_aprime>1
            if l_aprime>2
                for ii=2:(l_aprime-1)
                    temp_aprime_grid=aprime_grid(1+cumsum_n_aprime(ii-1):cumsum_n_aprime(ii));
                    PolicyValues(ii,:)=temp_aprime_grid(Policy(ii,:));
                end
            end
            temp_aprime_grid=aprime_grid(cumsum_n_aprime(end-1)+1:end);
            PolicyValues(end,:)=temp_aprime_grid(Policy(end,:));
        end

        if outputkron==0
            PolicyValues=reshape(PolicyValues,[l_aprime,n_a,n_z,N_j]);
        else
            PolicyValues=reshape(PolicyValues,[l_aprime,N_a,N_z,N_j]);
        end
    else
        Policy=reshape(Policy,[l_d+l_aprime,N_a*N_z*N_j]);
        PolicyValues=zeros(l_d+l_aprime,N_a*N_z*N_j,precision,'gpuArray');

        temp_d_grid=d_grid(1:cumsum_n_d(1));
        PolicyValues(1,:)=temp_d_grid(Policy(1,:));
        if l_d>1
            if l_d>2
                for ii=2:(l_d-1)
                    temp_d_grid=d_grid(1+cumsum_n_d(ii-1):cumsum_n_d(ii));
                    PolicyValues(ii,:)=temp_d_grid(Policy(ii,:));
                end
            end
            temp_d_grid=d_grid(cumsum_n_d(end-1)+1:end);
            PolicyValues(l_d,:)=temp_d_grid(Policy(l_d,:));
        end

        if l_aprime>0
            temp_aprime_grid=aprime_grid(1:cumsum_n_aprime(1));
            PolicyValues(l_d+1,:)=temp_aprime_grid(Policy(l_d+1,:));
            if l_aprime>1
                if l_aprime>2
                    for ii=2:(l_aprime-1)
                        temp_aprime_grid=aprime_grid(1+cumsum_n_aprime(ii-1):cumsum_n_aprime(ii));
                        PolicyValues(l_d+ii,:)=temp_aprime_grid(Policy(l_d+ii,:));
                    end
                end
                temp_aprime_grid=aprime_grid(cumsum_n_aprime(end-1)+1:end);
                PolicyValues(l_d+l_aprime,:)=temp_aprime_grid(Policy(l_d+l_aprime,:));
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
