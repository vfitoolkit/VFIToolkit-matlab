function PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid,vfoptions,outputkron)
% Can use simoptions or vfoptions. If user is calling it, will probably be
% vfoptions. But internally it gets used with simoptions. The options that
% it checks are all things that will be common to both.

if ~exist('outputkron','var')
    outputkron=0; % outputkron=1 is just for internal use
end

if isgpuarray(Policy)
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

if ~exist('vfoptions','var')
    l_aprime=l_a;
    aprime_grid=a_grid;
    n_aprime=n_a;
else
    ordinary=1;
    % If using a specific asset type, then remove from aprime
    if isfield(vfoptions,'experienceasset')
        if vfoptions.experienceasset>0
            l_aprime=l_a-1;
            aprime_grid=a_grid(1:sum(n_a(1:end-1)));
            n_aprime=n_a(1:end-1);
            ordinary=0;
        end
    elseif isfield(vfoptions,'experienceassetu')
        if vfoptions.experienceassetu>0
            l_aprime=l_a-1;
            aprime_grid=a_grid(1:sum(n_a(1:end-1)));
            n_aprime=n_a(1:end-1);
            ordinary=0;
        end
    elseif isfield(vfoptions,'riskyasset')
        if vfoptions.riskyasset>0
            l_aprime=l_a-1;
            aprime_grid=a_grid(1:sum(n_a(1:end-1)));
            n_aprime=n_a(1:end-1);
            ordinary=0;
        end
    elseif isfield(vfoptions,'residualasset')
        if vfoptions.residualasset>0
            l_aprime=l_a-1;
            aprime_grid=a_grid(1:sum(n_a(1:end-1)));
            n_aprime=n_a(1:end-1);
            ordinary=0;
        end
    end
    
    if isfield(vfoptions,'gridinterplayer')
        if vfoptions.gridinterplayer==1
            ordinary=0;
            l_aprime=l_a;
            if l_a==1
                aprime_grid=interp1(gpuArray(1:1:N_a)',a_grid,linspace(1,N_a,N_a+(N_a-1)*vfoptions.ngridinterp)');
                n_aprime=n_a+(n_a-1)*vfoptions.ngridinterp; % =length(aprime_grid)
            else
                a1prime_grid=interp1(gpuArray(1:1:n_a(1))',a_grid(1:n_a(1)),linspace(1,n_a(1),n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp)');
                aprime_grid=[a1prime_grid; a_grid(n_a(1)+1:end)];
                n_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp; % =length(aprime_grid)
                n_aprime=[n_a1prime,n_a(2:end)];
            end
            % Put the last two parts of Policy together to get the aprime index
            tempsize=size(Policy);
            Policy=reshape(Policy,[tempsize(1),prod(tempsize)/tempsize(1)]); % note: prod(tempsize) is just a presumably faster way to numel(Policy)
            Policy(l_d+1,:)=(vfoptions.ngridinterp+1)*(Policy(l_d+1,:)-1)+Policy(end,:); % combine first asset index with the last index (lower grid point and 2nd layer point) to get aprime index
            tempsize(1)=tempsize(1)-1; % we put two policies together, so this is how many are left
            Policy=reshape(Policy(1:end-1,:),tempsize); % get rid of last policy entry
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

% cumsum_n_a=cumsum(n_a);


%%
if Parallel==2
    if N_z==0
        if l_d==0
            Policy=reshape(Policy,[l_aprime,N_a]);
            PolicyValues=zeros(l_aprime,N_a,'gpuArray');

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
                PolicyValues=reshape(PolicyValues,[l_aprime,n_a]);
            else
                PolicyValues=reshape(PolicyValues,[l_aprime,N_a]);
            end
        else
            Policy=reshape(Policy,[l_d+l_aprime,N_a]);
            PolicyValues=zeros(l_d+l_aprime,N_a,'gpuArray');

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
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,n_a]);
            else
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,N_a]);
            end
        end
    else % N_z>0
        if l_d==0
            Policy=reshape(Policy,[l_aprime,N_a*N_z]);
            PolicyValues=zeros(l_aprime,N_a*N_z,'gpuArray');

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
                PolicyValues=reshape(PolicyValues,[l_aprime,n_a,n_z]);
            else
                PolicyValues=reshape(PolicyValues,[l_aprime,N_a,N_z]);
            end
        else
            Policy=reshape(Policy,[l_d+l_aprime,N_a*N_z]);
            PolicyValues=zeros(l_d+l_aprime,N_a*N_z,'gpuArray');

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
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,n_a,n_z]);
            else
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,N_a,N_z]);
            end
        end
    end    

end


%% On cpu, limited to just the most basic setup (cannot handle no z)
if Parallel~=2
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


end