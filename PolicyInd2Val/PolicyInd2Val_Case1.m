function PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,vfoptions,outputkron)

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
            PolicyIndexes=reshape(PolicyIndexes,[l_aprime,N_a]);
            PolicyValues=zeros(l_aprime,N_a,'gpuArray');

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
                PolicyValues=reshape(PolicyValues,[l_aprime,n_a]);
            else
                PolicyValues=reshape(PolicyValues,[l_aprime,N_a]);
            end
        else
            PolicyIndexes=reshape(PolicyIndexes,[l_d+l_aprime,N_a]);
            PolicyValues=zeros(l_d+l_aprime,N_a,'gpuArray');

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
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,n_a]);
            else
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,N_a]);
            end
        end
    else % N_z>0
        if l_d==0
            PolicyIndexes=reshape(PolicyIndexes,[l_aprime,N_a*N_z]);
            PolicyValues=zeros(l_aprime,N_a*N_z,'gpuArray');

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
                PolicyValues=reshape(PolicyValues,[l_aprime,n_a,n_z]);
            else
                PolicyValues=reshape(PolicyValues,[l_aprime,N_a,N_z]);
            end
        else
            PolicyIndexes=reshape(PolicyIndexes,[l_d+l_aprime,N_a*N_z]);
            PolicyValues=zeros(l_d+l_aprime,N_a*N_z,'gpuArray');

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
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,n_a,n_z]);
            else
                PolicyValues=reshape(PolicyValues,[l_d+l_aprime,N_a,N_z]);
            end
        end
    end

        % if l_d==0
        %     PolicyIndexes=reshape(PolicyIndexes,[l_a,N_a*N_z]);
        %     PolicyValues=zeros(l_a,N_a*N_z,'gpuArray');
        % 
        %     temp_a_grid=a_grid(1:n_a(1));
        %     PolicyValues(1,:)=temp_a_grid(PolicyIndexes(1,:));
        %     if l_a>1
        %         if l_a>2
        %             for ii=2:l_a-1
        %                 temp_a_grid=a_grid((1+cumsum_n_a(ii-1)):cumsum_n_a(ii));
        %                 PolicyValues(ii,:)=temp_a_grid(PolicyIndexes(ii,:));
        %             end
        %         end
        %         temp_a_grid=a_grid((1+cumsum_n_a(end-1)):end);
        %         PolicyValues(end,:)=temp_a_grid(PolicyIndexes(end,:));
        %     end
        %     PolicyValues=reshape(PolicyValues,[l_a,n_a,n_z]);
        % else
        %     PolicyIndexes=reshape(PolicyIndexes,[l_d+l_a,N_a*N_z]);
        %     PolicyValues=zeros(l_d+l_a,N_a*N_z,'gpuArray');
        % 
        %     temp_d_grid=d_grid(1:n_d(1));
        %     PolicyValues(1,:)=temp_d_grid(PolicyIndexes(1,:));
        %     if l_d>1
        %         if l_d>2
        %             for ii=2:l_d-1
        %                 temp_d_grid=d_grid((1+cumsum_n_d(ii-1)):cumsum_n_d(ii));
        %                 PolicyValues(ii,:)=temp_d_grid(PolicyIndexes(ii,:));
        %             end
        %         end
        %         temp_d_grid=d_grid((1+cumsum_n_d(l_d-1)):end);
        %         PolicyValues(l_d,:)=temp_d_grid(PolicyIndexes(l_d,:));
        %     end
        % 
        %     temp_a_grid=a_grid(1:n_a(1));
        %     PolicyValues(l_d+1,:)=temp_a_grid(PolicyIndexes(l_d+1,:));
        %     if l_a>1
        %         if l_a>2
        %             for ii=2:l_a-1
        %                 temp_a_grid=a_grid((1+cumsum_n_a(ii-1)):cumsum_n_a(ii));
        %                 PolicyValues(l_d+ii,:)=temp_a_grid(PolicyIndexes(l_d+ii,:));
        %             end
        %         end
        %         temp_a_grid=a_grid((1+cumsum_n_a(l_a-1)):end);
        %         PolicyValues(end,:)=temp_a_grid(PolicyIndexes(end,:));
        %     end
        % 
        %     PolicyValues=reshape(PolicyValues,[l_d+l_a,n_a,n_z]);
        % end

end


%% On cpu, limited to just the most basic setup (cannot handle no z)
if Parallel~=2
    if l_d==0
        PolicyIndexes=reshape(PolicyIndexes,[l_a,N_a*N_z]);
        PolicyValues=zeros(l_a,N_a*N_z);

        temp_a_grid=a_grid(1:n_a(1));
        PolicyValues(1,:)=temp_a_grid(PolicyIndexes(1,:));
        if l_a>1
            if l_a>2
                for ii=2:l_a-1
                    temp_a_grid=a_grid((1+cumsum_n_a(ii-1)):cumsum_n_a(ii));
                    PolicyValues(ii,:)=temp_a_grid(PolicyIndexes(ii,:));
                end
            end
            temp_a_grid=a_grid((1+cumsum_n_a(end-1)):end);
            PolicyValues(end,:)=temp_a_grid(PolicyIndexes(end,:));
        end
        PolicyValues=reshape(PolicyValues,[l_a,n_a,n_z]);
    else
        PolicyIndexes=reshape(PolicyIndexes,[l_d+l_a,N_a*N_z]);
        PolicyValues=zeros(l_d+l_a,N_a*N_z);

        temp_d_grid=d_grid(1:n_d(1));
        PolicyValues(1,:)=temp_d_grid(PolicyIndexes(1,:));
        if l_d>1
            if l_d>2
                for ii=2:l_d-1
                    temp_d_grid=d_grid((1+cumsum_n_d(ii-1)):cumsum_n_d(ii));
                    PolicyValues(ii,:)=temp_d_grid(PolicyIndexes(ii,:));
                end
            end
            temp_d_grid=d_grid((1+cumsum_n_d(l_d-1)):end);
            PolicyValues(l_d,:)=temp_d_grid(PolicyIndexes(l_d,:));
        end
        
        temp_a_grid=a_grid(1:n_a(1));
        PolicyValues(l_d+1,:)=temp_a_grid(PolicyIndexes(l_d+1,:));
        if l_a>1
            if l_a>2
                for ii=2:l_a-1
                    temp_a_grid=a_grid((1+cumsum_n_a(ii-1)):cumsum_n_a(ii));
                    PolicyValues(l_d+ii,:)=temp_a_grid(PolicyIndexes(l_d+ii,:));
                end
            end
            temp_a_grid=a_grid((1+cumsum_n_a(l_a-1)):end);
            PolicyValues(end,:)=temp_a_grid(PolicyIndexes(end,:));
        end
        
        PolicyValues=reshape(PolicyValues,[l_d+l_a,n_a,n_z]);
    end
end


end