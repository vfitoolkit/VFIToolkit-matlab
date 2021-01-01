function PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel)
% Parallel is an optional input

if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
    cumsum_n_d=cumsum(n_d);
end
l_a=length(n_a);

cumsum_n_a=cumsum(n_a);

N_a=prod(n_a);
N_z=prod(n_z);

if Parallel==2
    if l_d==0
        PolicyIndexes=reshape(PolicyIndexes,[l_a,N_a*N_z]);
        PolicyValues=zeros(l_a,N_a*N_z,'gpuArray');

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
        PolicyValues=zeros(l_d+l_a,N_a*N_z,'gpuArray');

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

% % % FOLLOWING COULD BE MADE MUCH FASTER BY VECTORIZATION
% % if Parallel~=2
% %     if n_d(1)==0
% %         PolicyValues=zeros(length(n_a),N_a,N_z); %NOTE: this is not actually in Kron form
% %         for a_c=1:N_a
% %             for z_c=1:N_z
% %                 temp_a=ind2grid_homemade(PolicyIndexes(a_c,z_c),n_a,a_grid);
% %                 for ii=1:length(n_a)
% %                     PolicyValues(ii,a_c,z_c)=temp_a(ii);
% %                 end
% %             end
% %         end
% %     else
% %         PolicyValues=zeros(length(n_d)+length(n_a),N_a,N_z);
% %         for a_c=1:N_a
% %             for z_c=1:N_z
% %                 temp_d=ind2grid_homemade(n_d,PolicyIndexes(1,a_c,z_c),d_grid);
% %                 for ii=1:length(n_d)
% %                     PolicyValues(ii,a_c,z_c)=temp_d(ii);
% %                 end
% %                 temp_a=ind2grid_homemade(n_a,PolicyIndexes(2,a_c,z_c),a_grid);
% %                 for ii=1:length(n_a)
% %                     PolicyValues(length(n_d)+ii,a_c,z_c)=temp_a(ii);
% %                 end
% %             end
% %         end
% %     end
% % end


end