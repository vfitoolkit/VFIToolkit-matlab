function PolicyValues=PolicyInd2Val_Case2(PolicyIndexes,n_d,n_a,n_z,d_grid,Parallel)
% Parallel is an optional input

if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
end

l_d=length(n_d);
% l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);

if Parallel==2
    PolicyIndexes=reshape(PolicyIndexes,[l_d,N_a*N_z]);
    PolicyValues=zeros(l_d,N_a*N_z,'gpuArray');
    
    cumsum_n_d=cumsum(n_d);
    
    temp_d_grid=d_grid(1:cumsum_n_d(1));
    PolicyValues(1,:)=temp_d_grid(PolicyIndexes(1,:));
    if l_d>1
        if l_d>2
            for ii=2:l_d
                temp_d_grid=d_grid(1+cumsum_n_d(ii-1):cumsum_n_d(ii));
                PolicyValues(ii,:)=temp_d_grid(PolicyIndexes(ii,:));
            end
        end
        temp_d_grid=d_grid(cumsum_n_d(end-1)+1:end);
        PolicyValues(end,:)=temp_d_grid(PolicyIndexes(end,:));
    end
    
    PolicyValues=reshape(PolicyValues,[l_d,n_a,n_z]);
end

% % FOLLOWING COULD BE MADE MUCH FASTER BY VECTORIZATION (ABOVE SHOULD WORK
% % REGARDLESS OF VALUE OF Parallel???)
% if Parallel~=2
%     PolicyValues=zeros(l_d,N_a,N_z);
%     for a_c=1:N_a
%         for z_c=1:N_z
%             temp_d=ind2grid_homemade(n_d,PolicyIndexes(a_c,z_c),d_grid);
%             for ii=1:length(n_d)
%                 PolicyValues(ii,a_c,z_c)=temp_d(ii);
%             end
%         end
%     end
% end
if Parallel~=2 % Is exactly the same as Parallel==2, except now array rather than gpuArray
    PolicyIndexes=reshape(PolicyIndexes,[l_d,N_a*N_z]);
    PolicyValues=zeros(l_d,N_a*N_z);
    
    cumsum_n_d=cumsum(n_d);
    
    temp_d_grid=d_grid(1:cumsum_n_d(1));
    PolicyValues(1,:)=temp_d_grid(PolicyIndexes(1,:));
    if l_d>1
        if l_d>2
            for ii=2:l_d
                temp_d_grid=d_grid(1+cumsum_n_d(ii-1):cumsum_n_d(ii));
                PolicyValues(ii,:)=temp_d_grid(PolicyIndexes(ii,:));
            end
        end
        temp_d_grid=d_grid(cumsum_n_d(end-1)+1:end);
        PolicyValues(end,:)=temp_d_grid(PolicyIndexes(end,:));
    end
    
    PolicyValues=reshape(PolicyValues,[l_d,n_a,n_z]);
end


end