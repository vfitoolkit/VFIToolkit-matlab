function PolicyPathKron=KronPolicyIndexes_InfHorz_TransPath(PolicyPath, n_d, n_a, n_z, T,simoptions)
% n_e is an optional input
%
% Input: Policy (l_d+l_a,n_a,n_z,T);
%
% Output: Policy=zeros(2,N_a,N_z,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,T) if there is no d
% Differs if simoptions.gridinterplayer=1

% When using n_e, is instead:
% Input: Policy (l_d+l_a,n_a,n_z,n_e,,T);
%
% Output: Policy=zeros(2,N_a,N_z,N_e,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_e,T) if there is no d
% Differs if simoptions.gridinterplayer=1

N_d=prod(n_d);
N_a=prod(n_a);

l_a=length(n_a);

if ~isfield(simoptions,'n_e')
    n_ze=n_z;
    N_e=0;
elseif prod(simoptions.n_e)==0
    n_ze=n_z;
    N_e=0;    
else
    n_ze=[n_z,n_e];
    N_e=prod(n_e);
end
N_ze=prod(n_ze);
PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_ze,T]);


%%
if N_d==0
    if l_a==1
        PolicyPathKron=PolicyPath(1,:,:,:);
    elseif simoptions.gridinterplayer==0
        temp=[0; ones(l_a-1,1,'gpuArray')];
        temp2=gpuArray(cumprod(n_a')); % column vector
        PolicyTemp=(reshape(PolicyPath(1:l_a,:,:,:),[l_a,N_a*N_ze,T])-temp).*[1; temp2(1:end-1)]; % note, lots of autofilling
        PolicyPathKron=reshape(sum(PolicyTemp,1),[N_a,N_ze,T]);
    elseif simoptions.gridinterplayer==1
        PolicyPathKron=zeros(2,N_a,N_ze,T,'gpuArray');
        PolicyPathKron(1,:,:,:)=PolicyPath(1,:,:,:);
        if l_a==2
            PolicyPathKron(2,:,:,:)=PolicyPath(2,:,:,:);
        else
            temp=[0; ones(l_a-2,1,'gpuArray')];
            temp2=gpuArray(cumprod(n_a(2:end)')); % column vector
            PolicyTemp=(reshape(PolicyPath(2:l_a,:,:,:),[l_a-1,N_a*N_ze,T])-temp).*[1; temp2(1:end-1)]; % note, lots of autofilling
            PolicyPathKron(2,:,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_ze,T]);
        end
    end

elseif N_d>0
    l_d=length(n_d);

    PolicyPathKron=zeros(2,N_a,N_ze,T,'gpuArray');
    if l_d==1
        PolicyPathKron(1,:,:,:)=PolicyPath(1,:,:,:);
    else
        temp=[0; ones(l_d-1,1,'gpuArray')];
        temp2=gpuArray(cumprod(n_d')); % column vector
        PolicyTemp=(reshape(PolicyPath(1:l_d,:,:,:),[l_d,N_a*N_ze,T])-temp).*[1;temp2(1:end-1)]; % note, lots of autofilling
        PolicyPathKron(1,:,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_ze,T]);
    end
    % Then, a
    if l_a==1
        PolicyPathKron(2,:,:,:)=PolicyPath(l_d+1,:,:,:);
    elseif simoptions.gridinterplayer==0
        temp=[0; ones(l_a-1,1,'gpuArray')];
        temp2=gpuArray(cumprod(n_a')); % column vector
        PolicyTemp=(reshape(PolicyPath(l_d+1:l_d+l_a,:,:,:),[l_a,N_a*N_ze,T])-temp).*[1; temp2(1:end-1)]; % note, lots of autofilling
        PolicyPathKron(2,:,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_ze,T]);
    elseif simoptions.gridinterplayer==1
        PolicyPathKron=zeros(2,N_a,N_ze,T,'gpuArray');
        PolicyPathKron(2,:,:,:)=PolicyPath(l_d+1,:,:,:);
        if l_a==2
            PolicyPathKron(3,:,:,:)=PolicyPath(l_d+2,:,:,:);
        else
            temp=[0; ones(l_a-2,1,'gpuArray')];
            temp2=gpuArray(cumprod(n_a(2:end)')); % column vector
            PolicyTemp=(reshape(PolicyPath(l_d+2:l_d+l_a,:,:,:),[l_a-1,N_a*N_ze,T])-temp).*[1; temp2(1:end-1)]; % note, lots of autofilling
            PolicyPathKron(3,:,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_ze,T]);
        end
    end
end

if simoptions.gridinterplayer==1
    if N_d==0
        PolicyPathKron=[PolicyPathKron; PolicyPath(l_a+1,:,:,:)]; % L2 indexes
    else
        PolicyPathKron=[PolicyPathKron; PolicyPath(l_d+l_a+1,:,:,:)]; % L2 indexes
    end
end

if N_e>0
    if N_d==0 && simoptions.gridinterplayer==0
        PolicyPathKron=reshape(PolicyPathKron,[N_a,N_z,N_e,T]);
    else
        PolicyPathKron=reshape(PolicyPathKron,[size(PolicyPathKron,1),N_a,N_z,N_e,T]);
    end
end

end