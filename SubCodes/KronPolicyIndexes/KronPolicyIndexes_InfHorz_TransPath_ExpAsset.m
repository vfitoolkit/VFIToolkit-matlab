function PolicyPathKron=KronPolicyIndexes_InfHorz_TransPath_ExpAsset(PolicyPath, n_d, n_a, n_z, T,simoptions)
% n_e is an optional input
%
% Input: Policy (l_d+l_aprime,n_a,n_z,T);
%
% Output: Policy=zeros(2,N_a,N_z,T); %first dim indexes the optimal choice for d2 and aprime rest of dimensions a,z 
%         Policy=zeros(3,N_a,N_z,T); %first dim indexes the optimal choice for d1,d2 and aprime rest of dimensions a,z 
% Differs if simoptions.gridinterplayer=1

% When using n_e, is instead:
% Input: Policy (l_d+l_aprime,n_a,n_z,n_e,,T);
%
% Output: Policy=zeros(2,N_a,N_z,N_e,T); %first dim indexes the optimal choice for d2 and aprime rest of dimensions a,z 
%         Policy=zeros(3,N_a,N_z,N_e,T); %first dim indexes the optimal choice for d1,d2 and aprime rest of dimensions a,z 
% Differs if simoptions.gridinterplayer=1

N_a=prod(n_a);

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
% Split decision variables into the standard ones and the one relevant to the experience asset
if isscalar(n_d)
    n_d1=0;
else
    n_d1=n_d(1:end-1);
end
n_d2=n_d(end); % n_d2 is the decision variable that influences next period vale of the experience asset
% Split endogenous assets into the standard ones and the experience asset
if isscalar(n_a)
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the experience asset

if isfield(simoptions,'aprimeFn')
    aprimeFn=simoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2)
    aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end


%%
nkron=1; % d2
dvars=1;
if n_d1>0
    nkron=nkron+1;
    dvars=2;
end
l_a1=0;
if n_a1(1)>0
    l_a1=length(n_a1);
    nkron=nkron+1;
    if length(n_a1)>1 && simoptions.gridinterplayer==1
        nkron=nkron+1;
    end
end
PolicyPathKron=zeros(nkron,N_a,N_ze,T,'gpuArray');


l_d2=length(n_d2);
if n_d1(1)==0
    if l_d2==1
        PolicyPathKron(1,:,:,:)=PolicyPath(1,:,:,:);
    else
        temp=[0; ones(l_d2-1,1,'gpuArray')];
        temp2=gpuArray(cumprod(n_d2')); % column vector
        PolicyTemp=(reshape(PolicyPath(1:l_d2,:,:,:),[l_d2,N_a*N_ze,T])-temp).*[1;temp2(1:end-1)]; % note, lots of autofilling
        PolicyPathKron(1,:,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_ze,T]);
    end
else
    l_d1=length(n_d1);

    if l_d1==1
        PolicyPathKron(1,:,:,:)=PolicyPath(1,:,:,:);
    else
        temp=[0; ones(l_d1-1,1,'gpuArray')];
        temp2=gpuArray(cumprod(n_d1')); % column vector
        PolicyTemp=(reshape(PolicyPath(1:l_d1,:,:,:),[l_d1,N_a*N_ze,T])-temp).*[1;temp2(1:end-1)]; % note, lots of autofilling
        PolicyPathKron(1,:,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_ze,T]);
    end

    if l_d2==1
        PolicyPathKron(2,:,:,:)=PolicyPath(l_d1+1,:,:,:);
    else
        temp=[0; ones(l_d2-1,1,'gpuArray')];
        temp2=gpuArray(cumprod(n_d2')); % column vector
        PolicyTemp=(reshape(PolicyPath(l_d1+1:l_d1+l_d2,:,:,:),[l_d2,N_a*N_ze,T])-temp).*[1;temp2(1:end-1)]; % note, lots of autofilling
        PolicyPathKron(2,:,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_ze,T]);
    end
end

l_d=length(n_d);
% Then, a1
if l_a1==1
    PolicyPathKron(dvars+1,:,:,:)=PolicyPath(l_d+1,:,:,:);
elseif simoptions.gridinterplayer==0
    temp=[0; ones(l_a-1,1,'gpuArray')];
    temp2=gpuArray(cumprod(n_a')); % column vector
    PolicyTemp=(reshape(PolicyPath(l_d+1:l_d+l_a,:,:,:),[l_a,N_a*N_ze,T])-temp).*[1; temp2(1:end-1)]; % note, lots of autofilling
    PolicyPathKron(dvars+1,:,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_ze,T]);
elseif simoptions.gridinterplayer==1
    PolicyPathKron(dvars+1,:,:,:)=PolicyPath(l_d+1,:,:,:);
    if l_a1==2
        PolicyPathKron(dvars+2,:,:,:)=PolicyPath(l_d+2,:,:,:);
    else
        temp=[0; ones(l_a-2,1,'gpuArray')];
        temp2=gpuArray(cumprod(n_a(2:end)')); % column vector
        PolicyTemp=(reshape(PolicyPath(l_d+2:l_d+l_a,:,:,:),[l_a-1,N_a*N_ze,T])-temp).*[1; temp2(1:end-1)]; % note, lots of autofilling
        PolicyPathKron(dvars+2,:,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_ze,T]);
    end
end

if simoptions.gridinterplayer==1
    PolicyPathKron(end,:,:,:)=PolicyPath(l_d+l_a+1,:,:,:); % L2 indexes
end

if N_e>0
    PolicyPathKron=reshape(PolicyPathKron,[size(PolicyPathKron,1),N_a,N_z,N_e,T]);
end

end