function PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyPath,n_d,n_a,n_z,N_j,T,d_grid,a_grid,vfoptions,outputkron,fastOLG)
% Note: can use vfoptions or simoptions
% When using this in the fastOLG setting, you must input d_gridvals and aprime_gridvals in place of d_grid and a_grid

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
if N_z==0
    N_z=1; % trick to avoid rewriting the thing
    noz=1; % so I know the output should have a different shape
else
    noz=0;
end

% Following two are for internal use, just set to 0 for end user
if ~exist('outputkron','var')
    outputkron=0;
end
if ~exist('fastOLG','var')
    fastOLG=0;
end

if ~isfield(vfoptions,'experienceasset')
    vfoptions.experienceasset=0;
end

if fastOLG==0
    l_aprime=length(n_a);
    if vfoptions.experienceasset==1
        l_aprime=l_aprime-1;
    end

    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,N_j,T]);

    if n_d(1)==0
        PolicyValuesPath=zeros(l_aprime,N_a,N_z,N_j,T,'gpuArray');
        if vfoptions.gridinterplayer==0
            PolicyValuesPath(1,:,:,:,:)=a_grid(PolicyPath(1,:,:,:,:));
        elseif vfoptions.gridinterplayer==1
            N_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
            temp=interp1(linspace(1,n_a(1),n_a(1)),a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)); % fine grid on a1prime
            PolicyValuesPath(1,:,:,:,:)=temp((vfoptions.ngridinterp+1)*(PolicyPath(1,:,:,:,:)-1)+PolicyPath(end,:,:,:,:)); % use fine index
        end
        if l_aprime>1
            for ii=2:l_aprime
                PolicyValuesPath(ii,:,:,:,:)=a_grid(prod(n_a(1:ii-1))+PolicyPath(ii,:,:,:,:));
            end
        end
    else
        l_d=length(n_d);
        PolicyValuesPath=zeros(l_d+l_aprime,N_a,N_z,N_j,T,'gpuArray');
        PolicyValuesPath(1,:,:,:,:)=d_grid(PolicyPath(1,:,:,:,:));
        if l_d>1
            if l_d>2
                for ii=2:l_d-1
                    PolicyValuesPath(ii,:,:,:,:)=d_grid(sum(n_d(1:ii-1))+PolicyPath(ii,:,:,:,:));
                end
            end
            PolicyValuesPath(l_d,:,:,:,:)=d_grid(sum(n_d(1:end-1))+PolicyPath(l_d,:,:,:,:));
        end
        if vfoptions.gridinterplayer==0
            PolicyValuesPath(l_d+1,:,:,:,:)=a_grid(PolicyPath(l_d+1,:,:,:,:));
        elseif vfoptions.gridinterplayer==1
            N_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
            temp=interp1(gpuArray(linspace(1,n_a(1),n_a(1)))',a_grid(1:n_a(1)),gpuArray(linspace(1,n_a(1),N_a1prime))'); % fine grid on a1prime
            PolicyValuesPath(l_d+1,:,:,:,:)=temp((vfoptions.ngridinterp+1)*(PolicyPath(l_d+1,:,:,:,:)-1)+PolicyPath(end,:,:,:,:)); % use fine index
        end
        if l_aprime>1
            for ii=2:l_aprime
                PolicyValuesPath(l_d+ii,:,:,:,:)=a_grid(sum(n_a(1:ii-1))+PolicyPath(l_d+ii,:,:,:,:));
            end
        end
    end
    
    if noz==1
        if outputkron==1
            PolicyValuesPath=reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_j,T]);
        elseif outputkron==0
            PolicyValuesPath=reshape(PolicyValuesPath,[size(PolicyValuesPath,1),n_a,N_j,T]);
        end
    elseif outputkron==0
        PolicyValuesPath=reshape(PolicyValuesPath,[size(PolicyValuesPath,1),n_a,n_z,N_j,T]);
    end

else
    % fastOLG, so has a different shape
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a*N_j*N_z*T]);
    % When using this in the fastOLG setting, you must input d_gridvals and aprime_gridvals in place of d_grid and a_grid
    % If using vfoptions.gridinterplater=1, it should be aprime_gridvals for the fine grid, not the rough grid

    l_aprime=size(a_grid,2);% it is aprime_gridvals, not actually a_grid
    
    if n_d(1)==0
        PolicyValuesPath=zeros(l_aprime,N_a*N_j*N_z*T,'gpuArray'); % N_a*N_j*N_z*T is because of how indexing gridvals will work
        if vfoptions.gridinterplayer==1
            PolicyPath(1,:)=(vfoptions.ngridinterp+1)*(PolicyPath(1,:)-1)+PolicyPath(end,:); % fine index
        end
        for ii=1:l_aprime
            PolicyValuesPath(ii,:)=a_grid(PolicyPath(ii,:),ii); % it is aprime_gridvals, not actually a_grid
        end
    else
        l_d=length(n_d);
        PolicyValuesPath=zeros(l_d+l_aprime,N_a*N_j*N_z*T,'gpuArray'); % N_a*N_j*N_z*T is because of how indexing gridvals will work
        for ii=1:l_d
            PolicyValuesPath(ii,:)=d_grid(PolicyPath(ii,:),ii); % it is d_gridvals, not actually d_grid
        end
        if vfoptions.gridinterplayer==1
            PolicyPath(l_d+1,:)=(vfoptions.ngridinterp+1)*(PolicyPath(l_d+1,:)-1)+PolicyPath(end,:); % fine index
        end
        for ii=1:l_aprime
            PolicyValuesPath(l_d+ii,:)=a_grid(PolicyPath(l_d+ii,:),ii);
        end
    end
    
    if noz==1
        if outputkron==1
            PolicyValuesPath=reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_j,T]);
        elseif outputkron==0
            PolicyValuesPath=reshape(PolicyValuesPath,[size(PolicyValuesPath,1),n_a,N_j,T]);
        end
    else
        if outputkron==1
            PolicyValuesPath=reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_j,N_z,T]);
        elseif outputkron==0
            PolicyValuesPath=reshape(PolicyValuesPath,[size(PolicyValuesPath,1),n_a,N_j,n_z,T]);
        end
    end
end




end