function PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_z,T,d_grid,a_grid,vfoptions,outputkron)
% Note: can use vfoptions or simoptions

if ~exist('outputkron','var')
    outputkron=0;
end

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

l_aprime=length(n_a);
if vfoptions.experienceasset==1
    l_aprime=l_aprime-1;
end

PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);

if n_d(1)==0
    PolicyValuesPath=zeros(l_aprime,N_a,N_z,T,'gpuArray');
    if vfoptions.gridinterplayer==0
        PolicyValuesPath(1,:,:,:)=a_grid(PolicyPath(1,:,:,:));
    elseif vfoptions.gridinterplayer==1
        N_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
        temp=interp1(linspace(1,n_a(1),n_a(1)),a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)); % fine grid on a1prime
        PolicyValuesPath(1,:,:,:)=temp((vfoptions.ngridinterp+1)*(PolicyPath(1,:,:,:)-1)+PolicyPath(end,:,:,:)); % use fine index
    end
    if l_aprime>1
        for ii=2:l_aprime
            PolicyValuesPath(ii,:,:,:)=a_grid(prod(n_a(1:ii-1))+PolicyPath(ii,:,:,:));
        end
    end
else
    l_d=length(n_d);
    PolicyValuesPath=zeros(l_d+l_aprime,N_a,N_z,T,'gpuArray');
    PolicyValuesPath(1,:,:,:)=d_grid(PolicyPath(1,:,:,:));
    if l_d>1
        if l_d>2
            for ii=2:l_d-1
                PolicyValuesPath(ii,:,:,:)=d_grid(sum(n_d(1:ii-1))+PolicyPath(ii,:,:,:));
            end
        end
        PolicyValuesPath(l_d,:,:,:)=d_grid(sum(n_d(1:end-1))+PolicyPath(l_d,:,:,:));
    end
    if vfoptions.gridinterplayer==0
        PolicyValuesPath(l_d+1,:,:,:)=a_grid(PolicyPath(l_d+1,:,:,:));
    elseif vfoptions.gridinterplayer==1
        N_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
        temp=interp1(gpuArray(linspace(1,n_a(1),n_a(1)))',a_grid(1:n_a(1)),gpuArray(linspace(1,n_a(1),N_a1prime))'); % fine grid on a1prime
        PolicyValuesPath(l_d+1,:,:,:)=temp((vfoptions.ngridinterp+1)*(PolicyPath(l_d+1,:,:,:)-1)+PolicyPath(end,:,:,:)); % use fine index
    end
    if l_aprime>1
        for ii=2:l_aprime
            PolicyValuesPath(l_d+ii,:,:,:)=a_grid(sum(n_a(1:ii-1))+PolicyPath(l_d+ii,:,:,:));
        end
    end
end

if outputkron==0
    PolicyValuesPath=reshape(PolicyValuesPath,[size(PolicyValuesPath,1),n_a,n_z,T]);
end



end