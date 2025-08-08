function PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_z,T,d_grid,a_grid,simoptions,inputkron)
% if inputkron==1, should be gridvals, not grid
% further, if inputkron==1 and simoptions.gridinterplayer==1, then a_gridvals must constain the 'fine' grid not the rough grid

if ~exist("inputkron",'var')
    inputkron=0;
end

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if inputkron==0 % usual setup
    l_aprime=length(n_a);
    if n_d(1)==0
        PolicyValuesPath=zeros(l_aprime,N_a,N_z,T,'gpuArray');
        if simoptions.gridinterplayer==0
            PolicyValuesPath(1,:,:,:)=a_grid(PolicyPath(1,:,:,:));
        elseif simoptions.gridinterplayer==1
            N_aprime=N_a+(N_a-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,N_a,N_a),a_grid(1:n_a(1)),linspace(1,N_a,N_aprime));
            PolicyValuesPath(1,:,:,:)=temp(PolicyPath(1,:,:,:));
        end
        if l_aprime>1
            if l_aprime>2
                for ii=2:l_aprime-1
                    PolicyValuesPath(ii,:,:,:)=a_grid(prod(n_a(1:ii-1))+PolicyPath(ii,:,:,:));
                end
            end
            PolicyValuesPath(l_aprime,:,:,:)=a_grid(prod(n_a(1:end-1))+PolicyPath(l_aprime,:,:,:));
        end
    else
        l_d=length(n_d);
        PolicyValuesPath=zeros(l_d+l_aprime,N_a,N_z,T,'gpuArray');
        PolicyValuesPath(1,:,:,:)=d_grid(PolicyPath(1,:,:,:));
        if l_d>1
            if l_d>2
                for ii=2:l_d-1
                    PolicyValuesPath(ii,:,:,:)=d_grid(prod(n_d(1:ii-1))+PolicyPath(ii,:,:,:));
                end
            end
            PolicyValuesPath(l_d,:,:,:)=d_grid(prod(n_d(1:end-1))+PolicyPath(l_d,:,:,:));
        end
        if simoptions.gridinterplayer==0
            PolicyValuesPath(l_d+1,:,:,:)=a_grid(PolicyPath(l_d+1,:,:,:));
        elseif simoptions.gridinterplayer==1
            N_aprime=N_a+(N_a-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,N_a,N_a),a_grid(1:n_a(1)),linspace(1,N_a,N_aprime));
            PolicyValuesPath(l_d+1,:,:,:)=temp(PolicyPath(l_d+1,:,:,:));
        end
        if l_aprime>1
            if l_aprime>2
                for ii=2:l_aprime-1
                    PolicyValuesPath(l_d+ii,:,:,:)=a_grid(prod(n_a(1:ii-1))+PolicyPath(l_d+ii,:,:,:));
                end
            end
            PolicyValuesPath(l_d+l_aprime,:,:,:)=a_grid(prod(n_a(1:end-1))+PolicyPath(l_d+l_aprime,:,:,:));
        end
    end

elseif inputkron==1 % for internal use
    % d_gridvals
    % a_gridvals
    l_aprime=length(n_a);
    if n_d(1)==0
        PolicyValuesPath=zeros(l_aprime,N_a,N_z,T,'gpuArray');
        if simoptions.gridinterplayer==0
            PolicyValuesPath(1:l_aprime,:,:,:)=reshape(a_grid(PolicyPath(1,:,:,:),:)',[l_aprime,N_a,N_z,T]);
        elseif simoptions.gridinterplayer==1
            PolicyPath(1,:,:,:)=(simoptions.ngridinterp+1)*(PolicyPath(1,:,:,:)-1)+PolicyPath(end,:,:,:); % switch aprime to the fine grid
            if ~isscalar(n_a)
                N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
                PolicyPath(1,:,:,:)=PolicyPath(1,:,:,:)+N_a1prime*(PolicyPath(2,:,:,:)-1);
            end
            PolicyValuesPath(1:l_aprime,:,:,:)=reshape(a_grid(PolicyPath(1,:,:,:),:)',[l_aprime,N_a,N_z,T]);
        end
    else
        l_d=length(n_d);
        PolicyValuesPath=zeros(l_d+l_aprime,N_a,N_z,T,'gpuArray');
        PolicyValuesPath(1:l_d,:,:,:)=reshape(d_grid(PolicyPath(1,:,:,:),:)',[l_d,N_a,N_z,T]);
        if simoptions.gridinterplayer==0
            PolicyValuesPath(l_d+1:l_d+l_aprime,:,:,:)=reshape(a_grid(PolicyPath(2,:,:,:),:)',[l_aprime,N_a,N_z,T]);
        elseif simoptions.gridinterplayer==1
            PolicyPath(2,:,:,:)=(simoptions.ngridinterp+1)*(PolicyPath(2,:,:,:)-1)+PolicyPath(end,:,:,:); % switch a1prime to the fine grid
            if ~isscalar(n_a)
                N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
                PolicyPath(2,:,:,:)=PolicyPath(2,:,:,:)+N_a1prime*(PolicyPath(3,:,:,:)-1);
            end
            PolicyValuesPath(l_d+1:l_d+l_aprime,:,:,:)=reshape(a_grid(PolicyPath(2,:,:,:),:)',[l_aprime,N_a,N_z,T]);     
        end
    end

end



end