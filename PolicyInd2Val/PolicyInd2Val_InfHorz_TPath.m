function PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_z,T,d_grid,a_grid,vfoptions,inputkron)
% if inputkron==1, should be gridvals, not grid
% further, if inputkron==1 and simoptions.gridinterplayer==1, then a_gridvals must constain the 'fine' grid not the rough grid
% Note: can use vfoptions or simoptions

if ~exist("inputkron",'var')
    inputkron=0;
end

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

l_aprime=length(n_a);
if vfoptions.experienceasset==1
    l_aprime=l_aprime-1;
    n_a=n_a(1:end-1); % Note, already did N_a which is used for this period endo states, so this is just next period from here on
    if inputkron==0 % usual setup
        a_grid=a_grid(1:sum(n_a)); % this is just next period from here on
    elseif inputkron==1 % is actually a_gridvals
        % a_grid=a_grid; % input is just a1prime_gridvals
    end
end

if inputkron==0 % usual setup
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);

    if n_d(1)==0
        PolicyValuesPath=zeros(l_aprime,N_a,N_z,T,'gpuArray');
        if vfoptions.gridinterplayer==0
            PolicyValuesPath(1,:,:,:)=a_grid(PolicyPath(1,:,:,:));
        elseif vfoptions.gridinterplayer==1
            N_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
            temp=interp1(linspace(1,n_a(1),n_a(1)),a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime));
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
        if vfoptions.gridinterplayer==0
            PolicyValuesPath(l_d+1,:,:,:)=a_grid(PolicyPath(l_d+1,:,:,:));
        elseif vfoptions.gridinterplayer==1
            N_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
            temp=interp1(gpuArray(linspace(1,n_a(1),n_a(1)))',a_grid(1:n_a(1)),gpuArray(linspace(1,n_a(1),N_a1prime))');
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
    % Instead of d_grid and a_grid, the inputs are actually
    % d_gridvals
    % aprime_gridvals % will be fine grid if vfoptions.gridinterplayer==1
    if n_d(1)==0
        PolicyValuesPath=zeros(l_aprime,N_a,N_z,T,'gpuArray');
        if vfoptions.gridinterplayer==0
            PolicyValuesPath(1:l_aprime,:,:,:)=reshape(a_grid(PolicyPath(1,:,:,:),:)',[l_aprime,N_a,N_z,T]);
        elseif vfoptions.gridinterplayer==1
            PolicyPath(1,:,:,:)=(vfoptions.ngridinterp+1)*(PolicyPath(1,:,:,:)-1)+PolicyPath(end,:,:,:); % switch aprime to the fine grid
            if ~isscalar(n_a)
                N_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
                PolicyPath(1,:,:,:)=PolicyPath(1,:,:,:)+N_a1prime*(PolicyPath(2,:,:,:)-1);
            end
            PolicyValuesPath(1:l_aprime,:,:,:)=reshape(a_grid(PolicyPath(1,:,:,:),:)',[l_aprime,N_a,N_z,T]);
        end
    else
        l_d=length(n_d);
        PolicyValuesPath=zeros(l_d+l_aprime,N_a,N_z,T,'gpuArray');
        PolicyValuesPath(1:l_d,:,:,:)=reshape(d_grid(PolicyPath(1,:,:,:),:)',[l_d,N_a,N_z,T]);
        if vfoptions.gridinterplayer==0
            PolicyValuesPath(l_d+1:l_d+l_aprime,:,:,:)=reshape(a_grid(PolicyPath(2,:,:,:),:)',[l_aprime,N_a,N_z,T]);
        elseif vfoptions.gridinterplayer==1
            % Convert to fine grid indexes
            PolicyPath(2,:,:,:)=(vfoptions.ngridinterp+1)*(PolicyPath(2,:,:,:)-1)+PolicyPath(end,:,:,:); % switch a1prime to the fine grid
            if ~isscalar(n_a)
                N_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
                PolicyPath(2,:,:,:)=PolicyPath(2,:,:,:)+N_a1prime*(PolicyPath(3,:,:,:)-1);
            end
            % Now evaluate as usual
            PolicyValuesPath(l_d+1:l_d+l_aprime,:,:,:)=reshape(a_grid(PolicyPath(2,:,:,:),:)',[l_aprime,N_a,N_z,T]);     
        end
    end

end



end