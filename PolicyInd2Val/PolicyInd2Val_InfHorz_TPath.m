function PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_z,T,d_grid,a_grid,vfoptions,outputkron)
% vfoptions is a required input (can pass simoptions instead; the fields used are common to both).
% d_grid and a_grid inputs each accept either the stacked column grid or joint gridvals
% (a [prod(n),l] matrix); which is detected from the number of columns (for a single
% variable the two coincide, and either interpretation gives the same answer).
% a_grid input is on the coarse grid. If vfoptions.policyind2val_finegridinput=1 then the
% first asset part of a_grid input is instead the fine grid: stacked input is
% [fine a1; rest of coarse stacked grid], gridvals input is the fine aprime_gridvals
% (only relevant to vfoptions.gridinterplayer=1).

if ~exist('outputkron','var')
    outputkron=0;
end
if ~isfield(vfoptions,'gridinterplayer')
    vfoptions.gridinterplayer=0;
end
if ~isfield(vfoptions,'experienceasset')
    vfoptions.experienceasset=0;
end
if ~isfield(vfoptions,'experienceassetz')
    vfoptions.experienceassetz=0;
end
if ~isfield(vfoptions,'policyind2val_finegridinput')
    vfoptions.policyind2val_finegridinput=0; % =1 means a_grid input contains the fine grid for the first asset
end

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

l_aprime=length(n_a);
if vfoptions.experienceasset>=1
    l_aprime=l_aprime-1;
end

n_aprime=n_a;
if vfoptions.policyind2val_finegridinput==1
    n_aprime(1)=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp; % a_grid input contains the fine grid for the first asset
end

PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);

if n_d(1)==0
    PolicyValuesPath=zeros(l_aprime,N_a,N_z,T,'gpuArray');
    if vfoptions.gridinterplayer==0
        PolicyValuesPath(1,:,:,:)=a_grid(PolicyPath(1,:,:,:));
    elseif vfoptions.gridinterplayer==1
        N_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
        if vfoptions.policyind2val_finegridinput==0
            temp=interp1(linspace(1,n_a(1),n_a(1)),a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)); % fine grid on a1prime
        elseif vfoptions.policyind2val_finegridinput==1
            temp=a_grid(1:N_a1prime); % a_grid input already contains the fine grid on a1prime
        end
        PolicyValuesPath(1,:,:,:)=temp((vfoptions.ngridinterp+1)*(PolicyPath(1,:,:,:)-1)+PolicyPath(end-1,:,:,:)); % use fine index [end-1 is the L2 index; end is the PolicyL2flag, which is not part of the index]
    end
    if l_aprime>1
        if size(a_grid,2)==1 % stacked grid
            for ii=2:l_aprime
                PolicyValuesPath(ii,:,:,:)=a_grid(sum(n_aprime(1:ii-1))+PolicyPath(ii,:,:,:));
            end
        else % gridvals: column ii only changes value every prod(n_aprime(1:ii-1)) rows, so index the start of the relevant block (as linear index into column ii)
            for ii=2:l_aprime
                PolicyValuesPath(ii,:,:,:)=a_grid(1+prod(n_aprime(1:ii-1))*(PolicyPath(ii,:,:,:)-1)+size(a_grid,1)*(ii-1));
            end
        end
    end
else
    l_d=length(n_d);

    PolicyValuesPath=zeros(l_d+l_aprime,N_a,N_z,T,'gpuArray');
    if size(d_grid,2)==1
        PolicyValuesPath(1,:,:,:)=d_grid(PolicyPath(1,:,:,:));
        if l_d>1
            if l_d>2
                for ii=2:l_d-1
                    PolicyValuesPath(ii,:,:,:)=d_grid(sum(n_d(1:ii-1))+PolicyPath(ii,:,:,:));
                end
            end
            PolicyValuesPath(l_d,:,:,:)=d_grid(sum(n_d(1:end-1))+PolicyPath(l_d,:,:,:));
        end
    else % using d_gridvals (and must be that l_d>1)
        d_gridvals=gpuArray(d_grid);
        % PolicyPath holds per-variable d indexes; column ii of the gridvals only changes value every prod(n_d(1:ii-1)) rows, so index the start of the relevant block (as linear index into column ii)
        PolicyValuesPath(1,:,:,:)=d_gridvals(PolicyPath(1,:,:,:)); % linear indexing hits the first rows of column 1, which are the full first d grid
        for ii=2:l_d
            PolicyValuesPath(ii,:,:,:)=d_gridvals(1+prod(n_d(1:ii-1))*(PolicyPath(ii,:,:,:)-1)+size(d_gridvals,1)*(ii-1));
        end
    end

    if vfoptions.gridinterplayer==0
        PolicyValuesPath(l_d+1,:,:,:)=a_grid(PolicyPath(l_d+1,:,:,:));
    elseif vfoptions.gridinterplayer==1
        N_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
        if vfoptions.policyind2val_finegridinput==0
            temp=interp1(gpuArray(linspace(1,n_a(1),n_a(1)))',a_grid(1:n_a(1)),gpuArray(linspace(1,n_a(1),N_a1prime))'); % fine grid on a1prime
        elseif vfoptions.policyind2val_finegridinput==1
            temp=a_grid(1:N_a1prime); % a_grid input already contains the fine grid on a1prime
        end
        PolicyValuesPath(l_d+1,:,:,:)=temp((vfoptions.ngridinterp+1)*(PolicyPath(l_d+1,:,:,:)-1)+PolicyPath(end-1,:,:,:)); % use fine index [end-1 is the L2 index; end is the PolicyL2flag, which is not part of the index]
    end
    if l_aprime>1
        if size(a_grid,2)==1 % stacked grid
            for ii=2:l_aprime
                PolicyValuesPath(l_d+ii,:,:,:)=a_grid(sum(n_aprime(1:ii-1))+PolicyPath(l_d+ii,:,:,:));
            end
        else % gridvals: column ii only changes value every prod(n_aprime(1:ii-1)) rows, so index the start of the relevant block (as linear index into column ii)
            for ii=2:l_aprime
                PolicyValuesPath(l_d+ii,:,:,:)=a_grid(1+prod(n_aprime(1:ii-1))*(PolicyPath(l_d+ii,:,:,:)-1)+size(a_grid,1)*(ii-1));
            end
        end
    end
end

if outputkron==0
    PolicyValuesPath=reshape(PolicyValuesPath,[size(PolicyValuesPath,1),n_a,n_z,T]);
end



end