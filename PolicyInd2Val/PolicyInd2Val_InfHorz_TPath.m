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
if N_z==0 % no z (and no e): the exogenous dimension is a singleton, n_z=0 would make every reshape below empty
    N_z=1;
end
% Users pass n_z (the markov z only), while semiz and e are in vfoptions, so the exogenous dimension of
% PolicyPath can be bigger than prod(n_z). Internal callers instead pass all of (semiz,z,e) as n_z. Either way
% everything after a is one exogenous dimension here, so take its size from PolicyPath itself.
N_ze=numel(PolicyPath)/(size(PolicyPath,1)*N_a*T);
n_zeout=n_z; % shape of the exogenous dimensions in the (non-kron) output
if prod(n_zeout)==0
    n_zeout=1; % no exogenous states at all
end
if N_ze~=N_z
    n_zeout=[];
    if isfield(vfoptions,'n_semiz') && prod(vfoptions.n_semiz)>0
        n_zeout=[n_zeout,vfoptions.n_semiz];
    end
    if prod(n_z)>0
        n_zeout=[n_zeout,n_z];
    end
    if isfield(vfoptions,'n_e') && prod(vfoptions.n_e)>0
        n_zeout=[n_zeout,vfoptions.n_e];
    end
    if prod(n_zeout)~=N_ze
        n_zeout=N_ze; % not recognised, so leave the exogenous states as one dimension
    end
    N_z=N_ze;
end

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
        else % gridvals
            % Joint row index, as for the d gridvals above: correct for a tensor-product
            % expansion and for a genuinely joint a_grid alike. Row 1 is already set above
            % (via the fine grid when gridinterplayer==1), so only ii>=2 is filled here.
            n_aprime_cumprod=cumprod(n_aprime);
            ajointindex=sum([1;n_aprime_cumprod(1:end-1)'].*(reshape(PolicyPath(1:l_aprime,:,:,:),[l_aprime,N_a*N_z*T])-[0;ones(l_aprime-1,1)]),1);
            for ii=2:l_aprime
                PolicyValuesPath(ii,:,:,:)=reshape(a_grid(ajointindex,ii),[1,N_a,N_z,T]);
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
        % Rebuild the joint row index from the per-variable d indexes, then read each column of
        % the gridvals at that row. This is correct both for a tensor-product expansion of a
        % stacked grid and for a genuinely joint d_grid, whose rows are an arbitrary restricted
        % set of combinations (e.g. n_d=[N,1] with an N-by-l_d grid). Indexing column ii by a
        % per-variable index instead assumes the tensor-product layout and silently returns the
        % wrong value on a joint grid.
        n_d_cumprod=cumprod(n_d);
        djointindex=sum([1;n_d_cumprod(1:end-1)'].*(reshape(PolicyPath(1:l_d,:,:,:),[l_d,N_a*N_z*T])-[0;ones(l_d-1,1)]),1);
        for ii=1:l_d
            PolicyValuesPath(ii,:,:,:)=reshape(d_gridvals(djointindex,ii),[1,N_a,N_z,T]);
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
        else % gridvals
            % Joint row index, as for the d gridvals above: correct for a tensor-product
            % expansion and for a genuinely joint a_grid alike. Row 1 is already set above
            % (via the fine grid when gridinterplayer==1), so only ii>=2 is filled here.
            n_aprime_cumprod=cumprod(n_aprime);
            ajointindex=sum([1;n_aprime_cumprod(1:end-1)'].*(reshape(PolicyPath(l_d+1:l_d+l_aprime,:,:,:),[l_aprime,N_a*N_z*T])-[0;ones(l_aprime-1,1)]),1);
            for ii=2:l_aprime
                PolicyValuesPath(l_d+ii,:,:,:)=reshape(a_grid(ajointindex,ii),[1,N_a,N_z,T]);
            end
        end
    end
end

if outputkron==0
    PolicyValuesPath=reshape(PolicyValuesPath,[size(PolicyValuesPath,1),n_a,n_zeout,T]);
end



end