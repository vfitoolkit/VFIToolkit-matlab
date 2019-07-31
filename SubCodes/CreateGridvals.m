function [d_gridvals, aprime_gridvals, a_gridvals, z_gridvals]=CreateGridvals(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,z_grid,Case1orCase2, MatrixOrCell)
% Creates the 'gridvals' versions of the standard grids. These allow for
% easier evaluation of functions on the grids via the EvalFnOnAgentDist
% commands.
% a_gridvals and z_gridvals are always returned.
% For Case1or2=1, aprime_gridvals is always returned, and d_gridvals is returned or equal to nan as appropriate based on n_d.
% For Case1or2=2, aprime_gridvals=nan, and d_gridvals is always returned.
% For MatrixOrCell=1, output takes form of matrices
% For MatrixOrCell=2, output takes form of cells.
%
% Gridvals contain N_a*N_z rows, and (e.g., for a_gridvals) the columns for a given row contain
% all the values of all the 'a' variables. (ie. a_gridvals is N_a*N_z-by-l_a)
% These contain no more information than the standard grid format (e.g.,
% a_grid), but are substantially larger (use more memory), however for
% certain purposes they are much easier to use quickly or in parallel.
%

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=reshape(PolicyIndexes,[size(PolicyIndexes,1),N_a,N_z]);

if l_d>0
    d_val=zeros(l_d,1);
end
aprime_val=zeros(l_a,1);

% First, create a_gridvals and z_gridvals, as these are always needed.
if MatrixOrCell==1
    z_gridvals=zeros(N_z,l_z);
    for i1=1:N_z
        sub=zeros(1,l_z);
        sub(1)=rem(i1-1,n_z(1))+1;
        for ii=2:l_z-1
            sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
        end
        sub(l_z)=ceil(i1/prod(n_z(1:l_z-1)));
        
        if l_z>1
            sub=sub+[0,cumsum(n_z(1:end-1))];
        end
        z_gridvals(i1,:)=z_grid(sub);
    end
    a_gridvals=zeros(N_a,l_a);
    for i2=1:N_a
        sub=zeros(1,l_a);
        sub(1)=rem(i2-1,n_a(1))+1;
        for ii=2:l_a-1
            sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
        end
        sub(l_a)=ceil(i2/prod(n_a(1:l_a-1)));
        
        if l_a>1
            sub=sub+[0,cumsum(n_a(1:end-1))];
        end
        a_gridvals(i2,:)=a_grid(sub);
    end
elseif MatrixOrCell==2
    z_gridvals=cell(N_z,l_z);
    for i1=1:N_z
        sub=zeros(1,l_z);
        sub(1)=rem(i1-1,n_z(1))+1;
        for ii=2:l_z-1
            sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
        end
        sub(l_z)=ceil(i1/prod(n_z(1:l_z-1)));
        
        if l_z>1
            sub=sub+[0,cumsum(n_z(1:end-1))];
        end
        z_gridvals(i1,:)=num2cell(z_grid(sub));
    end
    a_gridvals=cell(N_a,l_a);
    for i2=1:N_a
        sub=zeros(1,l_a);
        sub(1)=rem(i2-1,n_a(1))+1;
        for ii=2:l_a-1
            sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
        end
        sub(l_a)=ceil(i2/prod(n_a(1:l_a-1)));
        
        if l_a>1
            sub=sub+[0,cumsum(n_a(1:end-1))];
        end
        a_gridvals(i2,:)=num2cell(a_grid(sub));
    end   
end

% Now create those of d_gridvals and aprime_gridvals that are needed
% Check if doing Case1 or Case2, and if Case1, then check if need d_gridvals
if Case1orCase2==1
    if l_d>0
        if MatrixOrCell==1
            d_gridvals=zeros(N_a*N_z,l_d);
            aprime_gridvals=zeros(N_a*N_z,l_a);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                daprime_ind=PolicyIndexes(:,j1,j2);
                d_ind=daprime_ind(1:l_d);
                aprime_ind=daprime_ind((l_d+1):(l_d+l_a));
                for kk1=1:l_d
                    if kk1==1
                        d_val(kk1)=d_grid(d_ind(kk1));
                    else
                        d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                    end
                end
                for kk2=1:l_a
                    if kk2==1
                        aprime_val(kk2)=a_grid(aprime_ind(kk2));
                    else
                        aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                    end
                end
                d_gridvals(ii,:)=d_val;
                aprime_gridvals(ii,:)=aprime_val;
            end
        elseif MatrixOrCell==2
            d_gridvals=cell(N_a*N_z,l_d);
            aprime_gridvals=cell(N_a*N_z,l_a);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                daprime_ind=PolicyIndexes(:,j1,j2);
                d_ind=daprime_ind(1:l_d);
                aprime_ind=daprime_ind((l_d+1):(l_d+l_a));
                for kk1=1:l_d
                    if kk1==1
                        d_val(kk1)=d_grid(d_ind(kk1));
                    else
                        d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                    end
                end
                for kk2=1:l_a
                    if kk2==1
                        aprime_val(kk2)=a_grid(aprime_ind(kk2));
                    else
                        aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                    end
                end
                d_gridvals(ii,:)=num2cell(d_val);
                aprime_gridvals(ii,:)=num2cell(aprime_val);
            end
        end
    else % l_d==0
        if MatrixOrCell==1
            d_gridvals=nan;
            aprime_gridvals=zeros(N_a*N_z,l_a);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                daprime_ind=PolicyIndexes(:,j1,j2);
                aprime_ind=daprime_ind((l_d+1):(l_d+l_a));
                for kk2=1:l_a
                    if kk2==1
                        aprime_val(kk2)=a_grid(aprime_ind(kk2));
                    else
                        aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                    end
                end
                aprime_gridvals(ii,:)=aprime_val;
            end
        elseif MatrixOrCell==2
            d_gridvals=nan;
            aprime_gridvals=cell(N_a*N_z,l_a);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                daprime_ind=PolicyIndexes(:,j1,j2);
                aprime_ind=daprime_ind((l_d+1):(l_d+l_a));
                for kk2=1:l_a
                    if kk2==1
                        aprime_val(kk2)=a_grid(aprime_ind(kk2));
                    else
                        aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                    end
                end
                aprime_gridvals(ii,:)=num2cell(aprime_val);
            end
        end
    end
elseif Case1orCase2==2
    if MatrixOrCell==1
        d_gridvals=zeros(N_a*N_z,l_d);
        aprime_gridvals=nan;
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            daprime_ind=PolicyIndexes(:,j1,j2);
            d_ind=daprime_ind(1:l_d);
            for kk1=1:l_d
                if kk1==1
                    d_val(kk1)=d_grid(d_ind(kk1));
                else
                    d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                end
            end
            d_gridvals(ii,:)=d_val;
        end
    elseif MatrixOrCell==2
        d_gridvals=cell(N_a*N_z,l_d);
        aprime_gridvals=nan;
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            daprime_ind=PolicyIndexes(:,j1,j2);
            d_ind=daprime_ind(1:l_d);
            for kk1=1:l_d
                if kk1==1
                    d_val(kk1)=d_grid(d_ind(kk1));
                else
                    d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                end
            end
            d_gridvals(ii,:)=num2cell(d_val);
        end
    end
end


end
