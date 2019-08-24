function [dPolicy_gridvals, aprimePolicy_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_aprime,n_a,n_z,d_grid,aprime_grid,Case1orCase2, MatrixOrCell)
% Creates the 'gridvals' versions of the optimal policy. These allow for
% easier evaluation of functions on the grids via the EvalFnOnAgentDist
% commands.
% For Case1or2=1, aprime_gridvals is always returned, and d_gridvals is returned or equal to nan as appropriate based on n_d.
% For Case1or2=2, aprime_gridvals=nan, and d_gridvals is always returned.
% For MatrixOrCell=1, output takes form of matrices
% For MatrixOrCell=2, output takes form of cells.
%
% Gridvals contain N_a*N_z rows, and the columns for a given row contain
% all the values of all the 'a' variables. (ie. a_gridvals is N_a*N_z-by-l_a)
% These contain no more information than the standard grid format (e.g.,
% a_grid), but are substantially larger (use more memory), however for
% certain purposes they are much easier to use quickly or in parallel.
%
% If either of d or aprime is not relevant, then a value of nan will be returned for the corresponding gridvals output.

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_aprime=length(n_aprime);

N_a=prod(n_a);
N_z=prod(n_z);

% Create d_gridvals and aprime_gridvals as appropriate.
aprime_val=zeros(l_aprime,1);

% Now create those of d_gridvals and aprime_gridvals that are needed
% Check if doing Case1 or Case2, and if Case1, then check if need d_gridvals
if Case1orCase2==1
    PolicyIndexes=reshape(PolicyIndexes,[size(PolicyIndexes,1),N_a,N_z]);
    if l_d>0
        d_val=zeros(l_d,1);
        if MatrixOrCell==1
            if isa(d_grid, 'gpuArray')
                dPolicy_gridvals=zeros(N_a*N_z,l_d,'gpuArray');
            else
                dPolicy_gridvals=zeros(N_a*N_z,l_d);
            end
            if isa(aprime_grid, 'gpuArray')
                aprimePolicy_gridvals=zeros(N_a*N_z,l_aprime,'gpuArray');
            else
                aprimePolicy_gridvals=zeros(N_a*N_z,l_aprime);
            end
            
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                daprime_sub=PolicyIndexes(:,j1,j2);
                d_sub=daprime_sub(1:l_d);
                aprime_sub=daprime_sub((l_d+1):(l_d+l_aprime));
                for kk1=1:l_d
                    if kk1==1
                        d_val(kk1)=d_grid(d_sub(kk1));
                    else
                        d_val(kk1)=d_grid(d_sub(kk1)+sum(n_d(1:kk1-1)));
                    end
                end
                for kk2=1:l_aprime
                    if kk2==1
                        aprime_val(kk2)=aprime_grid(aprime_sub(kk2));
                    else
                        aprime_val(kk2)=aprime_grid(aprime_sub(kk2)+sum(n_aprime(1:kk2-1)));
                    end
                end
                dPolicy_gridvals(ii,:)=d_val;
                aprimePolicy_gridvals(ii,:)=aprime_val;
            end
        elseif MatrixOrCell==2
            dPolicy_gridvals=cell(N_a*N_z,l_d);
            aprimePolicy_gridvals=cell(N_a*N_z,l_aprime);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                daprime_sub=PolicyIndexes(:,j1,j2);
                d_sub=daprime_sub(1:l_d);
                aprime_sub=daprime_sub((l_d+1):(l_d+l_aprime));
                for kk1=1:l_d
                    if kk1==1
                        d_val(kk1)=d_grid(d_sub(kk1));
                    else
                        d_val(kk1)=d_grid(d_sub(kk1)+sum(n_d(1:kk1-1)));
                    end
                end
                for kk2=1:l_aprime
                    if kk2==1
                        aprime_val(kk2)=aprime_grid(aprime_sub(kk2));
                    else
                        aprime_val(kk2)=aprime_grid(aprime_sub(kk2)+sum(n_aprime(1:kk2-1)));
                    end
                end
                dPolicy_gridvals(ii,:)=num2cell(d_val);
                aprimePolicy_gridvals(ii,:)=num2cell(aprime_val);
            end
        end
    else % l_d==0
        if MatrixOrCell==1
            dPolicy_gridvals=nan;
            if isa(aprime_grid, 'gpuArray')
                aprimePolicy_gridvals=zeros(N_a*N_z,l_aprime,'gpuArray');
            else
                aprimePolicy_gridvals=zeros(N_a*N_z,l_aprime);
            end
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                daprime_sub=PolicyIndexes(:,j1,j2);
                aprime_sub=daprime_sub((l_d+1):(l_d+l_aprime));
                for kk2=1:l_aprime
                    if kk2==1
                        aprime_val(kk2)=aprime_grid(aprime_sub(kk2));
                    else
                        aprime_val(kk2)=aprime_grid(aprime_sub(kk2)+sum(n_aprime(1:kk2-1)));
                    end
                end
                aprimePolicy_gridvals(ii,:)=aprime_val;
            end
        elseif MatrixOrCell==2
            dPolicy_gridvals=nan;
            aprimePolicy_gridvals=cell(N_a*N_z,l_aprime);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                daprime_sub=PolicyIndexes(:,j1,j2);
                aprime_sub=daprime_sub((l_d+1):(l_d+l_aprime));
                for kk2=1:l_aprime
                    if kk2==1
                        aprime_val(kk2)=aprime_grid(aprime_sub(kk2));
                    else
                        aprime_val(kk2)=aprime_grid(aprime_sub(kk2)+sum(n_aprime(1:kk2-1)));
                    end
                end
                aprimePolicy_gridvals(ii,:)=num2cell(aprime_val);
            end
        end
    end
elseif Case1orCase2==2
%     PolicyIndexes=reshape(PolicyIndexes,[N_a,N_z]);
    d_val=zeros(l_d,1);
    if MatrixOrCell==1
        if isa(d_grid, 'gpuArray')
            dPolicy_gridvals=zeros(N_a*N_z,l_d,'gpuArray');
        else
            dPolicy_gridvals=zeros(N_a*N_z,l_d);
        end
        aprimePolicy_gridvals=nan;
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            d_ind=PolicyIndexes(j1,j2);
            d_sub=ind2sub_homemade_gpu(n_d,d_ind);
            for kk1=1:l_d
                if kk1==1
                    d_val(kk1)=d_grid(d_sub(kk1));
                else
                    d_val(kk1)=d_grid(d_sub(kk1)+sum(n_d(1:kk1-1)));
                end
            end
            dPolicy_gridvals(ii,:)=d_val;
        end
    elseif MatrixOrCell==2
        dPolicy_gridvals=cell(N_a*N_z,l_d);
        aprimePolicy_gridvals=nan;
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            d_ind=PolicyIndexes(j1,j2);
            d_sub=ind2sub_homemade_gpu(n_d,d_ind);
            for kk1=1:l_d
                if kk1==1
                    d_val(kk1)=d_grid(d_sub(kk1));
                else
                    d_val(kk1)=d_grid(d_sub(kk1)+sum(n_d(1:kk1-1)));
                end
            end
            dPolicy_gridvals(ii,:)=num2cell(d_val);
        end
    end
end


end
