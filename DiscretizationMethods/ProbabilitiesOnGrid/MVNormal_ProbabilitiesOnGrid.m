function P=MVNormal_ProbabilitiesOnGrid(z_grid,Mew, Sigma, znum, mvnoptions)
% Given the grid Z_grid, generate probabilities on this grid to approximate
% a multivariate-normal distribution with mean Mew, and Variance-Covariance
% matrix Sigma.
%
% Z~N(Mew, Sigma), Z is M-dimensional
%
% Inputs:
%    znum      - 1-by-M, the number of grid points in each dimension
%    z_grid    - sum(znum)-by-1, a stacked grid 
%    Mew       - M-by-1 vector of means
%    Sigma     - M-by-M variance-covariance matrix
%               OR M-by-1, intrepreted as a diagonal var-covar matrix with zeros on the off-diagonals
%
% Optional inputs (mvnoptions)
%   parallel: set equal to 2 to use GPU, 0 to use CPU

if exist('mvnoptions','var')==0
    % Recommended choice for Parallel is 2 (on GPU). It is substantially faster (albeit only for very large grids; for small grids cpu is just as fast)
    mvnoptions.parallel=1+(gpuDeviceCount>0);
    mvnoptions.verbose=1;
else
    %Check mvnoptions for missing fields, if there are some fill them with the defaults
    if isfield(mvnoptions,'parallel')==0
        mvnoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(mvnoptions,'verbose')==0
        mvnoptions.verbose=1;
    end
end

bruteforce=0; % use mvncdf() [bruteforce=1 is just legacy code that acts as a double-check]

if size(znum,1)>1 && size(znum,2)==1
    znum=znum'; % make it a row vector
end
if size(Mew,2)==1 && size(Mew,1)>1
    Mew=Mew';
    % I lie about the input size for Mew, mvncdf() actually needs to
    % have it as a row vector, but toolkit likes it as a column vector
end

l_z=length(znum);
if l_z>=5
    dbstack
    error('Have only coded for up to five dimensions, contact me if you need more')
end

%% Sigma can be a diagonal matrix or column vector. Or Sigma can be a matrix.
% these two cases are treated seperately.

if size(Sigma,1)==1 || size(Sigma,2)==1
    % vector, so interpret as the diagonals
    Sigma=diag(Sigma); % Turn into full var-covar matrix, makes things easier.
end

l_z=length(znum);
if all(size(z_grid)==[sum(znum),1]) % stacked column vector
    z_gridvals=CreateGridvals(znum,z_grid,1);
else
    error('z_grid is the wrong size [input to MVNormal_ProbabilitiesOnGrid()]')
end

if l_z>=1
    z1_grid=z_grid(1:znum(1));
    z1_gridspacing_up=[(z1_grid(2:end)-z1_grid(1:end-1))/2; Inf];
    z1_gridspacing_down=[Inf; (z1_grid(2:end)-z1_grid(1:end-1))/2]; % Note: will be subtracted from grid point, hence Inf, not -Inf
    if l_z>=2
        z2_grid=z_grid(znum(1)+1:sum(znum(1:2)));
        z2_gridspacing_up=[(z2_grid(2:end)-z2_grid(1:end-1))/2; Inf];
        z2_gridspacing_down=[Inf; (z2_grid(2:end)-z2_grid(1:end-1))/2];
        if l_z>=3
            z3_grid=z_grid(sum(znum(1:2))+1:sum(znum(1:3)));
            z3_gridspacing_up=[(z3_grid(2:end)-z3_grid(1:end-1))/2; Inf];
            z3_gridspacing_down=[Inf; (z3_grid(2:end)-z3_grid(1:end-1))/2];
            if l_z>=4
                z4_grid=z_grid(sum(znum(1:3))+1:sum(znum(1:4)));
                z4_gridspacing_up=[(z4_grid(2:end)-z4_grid(1:end-1))/2; Inf];
                z4_gridspacing_down=[Inf; (z4_grid(2:end)-z4_grid(1:end-1))/2];
                if l_z>=5
                    z5_grid=z_grid(sum(znum(1:4))+1:sum(znum(1:5)));
                    z5_gridspacing_up=[(z5_grid(2:end)-z5_grid(1:end-1))/2; Inf];
                    z5_gridspacing_down=[Inf; (z5_grid(2:end)-z5_grid(1:end-1))/2];
                end
            end
        end
    end
end



% z1_gridspacing_down
% z1_gridspacing_up
% z2_gridspacing_down
% z2_gridspacing_up

if bruteforce==0
    %% Now do the actual multivariate normal cdf calculation
    if l_z==1
        z_gridspacing_up=z1_gridspacing_up;
        z_gridspacing_down=z1_gridspacing_down;
    elseif l_z==2
        z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up],1);
        z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down],1);
    elseif l_z==3
        z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up;z3_gridspacing_up],1);
        z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down;z3_gridspacing_down],1);
    elseif l_z==4
        z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up;z3_gridspacing_up;z4_gridspacing_up],1);
        z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down;z3_gridspacing_down;z4_gridspacing_down],1);
    elseif l_z==5
        z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up;z3_gridspacing_up;z4_gridspacing_up;z5_gridspacing_up],1);
        z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down;z3_gridspacing_down;z4_gridspacing_down;z5_gridspacing_down],1);
    end

    if l_z==1
        P = normcdf(z_gridvals+z_gridspacing_up,Mew,Sigma)-normcdf(z_gridvals-z_gridspacing_down,Mew,Sigma);
    else
        P=reshape(mvncdf(z_gridvals-z_gridspacing_down,z_gridvals+z_gridspacing_up,Mew,Sigma),znum);
    end

else 
    %% bruteforce=1
    % This bruteforce is really just left here for legacy reasons (and so
    % you can see that simulating points and then count onto grid and then
    % normalize will give the same answer, which acts as a double-check on the
    % above)

    nsample=10^6; % I should make this an option. Note, when I set it to 10^7 I actually got an out-of-memmory error as 'sample' was too big to fit in memory
    sample=mvnrnd(Mew,Sigma,nsample);

    % Preallocate for speed
    P=zeros(znum);

    if l_z==2

        n1vec=sample(:,1);
        temp1=(n1vec>(z1_grid'-z1_gridspacing_down')); % gives a big matrix
        temp2=(n1vec<(z1_grid'+z1_gridspacing_up')); % gives a big matrix
        [~,ind1]=max(temp1.*temp2,[],2);

        n2vec=sample(:,2);
        temp1=(n2vec>(z2_grid'-z2_gridspacing_down')); % gives a big matrix
        temp2=(n2vec<(z2_grid'+z2_gridspacing_up')); % gives a big matrix
        [~,ind2]=max(temp1.*temp2,[],2);

        for ii=1:nsample
            P(ind1(ii),ind2(ii))=P(ind1(ii),ind2(ii))+1;
        end

    elseif l_z==3

        n1vec=sample(:,1);
        temp1=(n1vec>(z1_grid'-z1_gridspacing_down')); % gives a big matrix
        temp2=(n1vec<(z1_grid'+z1_gridspacing_up')); % gives a big matrix
        [~,ind1]=max(temp1.*temp2,[],2);

        n2vec=sample(:,2);
        temp1=(n2vec>(z2_grid'-z2_gridspacing_down')); % gives a big matrix
        temp2=(n2vec<(z2_grid'+z2_gridspacing_up')); % gives a big matrix
        [~,ind2]=max(temp1.*temp2,[],2);

        n3vec=sample(:,3);
        temp1=(n3vec>(z3_grid'-z3_gridspacing_down')); % gives a big matrix
        temp2=(n3vec<(z3_grid'+z3_gridspacing_up')); % gives a big matrix
        [~,ind3]=max(temp1.*temp2,[],2);

        for ii=1:nsample
            P(ind1(ii),ind2(ii),ind3(ii))=P(ind1(ii),ind2(ii),ind3(ii))+1;
        end

    end
    % Now just normalize
    P=P/sum(P(:));

end

%%
if mvnoptions.parallel==2 
    P=gpuArray(P); %(z,zprime)
end


end