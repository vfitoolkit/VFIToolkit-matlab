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

if size(znum,1)>1 && size(znum,2)==1
    znum=znum'; % make it a row vector
end
if size(Mew,2)==1 && size(Mew,1)>1
    Mew=Mew';
    % I lie about the input size for Mew, mvncdf() actually needs to
    % have it as a row vector, but I like it more as a column vector
end

l_z=length(znum);
if l_z>=5
    dbstack
    error('Have only coded for up to five dimesions, contact me if you need more')
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

% Preallocate for speed
if l_z==1
    P_part1=zeros(znum,1);
    P_part2=zeros(znum,1);    
else
    P=zeros(znum);
    P_part1=zeros(znum);
    P_part2=zeros(znum);    
end

% z1_gridspacing_down
% z1_gridspacing_up
% z2_gridspacing_down
% z2_gridspacing_up

% Test1=1; % This is hardcoded. I leave it as something you can change so you can see the difference (with l_z=2)
% % Now do the actual multivariate normal cdf calculation
% if l_z==1
%     for z1_c=1:znum(1)
%         P_part1(z1_c)=mvncdf(z_grid(z1_c,:)+z1_gridspacing_up(z1_c),Mew,Sigma);
%         P_part2(z1_c)=mvncdf(z_grid(z1_c,:)-z1_gridspacing_down(z1_c),Mew,Sigma);
%     end
%     P=P_part1-P_part2;
% elseif l_z==2
%     if Test1==1
%         for z1_c=1:znum(1)
%             for z2_c=1:znum(2)
%                 z_c=z1_c+znum(1)*(z2_c-1);
%                 P_part1(z1_c,z2_c)=mvncdf(z_gridvals(z_c,:)+[z1_gridspacing_up(z1_c),z2_gridspacing_up(z2_c)],Mew,Sigma);
%                 P_part2(z1_c,z2_c)=mvncdf(z_gridvals(z_c,:)-[z1_gridspacing_down(z1_c),z2_gridspacing_down(z2_c)],Mew,Sigma);
% 
%                 [z1_c,z2_c,z_c]
%                 z_gridvals(z_c,:)-[z1_gridspacing_down(z1_c),z2_gridspacing_down(z2_c)]
%             end
%         end
%         P_part1
%         P_part2
% 
%         P=P_part1-P_part2;
%     elseif Test1==2
%         for z1_c=1:znum(1)
%             for z2_c=1:znum(2)
%                 z_c=z1_c+znum(1)*(z2_c-1);
%                 P_part1a(z1_c,z2_c)=mvncdf(z_gridvals(z_c,:)+[z1_gridspacing_up(z1_c),0],Mew,Sigma);
%                 P_part2a(z1_c,z2_c)=mvncdf(z_gridvals(z_c,:)-[z1_gridspacing_down(z1_c),0],Mew,Sigma);
%                 P_part1b(z1_c,z2_c)=mvncdf(z_gridvals(z_c,:)+[0,z2_gridspacing_up(z2_c)],Mew,Sigma);
%                 P_part2b(z1_c,z2_c)=mvncdf(z_gridvals(z_c,:)-[0,z2_gridspacing_down(z2_c)],Mew,Sigma);
%             end
%         end
%         P=(P_part1a-P_part2a)+(P_part1b-P_part2b); % This seems likely to be inferior. I just try it out as HVY2011 codes do something along these lines (taking mid grid points in each dimension seperately)
%     end
% end


% Getting mvncdf() to work (unless I just ignore the correlations and use the marginals, which I don't 
% like the vibe of) doesn't seem to work. So I just take a lazy approach. Generate a whole bunch of
% random numbers, count them to nearest grid points, renormalize.
nsample=10^6; % I should make this an option. Note, when I set it to 10^7 I actually got an out-of-memmory error as 'sample' was too big to fit in memory
sample=mvnrnd(Mew,Sigma,nsample);

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


%%
if mvnoptions.parallel==2 
    P=gpuArray(P); %(z,zprime)
end


end