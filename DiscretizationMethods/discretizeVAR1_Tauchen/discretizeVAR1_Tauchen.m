function [z_grid, P]=discretizeVAR1_Tauchen(Mew,Rho,Sigma,znum,Tauchen_q, tauchenoptions)
% Create states vector and transition matrix for the discrete markov process approximation of 
% M-variable VAR(1) process:
%      z'=mew+rho*z+e, e~N(0,sigmasq), by Tauchens method
% (Abuse of notation: sigmasq in codes is a column vector, in the VAR notation it is a matrix in which all of the non-diagonal elements are zeros) 
%
% Inputs
%   Mew          - (M x 1) constant vector
%   Rho          - (M x M) matrix of impact coefficients
%   Sigma        - M-by-1 column vector [Is assumed that covariances are zero].
%                - OR (M x M) variance-covariance matrix of the innovations
%   znum         - Desired number of discrete points in each dimension
%   Tauchen_q    - M-by-1 column vector, or scalar; (Hyperparameter) Defines max/min grid points as mew+-sigma*sigmaz (I suggest 2 or 3)
%   znum         - M-by-1 vector, number of states in discretization for each variable in z (must be an odd number)
% Optional inputs (tauchenoptions)
%   parallel: set equal to 2 to use GPU, 0 to use CPU
% Outputs
%   z_grid         - sum(znum)-by-1 column vector; stacked column vector containing the sum(znum) states of 
%                    the discrete approximation of z for each of the n variables
%   P              - prod(znum)-by-prod(znum) matrix; transition matrix of the discrete approximation of z;
%                    transmatrix(i,j) is the probability of transitioning from state i to state j
%
% Thanks to Iskander Karibzhanov who let me borrow heavily from his codes for this.
%%%%%%%%%%%%%%%
% Original paper:
% Tauchen (1986) - "Finite state Markov-chain approximations to univariate and vector autoregressions"

if exist('tauchenoptions','var')==0
    % Recommended choice for Parallel is 2 (on GPU). It is substantially faster (albeit only for very large grids; for small grids cpu is just as fast)
    tauchenoptions.parallel=1+(gpuDeviceCount>0);
    tauchenoptions.verbose=1;
else
    %Check tauchenoptions for missing fields, if there are some fill them with the defaults
    if isfield(tauchenoptions,'parallel')==0
        tauchenoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(tauchenoptions,'verbose')==0
        tauchenoptions.verbose=1;
    end
end

if size(znum,2)>1 && size(znum,1)==1
    znum=znum';
end
if size(Tauchen_q,2)>1 && size(Tauchen_q,1)==1
    Tauchen_q=Tauchen_q';
end
if isscalar(Tauchen_q)
    Tauchen_q=Tauchen_q*ones(size(znum));
end

% Below is largely just a copy of the code of Iskander Karibzhanov. Main difference is
% just the shape of the output of the 'state' variable and transition matrix (mine is transpose of his;
% I follow the economics/maths standard of expressing transition matrix as (z,zprime), he follows 
% computer science/algorithms standard of (zprime,z)). The
% other noteworthy difference from original is that his used dlyap which requires 
% the Control System Toolbox (which I avoid by using 'substitutefor_dlyap').

cumprod_znum=cumprod(znum);
cumsum_znum=cumsum(znum);
M=length(znum); % number of variables in VAR
prod_znum=prod(znum); % number of states in Markov chain

%% Sigma can be a diagonal matrix or column vector. Or Sigma can be a matrix.
% these two cases are treated seperately.

if size(Sigma,1)==size(Sigma,2) % Matrix
    if isdiag(Sigma)
        diagonalSigma=1;
        % Turn it into column vector for use below
        temp=Sigma;
        Sigma=zeros(size(Sigma,1),1);
        for z_c=1:length(Sigma)
            Sigma(z_c)=temp(z_c,z_c);
        end
    else
        diagonalSigma=0;
        % When using diagonalSigma=0 the code assumes an odd number of grid points for each dimension
        if any((mod(znum,2)==0)) % if any of znum are even
            error('When discretizing VAR with Tauchen and a Variance-Covariance matrix which is not diagonal you must have an odd number of grid points in each dimension')
        end
    end
else % it is a column vector
    diagonalSigma=1;
end

if diagonalSigma==0 && tauchenoptions.verbose==1
    fprintf('Comment on discretizeVAR1_Tauchen: When using non-diagonal variance-covariance matrix \n')
    fprintf('                                   it is actually a refinement of Tauchen rather than the original \n')
    % Tauchen: originally uses the marginal distributions and then
    % multiplies them all up together under the assumption that the
    % variance-covariance matrix is diagonal. The refinement here is just
    % to use the multivariate normal cdf.
end

if diagonalSigma==1
    q_sigmaz=real(Tauchen_q.*sqrt(diag(substitutefor_dlyap(Rho,diag(Sigma))))); % std.dev. of z
    zgrids=cell(M,1); z=nan(M,prod_znum);
    for i=1:M
        if znum(i)>1
            zgrids{i}=linspace(-1,1,znum(i))*q_sigmaz(i);
        else
            zgrids{i}=0;
        end
        z(i,:)=reshape(repmat(zgrids{i},cumprod_znum(i)/znum(i),prod_znum/cumprod_znum(i)),1,prod_znum);
    end
    q_sigmaz=q_sigmaz./(znum(:)-1); P=1;
    for i=1:M
        h=normcdf(bsxfun(@minus,zgrids{i}'+q_sigmaz(i),Rho(i,:)*z)/Sigma(i));
        h(znum(i),:)=1; h=permute([h(1,:);diff(h,1,1)],[3 1 2]);
        P=reshape(repmat(h,cumprod_znum(i)/znum(i),prod_znum/cumprod_znum(i)),prod_znum,prod_znum).*P;
    end
elseif diagonalSigma==0
    q_sigmaz=real(Tauchen_q.*sqrt(diag(substitutefor_dlyap(Rho,Sigma)))); % std.dev. of z
    omega=q_sigmaz./((znum-1)/2); % The spacing between grid points
    
    z_grid=zeros(sum(znum),1); % Stacked column vector
    z_grid(1:cumsum_znum(1))=linspace(-1,1,znum(1))*q_sigmaz(1); % ii=1
    for z_c=2:M
        z_grid(cumsum_znum(z_c-1)+1:cumsum_znum(z_c))=linspace(-1,1,znum(z_c))*q_sigmaz(z_c); % For the ii-th dimension, points evenly spaced from -q_sigmaz(ii) to +q_sigmaz(ii)
    end
    z_gridvals_trans=CreateGridvals(znum,z_grid,1)';
        
    conditional_mew=zeros(M,1);    
    P=zeros(prod_znum,prod_znum);
    P_part1=zeros(prod_znum,prod_znum);
    P_part2=zeros(prod_znum,prod_znum);
    
    for z_c=1:prod_znum
        for zprime_c=1:prod_znum
            P_part1(z_c,zprime_c)=mvncdf(z_gridvals_trans(:,zprime_c)+omega/2-Rho*z_gridvals_trans(:,z_c),conditional_mew,Sigma);
            P_part2(z_c,zprime_c)=mvncdf(z_gridvals_trans(:,zprime_c)-omega/2-Rho*z_gridvals_trans(:,z_c),conditional_mew,Sigma);
        end
    end
    
%     for z_c=1:prod_znum
%         z_sub=ind2sub_homemade(znum,z_c);
%         for zprime_c=1:prod_znum
%             zprime_sub=ind2sub_homemade(znum,zprime_c);
%             if zprime_sub(1)==1
%                 P(:,zprime_c)=P_part1(:,1);
%             end
%         end
%     end


%     P=P_part1-P_part2;
%     P(:,1)=P_part1(:,1);
%     P(:,znum)=1-P_part2(:,znum);  
end


%% Add in the mean (Mew) and put z_grid into appropriate shape.
%z=bsxfun(@plus,z,((eye(n,n)-rho)^(-1))*mew); % This is the form in which
%Iskander's codes output state, as a n-by-prod(znum) matrix, rather than
%the stacked column vector form that I use.
zmean=((eye(M,M)-Rho)^(-1))*Mew;
if diagonalSigma==1
    z_grid=zgrids{1}'+ones(znum(1),1)*zmean(1);
    for i=2:M
        z_grid=[z_grid;zgrids{i}'+ones(znum(i),1)*zmean(i)];
    end
elseif diagonalSigma==0
    % z_grid is already in right format, just add zmean
    z_grid(1:cumsum_znum(1))=z_grid(1:cumsum_znum(1))+zmean(1)*ones(znum(1),1); % ii=1
    for z_c=2:M
        z_grid(cumsum_znum(z_c-1)+1:cumsum_znum(z_c))=z_grid(cumsum_znum(z_c-1)+1:cumsum_znum(z_c))+zmean(z_c)*ones(znum(z_c),1); % ii=1
    end
end

%%
if tauchenoptions.parallel==2 
    z_grid=gpuArray(z_grid);
    P=gpuArray(P); %(z,zprime)
end
P=P'; %(z,zprime)

end
