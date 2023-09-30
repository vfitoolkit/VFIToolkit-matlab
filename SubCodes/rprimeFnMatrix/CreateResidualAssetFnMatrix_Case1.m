function [rprimeIndexes,rprimeProbs]=CreateResidualAssetFnMatrix_Case1(rprimeFn, n_d, n_a, n_r, n_z, d_grid, a_grid, r_grid, z_grid, rprimeFnParamsVec) % note, no n_r
% Note: rprimeIndex is [N_d*N_a*N_a*N_z,1], whereas rprimeProbs is [N_d*N_a*N_a*N_z,1]
%
% Creates the grid points and their 'interpolation' probabilities
% Note: rprimeIndexes is always the 'lower' point (the upper points are
% just rprimeIndexes+1, so no need to waste memory storing them), and the
% rprimeProbs are the probability of this lower point (prob of upper point
% is just 1 minus this).

% Because we have rprime(d,aprime,a,z) we are actually effectively just doing return fn, so simply redirect to there
rprimeVals=CreateReturnFnMatrix_Case1_Disc_Par2(rprimeFn, n_d, n_a,n_z, d_grid, a_grid, z_grid, rprimeFnParamsVec,0);

l_r=length(n_r);
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if N_d==0
    N_d=1; % Given what we use it for below we want to do this
end

%% Calcuate grid indexes and probs from the values
if l_r==1
    rprimeVals=reshape(rprimeVals,[1,N_d*N_a*N_a*N_z]);

    r_griddiff=r_grid(2:end)-r_grid(1:end-1); % Distance between point and the next point
    
    temp=r_grid-rprimeVals;
    temp(temp>0)=1; % Equals 1 when a_grid is greater than aprimeVals
    
    [~,rprimeIndexes]=max(temp,[],1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
    % Note, this is going to find the 'first' grid point such that aprimeVals is smaller than or equal to that grid point
    % This is the 'upper' grid point
        
    % Switch to lower grid point index
    rprimeIndexes=rprimeIndexes-1;
    rprimeIndexes(rprimeIndexes==0)=1;
        
    % Now, find the probabilities
    rprime_residual=rprimeVals'-r_grid(rprimeIndexes);
    % Probability of the 'lower' points
    rprimeProbs=1-rprime_residual./r_griddiff(rprimeIndexes);
        
    % Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
    offTopOfGrid=(rprimeVals>=r_grid(end));
    rprimeProbs(offTopOfGrid)=0;
    % Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
    offBottomOfGrid=(rprimeVals<=r_grid(1));
    rprimeProbs(offBottomOfGrid)=1;
    
    rprimeIndexes=rprimeIndexes';
    rprimeProbs=reshape(rprimeProbs,[N_d*N_a*N_a*N_z,1]); % This is probably just transpose??
end



end


