function [C,digestweights,qlimitvec]=createDigest(values, weights,delta,presorted)
% Creates a t-digest from the distribution.
% For explanation of t-digest:
%   Kirkby - Computing Quantiles of Functions of the Agent Distribution using t-Digests
%   Dunning & Ertl (2019) - Computing Extremely Accurate Quantiles Using t-digests
%
% Inputs:
%     values - a column vector of values/observations
%     weights - a column vector of corresponding weights
%     delta - scale fn parameter
% Optional Inputs:
%     presorted (default=0) - can set presorted=1 to skip the sorting step
%                            (reduces runtimes for when you already know that the input is sorted)
%
% Outputs:
%     C - the centroid means
%     digestweights - the centroid masses (the share of the total weight of the points in each centroid; they sum to one)
%     qlimitvec - the qlimit at which each centroid was closed (each centroid ends at the first point whose cumulative weight
%                 reaches its qlimit, so the cumulative masses overshoot qlimitvec; digestweights are not its differences)
%
% My implementation uses the k1() scaling function.
%
% Delta: scaling parameter, essentially the higher delta the more points used and so the more accurate.
% It is recommended to use delta=100, 1000, or 10000.

if nargin<4
    presorted=0;
end

% Before we start, just throw out all the points with weight of zero. [Helps make the rest faster]
temp=(weights~=0);
values=values(temp);
weights=weights(temp);

if presorted==0
    % Sort the values, use the same index to sort the weights
    [values,sortindex]=sort(values);
    weights=weights(sortindex);
end

cumweights=cumsum(weights);
totalweight=cumweights(end);
% Make sure the weights are normalized to one
cumweights=cumweights./totalweight;

% I hard-code the k1 function (see Dunning & Ertl (2019))
kfn=@(q,delta) (delta/(2*pi))*asin(2*q-1);
kinvfn=@(k,delta) (sin(k/(delta/(2*pi)))+1)/2;

% The sequence of qlimits does not depend on the data: each one is kinvfn(kfn(previous)+1,delta), starting from q0=0. Compute it
% up to the first that is within 1e-7 of one (where the loop over the points stops), or the last before it stops increasing
% (beyond the top of k1 the sequence turns back down; only for unusual delta).
qlimit=kinvfn(kfn(0,delta)+1,delta);
qlimitseq=qlimit;
while 1-qlimit>=10^(-7)
    qlimitnext=kinvfn(kfn(qlimit,delta)+1,delta);
    if ~(qlimitnext>qlimit)
        break
    end
    qlimit=qlimitnext;
    qlimitseq=[qlimitseq;qlimit]; %#ok<AGROW>
end
% The qlimits the vectorised part can reach: all of them except the one within 1e-7 of one (reaching it stops the loop)
Kp=length(qlimitseq);
if 1-qlimitseq(end)<10^(-7)
    Kp=Kp-1;
end

% A point ii closes a centroid when its cumulative weight reaches the current qlimit, and the next qlimit is then the first one
% above that cumulative weight. So, while the cumulative weight is below qlimitseq(Kp), a centroid ends at every point where the
% number of qlimits at or below the cumulative weight goes up, and the qlimit of each centroid is the first one above the end of
% the one before. The points from qlimitseq(Kp) on (the very top of the distribution) are done one at a time, as before.
ncw=length(cumweights);
if Kp>0
    p=find(cumweights(1:ncw-1)>=qlimitseq(Kp),1)-1; % the points done vectorised are 1:p
    if isempty(p)
        p=ncw-1;
    end
    p=gather(p);
else
    p=0;
end
if p>0
    nqbelow=discretize(cumweights(1:p),[-Inf;qlimitseq(1:Kp);Inf])-1; % the number of qlimits at or below the cumulative weight
    closes=find(diff([0;nqbelow(:)])>0); % the points that close a centroid (as columns: with p=1 a scalar indexed by 1:0 is 1x0, which a gpuArray vertcat refuses)
    nqbelow_closes=gather(nqbelow(closes));
    closes=gather(closes);
else
    closes=zeros(0,1);
    nqbelow_closes=zeros(0,1);
end

nclose=length(closes);
if nclose>0
    % The centroid of each point up to the last close (a new one starts after every close), and the sums over each centroid, all at once
    lastclose=closes(nclose);
    centroidid=zeros(lastclose,1,'like',weights);
    centroidid([1;reshape(closes(1:nclose-1),[],1)+1])=1;
    centroidid=cumsum(centroidid);
    mass=accumarray(centroidid,weights(1:lastclose),[nclose,1]);
    C=accumarray(centroidid,weights(1:lastclose).*values(1:lastclose),[nclose,1])./mass;
    digestweights=mass/totalweight; % the actual mass of each centroid
    qlimitvec=qlimitseq([0;reshape(nqbelow_closes(1:nclose-1),[],1)]+1); % the qlimit each centroid closed at: the first above the end of the one before
    ibegin=lastclose+1;
    nqbelow_prev=nqbelow_closes(nclose);
else
    C=zeros(0,1);
    qlimitvec=zeros(0,1);
    digestweights=zeros(0,1);
    ibegin=1;
    nqbelow_prev=0;
end
count=nclose+1;
qlimit=qlimitseq(nqbelow_prev+1);

% The top of the distribution, one point at a time
for ii=p+1:ncw-1
    q=cumweights(ii);
    %         if q<qlimit
    %             % Nothing, keep counting up the points as have not yet reached the next quantile
    if q>=qlimit
        % Passed qlimit, so store sigma, then create a new qlimit and reset sigma
        mass=sum(weights(ibegin:ii));
        C(count,1)=sum(weights(ibegin:ii).*values(ibegin:ii))/mass;
        digestweights(count,1)=mass/totalweight; % the actual mass of the centroid
        ibegin=ii+1;
        qlimitvec(count,1)=qlimit;
        count=count+1;
        qlimit=kinvfn(kfn(qlimit,delta)+1,delta);
        % Need to ensure that the new qlimit is actually relevant
        % (otherwise when we don't have much data the size of going
        % from one element of cumsortweights to next is larger than
        % step size of qlimit and so we have problems)
        while q>=qlimit && q<1 % Had to add q<1 due to errors at level of floating point accuracy
            qlimit=kinvfn(kfn(qlimit,delta)+1,delta);
        end
    end
    if 1-qlimit<10^(-7) % Accuracy of qlimit was getting rather ridiculous near 1, so just cut it off
        break
    end
end
ii=ncw; % Have to treat this separate as otherwise causes problems with q>qlimit never reached in the while statement
% Seem to get nan at the very top of the digests (on large models) so implemented the following if-statement as likely source was dividing by zero
if sum(weights(ibegin:ii))>0
    mass=sum(weights(ibegin:ii));
    C(count,1)=sum(weights(ibegin:ii).*values(ibegin:ii))/mass;
    digestweights(count,1)=mass/totalweight; % the actual mass of the centroid
    qlimitvec(count,1)=qlimit;
end


end
