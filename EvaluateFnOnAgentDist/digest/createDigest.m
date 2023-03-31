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
%     digestweights - the weights
%     qlimitvec - essentially the cumultive weights (note: digestweights are just the first difference of qlimitvec)
%
% My implementation uses the k1() scaling function.
%
% Delta: scaling parameter, essentially the higher delta the more points used and so the more accurate.
% It is recommended to use delta=100, 1000, or 10000 simply as these have been set up to preallocate memory so will be marginally faster.

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
% If the weights are not normalized to one, then do so.
S=cumweights(end);
if S~=1
    cumweights=cumweights./S;
end

% For some values of delta I calculated a rough upper limit on how many elements there will be in
% the t-digest so that the memory can be preallocated. This will be too
% large for smaller datasets, but helps speed the larger ones which is what matters.
Nq=0;
if delta==100
    Nq=51;
elseif delta==1000
    Nq=510;
elseif delta==10000
    Nq=5100;
end
% Note: in practice there will be a less points, so I trim the zeros at the end


% I hard-code the k1 function (see Dunning & Ertl (2019))
kfn=@(q,delta) (delta/(2*pi))*asin(2*q-1);
kinvfn=@(k,delta) (sin(k/(delta/(2*pi)))+1)/2;

if Nq~=0
    C=zeros(Nq,1);
    qlimitvec=zeros(Nq,1);
    q0=0;
    qlimit=kinvfn(kfn(q0,delta)+1,delta);
    ibegin=1;
    count=1;
    for ii=1:length(cumweights)-1
        q=cumweights(ii);
%         if q<qlimit
%             % Nothing, keep counting up the points as have not yet reached the next quantile
        if q>=qlimit
            % Passed qlimit, so store sigma, then create a new qlimit and reset sigma
            C(count)=sum(weights(ibegin:ii).*values(ibegin:ii))/sum(weights(ibegin:ii));
            ibegin=ii+1;
            qlimitvec(count)=qlimit;
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
    ii=length(cumweights); % Have to treat this seperate as otherwise causes problems with q>qlimit never reached in the while statement
    % Seem to get nan at the very top of the digests (on large models) so implemented the following if-statement as likely source was dividing by zero
    if sum(weights(ibegin:ii))>0
        C(count)=sum(weights(ibegin:ii).*values(ibegin:ii))/sum(weights(ibegin:ii));
        qlimitvec(count)=qlimit;
    end
    
    % Some elements near the end will be zeros, so find and trim these
    temp=~(qlimitvec==0);
    C=C(temp);
    qlimitvec=qlimitvec(temp);

    digestweights=[qlimitvec(2:end);1]-qlimitvec;

else % Have not precalculated number of elements, so memory usage not preallocated (so will be slower)
    C=[];
    qlimitvec=[];
    q0=0;
    qlimit=kinvfn(kfn(q0,delta)+1,delta);
    ibegin=1;
    for ii=1:length(cumweights)-1
        q=cumweights(ii);
%         if q<qlimit
%             % Nothing, keep counting up the points as have not yet reached the next quantile
        if q>=qlimit
            % Passed qlimit, so store sigma, then create a new qlimit and reset sigma
            C=[C;sum(weights(ibegin:ii).*values(ibegin:ii))/sum(weights(ibegin:ii))];
            ibegin=ii+1;
            qlimitvec=[qlimitvec;qlimit];
            qlimit=kinvfn(kfn(qlimit,delta)+1,delta);
            % Need to ensure that the new qlimit is actually relevant
            % (otherwise when we don't have much data the size of going
            % from one element of cumsortweights to next is larger than
            % step size of qlimit and so we have problems)
            while q>=qlimit && q<1
                qlimit=kinvfn(kfn(qlimit,delta)+1,delta); % Note: this is only really releant near the two ends (where the step size for qlimit becomes really small)
            end
        end
        if 1-qlimit<10^(-7) % Accuracy of qlimit was getting rather ridiculous near 1, so just cut it off
            break
        end
    end
    ii=length(cumweights); % Have to treat this seperate as otherwise causes problems with q>qlimit never reached in the while statement
    % Seem to get nan at the very top of the merged digests (on very large models) so implemented the following if-statement as likely source was dividing by zero
    if sum(weights(ibegin:ii))>0
        C=[C;sum(weights(ibegin:ii).*values(ibegin:ii))/sum(weights(ibegin:ii))];
        qlimitvec=[qlimitvec;qlimit];
    end
    
    digestweights=[qlimitvec(2:end);1]-qlimitvec;

end

end