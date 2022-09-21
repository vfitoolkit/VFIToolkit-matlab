function [C,digestweights,qlimitvec]=createDigest(values, weights,delta)
% Creates a t-digest from the distribution.
% For explanation of t-digest: 
%   Kirkby - 
%   Dunning & Ertl (2019) - Computing Extremely Accurate Quantiles Using t-digests
%
% trim: an optional input to trim any zeros from the tail
%   (I recommend trim=0 if you later plan to merge, otherwise trim=1)
%
% Outputs: 
%     C - the centroid means
%     digestweights - the weights
%     qlimitvec - essentially the cumultive weights (note: digestweights are just the first difference of qlimitvec)
%     Nq_trim - if trim=0, then this is the point that would be where it would get trimmed
%
% My implementation uses the k1() scaling function.
%
% Delta: scaling parameter, essentially the higher delta the more points used and so the more accurate.
% It is recommended to use delta=100, 1000, or 10000 simply as these have been set up to preallocate memory so will be marginally faster.

% If the weights are not normalized to one, then do so.
S=sum(weights);
if S~=1
    weights=weights./S;
end

[sortvalues,sortindex]=sort(values);
sortweights=weights(sortindex);

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
    cumsortweights=cumsum(sortweights);
    C=zeros(Nq,1);
    qlimitvec=zeros(Nq,1);
    q0=0;
    qlimit=kinvfn(kfn(q0,delta)+1,delta);
    ibegin=1;
    count=1;
    for ii=1:length(cumsortweights)-1
        q=cumsortweights(ii);
%         if q<qlimit
%             % Nothing, keep counting up the points as have not yet reached the next quantile
        if q>=qlimit
            % Passed qlimit, so store sigma, then create a new qlimit and reset sigma
            C(count)=sum(sortweights(ibegin:ii).*sortvalues(ibegin:ii))/sum(sortweights(ibegin:ii));
            ibegin=ii+1;
            qlimitvec(count)=qlimit;
            count=count+1;
            qlimit=kinvfn(kfn(qlimit,delta)+1,delta);
            % Need to ensure that the new qlimit is actually relevant
            % (otherwise when we don't have much data the size of going
            % from one element of cumsortweights to next is larger than
            % step size of qlimit and so we have problems)
            while q>qlimit && q<1 % Had to add q<1 due to errors at level of floating point accuracy
                qlimit=kinvfn(kfn(qlimit,delta)+1,delta);
            end
        end
        if 1-qlimit<10^(-7) % Accuracy of qlimit was getting rather ridiculous near 1, so just cut it off
            break
        end
    end
    ii=length(cumsortweights); % Have to treat this seperate as otherwise causes problems with q>qlimit never reached in the while statement
    % Seem to get nan at the very top of the digests (on large models) so implemented the following if-statement as likely source was dividing by zero
    if sum(sortweights(ibegin:ii))>0
        C(count)=sum(sortweights(ibegin:ii).*sortvalues(ibegin:ii))/sum(sortweights(ibegin:ii));
        qlimitvec(count)=qlimit;
    end

    % Some elements near the end will be zeros, so find and trim these
    temp=~(qlimitvec==0);
    C=C(temp);
    qlimitvec=qlimitvec(temp);

    digestweights=[qlimitvec(2:end);1]-qlimitvec;

else % Have not precalculated number of elements, so memory usage not preallocated (so will be slower)
    cumsortweights=cumsum(sortweights);
    C=[];
    qlimitvec=[];
    q0=0;
    qlimit=kinvfn(kfn(q0,delta)+1,delta);
    ibegin=1;
    for ii=1:length(cumsortweights)-1
        q=cumsortweights(ii);
%         if q<qlimit
%             % Nothing, keep counting up the points as have not yet reached the next quantile
        if q>=qlimit
            % Passed qlimit, so store sigma, then create a new qlimit and reset sigma
            C=[C;sum(sortweights(ibegin:ii).*sortvalues(ibegin:ii))/sum(sortweights(ibegin:ii))];
            ibegin=ii+1;
            qlimitvec=[qlimitvec;qlimit];
            qlimit=kinvfn(kfn(qlimit,delta)+1,delta);
            % Need to ensure that the new qlimit is actually relevant
            % (otherwise when we don't have much data the size of going
            % from one element of cumsortweights to next is larger than
            % step size of qlimit and so we have problems)
            while q>qlimit && q<1
                qlimit=kinvfn(kfn(qlimit,delta)+1,delta); % Note: this is only really releant near the two ends (where the step size for qlimit becomes really small)
            end
        end
        if 1-qlimit<10^(-7) % Accuracy of qlimit was getting rather ridiculous near 1, so just cut it off
            break
        end
    end
    ii=length(cumsortweights); % Have to treat this seperate as otherwise causes problems with q>qlimit never reached in the while statement
    % Seem to get nan at the very top of the merged digests (on very large models) so implemented the following if-statement as likely source was dividing by zero
    if sum(sortweights(ibegin:ii))>0
        C=[C;sum(sortweights(ibegin:ii).*sortvalues(ibegin:ii))/sum(sortweights(ibegin:ii))];
        qlimitvec=[qlimitvec;qlimit];
    end
    
    digestweights=[qlimitvec(2:end);1]-qlimitvec;

end

end