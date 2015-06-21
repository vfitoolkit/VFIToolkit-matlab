function filteredseries = LucasFilter(beta, timeseries)
% Can accept the time series as either a column or row vector, and returns
% the same.

T=length(timeseries);

alpha=((1-beta)^2)/(1-beta^2-2*(beta^((T+1/2)))*(1-beta));

%Apply the filter
filteredseries=zeros(size(timeseries));
for i=1:T
    for j=1:T
        filteredseries(i)=filteredseries(i)+alpha*(beta^(abs(j-i)))*timeseries(j);
    end
end


end