function StdBusCycStats=StdBusCycStatsMatrix(LagsAndLeads, Ycycle, OtherCycles, Version)
% LagsAndLeads: number of lags and leads
% Ycycle: time series (as column vector) for the cycle of the reference variable (typically output)
% OtherCycles: matrix containing the times series for the cycles of the other variables (each time series as one column)

% Version: 1=just the leads and lags, 2=add first column with standard
% deviations, 3=add first column with standard deviations and second
% column with first-order autocorrelations

if nargin<4
    %If Version is not given, just use all the defaults
    Version=1;
end
    
NumVars=size(OtherCycles,2);

if Version==1

    StdBusCycStats=zeros(NumVars+1,2*LagsAndLeads+1);
%    StdBusCycStats(:,1)=[std(Ycycle); std(OtherCycles,[],1)'];
    StdBusCycStats(1,LagsAndLeads+1)=corr(Ycycle,Ycycle);
    for ii=1:(2*LagsAndLeads+1)
        jj=LagsAndLeads+1; %midpoint
        if ii<jj
            for kk=1:NumVars
                StdBusCycStats(kk+1,ii)=corr(Ycycle(jj-ii+1:end),OtherCycles(1:end-(jj-ii),kk));
            end
        elseif ii==jj
            for kk=1:NumVars
                StdBusCycStats(kk+1,ii)=corr(Ycycle,OtherCycles(:,kk));
            end
        elseif ii>jj
            for kk=1:NumVars
                StdBusCycStats(kk+1,ii)=corr(Ycycle(1:end-(ii-jj)),OtherCycles((ii-jj+1):end,kk));
            end
        end
    end

elseif Version==2
    
    StdBusCycStats=zeros(NumVars+1,1+2*LagsAndLeads+1);
    StdBusCycStats(:,1)=[std(Ycycle); std(OtherCycles,[],1)'];
    StdBusCycStats(1,1+LagsAndLeads+1)=corr(Ycycle,Ycycle);
    for ii=1:(2*LagsAndLeads+1)
        jj=LagsAndLeads+1; %midpoint
        if ii<jj
            for kk=1:NumVars
                StdBusCycStats(kk+1,ii+1)=corr(Ycycle(jj-ii+1:end),OtherCycles(1:end-(jj-ii),kk));
            end
        elseif ii==jj
            for kk=1:NumVars
                StdBusCycStats(kk+1,ii+1)=corr(Ycycle,OtherCycles(:,kk));
            end
        elseif ii>jj
            for kk=1:NumVars
                StdBusCycStats(kk+1,ii+1)=corr(Ycycle(1:end-(ii-jj)),OtherCycles((ii-jj+1):end,kk));
            end
        end
    end
    
elseif Version==3
    
    StdBusCycStats=zeros(NumVars+1,2+2*LagsAndLeads+1);
    StdBusCycStats(:,1)=[std(Ycycle); std(OtherCycles,[],1)'];
    temp=corrcoef(Ycycle(2:end),Ycycle(1:end-1));
    StdBusCycStats(1,2)=temp(1,2);
    for ii=1:NumVars
        temp=corrcoef(OtherCycles(2:end,ii),OtherCycles(1:end-1,ii));
        StdBusCycStats(ii+1,2)=temp(1,2);
    end
    StdBusCycStats(1,2+LagsAndLeads+1)=corr(Ycycle,Ycycle);
    for ii=1:(2*LagsAndLeads+1)
        jj=LagsAndLeads+1; %midpoint
        if ii<jj
            for kk=1:NumVars
                StdBusCycStats(kk+1,ii+2)=corr(Ycycle(jj-ii+1:end),OtherCycles(1:end-(jj-ii),kk));
            end
        elseif ii==jj
            for kk=1:NumVars
                StdBusCycStats(kk+1,ii+2)=corr(Ycycle,OtherCycles(:,kk));
            end
        elseif ii>jj
            for kk=1:NumVars
                StdBusCycStats(kk+1,ii+2)=corr(Ycycle(1:end-(ii-jj)),OtherCycles((ii-jj+1):end,kk));
            end
        end
    end

end



end
