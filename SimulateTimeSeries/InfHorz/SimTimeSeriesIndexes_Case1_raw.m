function SimTimeSeriesKron=SimTimeSeriesIndexes_Case1_raw(PolicyKron,l_d,n_a,cumsum_pi_z,seedpoint,simoptions)

% burnin=simoptions.burnin;
% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

% Simulates path based on PolicyKron of length 'periods' after a burn
% in of length 'burnin' (burn-in are the initial run of points that are then
% dropped). The burn in begins at point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)

SimTimeSeriesKron=zeros(2,simoptions.simperiods);

currstate=seedpoint;
% cumsum_pi_z=cumsum(pi_z,2);

l_a=length(n_a);

if simoptions.gridinterplayer==0
    if l_d==0
        if simoptions.burnin>0
            for t=1:simoptions.burnin
                currstate(1)=PolicyKron(currstate(1),currstate(2));
                [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
        end
        for t=1:simoptions.simperiods
            SimTimeSeriesKron(1,t)=currstate(1); %a_c
            SimTimeSeriesKron(2,t)=currstate(2); %z_c

            currstate(1)=PolicyKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
    else
        if simoptions.burnin>0
            for t=1:simoptions.burnin
                currstate(1)=PolicyKron(2,currstate(1),currstate(2));
                [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
        end
        for t=1:simoptions.simperiods
            SimTimeSeriesKron(1,t)=currstate(1); %a_c
            SimTimeSeriesKron(2,t)=currstate(2); %z_c

            currstate(1)=PolicyKron(2,currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
    end
elseif simoptions.gridinterplayer==1

    PolicyKron(end,:,:)=(PolicyKron(end,:,:)-1)/(1+simoptions.ngridinterp); % convert into probability of upper grid point

    if l_d==0
        if l_a==1
            if simoptions.burnin>0
                for t=1:simoptions.burnin
                    loweraprime=PolicyKron(1,currstate(1),currstate(2));
                    aprimeprob=PolicyKron(end,currstate(1),currstate(2));
                    currstate(1)=loweraprime+(rand(1)<aprimeprob);
                    [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
                end
            end
            for t=1:simoptions.simperiods
                SimTimeSeriesKron(1,t)=currstate(1); %a_c
                SimTimeSeriesKron(2,t)=currstate(2); %z_c

                loweraprime=PolicyKron(1,currstate(1),currstate(2));
                aprimeprob=PolicyKron(end,currstate(1),currstate(2));
                currstate(1)=loweraprime+(rand(1)<aprimeprob);
                [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
        else % l_a>1
            N_a1=n_a(1);
            if simoptions.burnin>0
                for t=1:simoptions.burnin
                    loweraprime=PolicyKron(1,currstate(1),currstate(2));
                    aprimeprob=PolicyKron(end,currstate(1),currstate(2));
                    a2prime=PolicyKron(2,currstate(1),currstate(2));
                    a1prime=loweraprime+(rand(1)<aprimeprob);
                    currstate(1)=a1prime+N_a1*(a2prime-1);
                    [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
                end
            end
            for t=1:simoptions.simperiods
                SimTimeSeriesKron(1,t)=currstate(1); %a_c
                SimTimeSeriesKron(2,t)=currstate(2); %z_c

                loweraprime=PolicyKron(1,currstate(1),currstate(2));
                aprimeprob=PolicyKron(end,currstate(1),currstate(2));
                a2prime=PolicyKron(2,currstate(1),currstate(2));
                a1prime=loweraprime+(rand(1)<aprimeprob);
                currstate(1)=a1prime+N_a1*(a2prime-1);
                [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
        end
    else % l_d>0
        if l_a==1
            if simoptions.burnin>0
                for t=1:simoptions.burnin
                    loweraprime=PolicyKron(2,currstate(1),currstate(2));
                    aprimeprob=PolicyKron(end,currstate(1),currstate(2));
                    currstate(1)=loweraprime+(rand(1)<aprimeprob);
                    [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
                end
            end
            for t=1:simoptions.simperiods
                SimTimeSeriesKron(1,t)=currstate(1); %a_c
                SimTimeSeriesKron(2,t)=currstate(2); %z_c

                loweraprime=PolicyKron(2,currstate(1),currstate(2));
                aprimeprob=PolicyKron(end,currstate(1),currstate(2));
                currstate(1)=loweraprime+(rand(1)<aprimeprob);
                [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
        else % l_a>1
            N_a1=n_a(1);
            if simoptions.burnin>0
                for t=1:simoptions.burnin
                    loweraprime=PolicyKron(2,currstate(1),currstate(2));
                    aprimeprob=PolicyKron(end,currstate(1),currstate(2));
                    a2prime=PolicyKron(3,currstate(1),currstate(2));
                    a1prime=loweraprime+(rand(1)<aprimeprob);
                    currstate(1)=a1prime+N_a1*(a2prime-1);
                    [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
                end
            end
            for t=1:simoptions.simperiods
                SimTimeSeriesKron(1,t)=currstate(1); %a_c
                SimTimeSeriesKron(2,t)=currstate(2); %z_c

                loweraprime=PolicyKron(2,currstate(1),currstate(2));
                aprimeprob=PolicyKron(end,currstate(1),currstate(2));
                a2prime=PolicyKron(3,currstate(1),currstate(2));
                a1prime=loweraprime+(rand(1)<aprimeprob);
                currstate(1)=a1prime+N_a1*(a2prime-1);
                [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
        end
    end
end



end
