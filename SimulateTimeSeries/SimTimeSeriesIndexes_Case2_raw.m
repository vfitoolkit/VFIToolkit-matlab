function SteadyStateDistKron=SimTimeSeriesIndexes_Case2_raw(seedpoint,periods, burnin, PolicyIndexesKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z)
%Simulates a path based on PolicyIndexes (and Phi_aprime) of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)
%Note: N_d is not actually needed, it is just left in so inputs are more
%like those for Case1

%First, generate the transition matrix P=phi of Q (in the notation of SLP)
P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
if Case2_Type==1
    for z_c=1:N_z
        for a_c=1:N_a
            optd=PolicyIndexesKron(a_c,z_c);
            for zprime_c=1:N_z
                optaprime=Phi_aprimeKron(optd,a_c,z_c,zprime_c);
                P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
            end
        end
    end
elseif Case2_Type==2
    for z_c=1:N_z
        for a_c=1:N_a
            optd=PolicyIndexesKron(a_c,z_c);
            for zprime_c=1:N_z
                optaprime=Phi_aprimeKron(optd,z_c,zprime_c);
                P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
            end
        end
    end
elseif Case2_Type==3
    for z_c=1:N_z
        for a_c=1:N_a
            optd=PolicyIndexesKron(a_c,z_c);
            optaprime=Phi_aprimeKron(optd);
            for zprime_c=1:N_z
                P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
            end
        end
    end
end
P=reshape(P,[N_a*N_z,N_a*N_z]);
%Now turn P into a cumulative distn
P=cumsum(P,2);

SimTimeSeriesKron=zeros(2,periods);

currstate=sub2ind_homemade([N_a,N_z],seedpoint); 

for t=1:burnin
    [~,currstate]=max(P(currstate,:)>rand(1,1));
end
for t=1:periods
    temp=ind2sub_homemade([N_a,N_z], currstate);
    SimTimeSeriesKron(1,t)=temp(1); %a_c
    SimTimeSeriesKron(2,t)=temp(2); %z_c
    [~,currstate]=max(P(currstate,:)>rand(1,1));
end

end