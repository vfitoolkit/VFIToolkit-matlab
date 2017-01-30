function StationaryDistKron=StationaryDist_Case2_Iteration_raw(StationaryDistKron,PolicyKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z,simoptions)
%Note: N_d is not actually needed, it is included to make it more like Case1 code.

% Options needed
%    simoptions.parallel
%    simoptions.maxit
%    simoptions.tolerance

if simoptions.parallel~=2
    %First, generate the transition matrix P=phi of Q (in the notation of SLP)
    P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
    if Case2_Type==1
        for z_c=1:N_z
            for a_c=1:N_a
                optd=PolicyKron(a_c,z_c);
                for zprime_c=1:N_z
                    optaprime=Phi_aprimeKron(optd,a_c,z_c,zprime_c);
                    P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
                end
            end
        end
    elseif Case2_Type==2
        for z_c=1:N_z % TRY AND TURN THESE TWO FOR LOOPS INTO A PARFOR AS IN TransitionPath_Case2 (does it improve run times???)
            for a_c=1:N_a
                optd=PolicyKron(a_c,z_c);
                for zprime_c=1:N_z
                    optaprime=Phi_aprimeKron(optd,z_c,zprime_c);
                    P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
                end
            end
        end
    elseif Case2_Type==3
        for z_c=1:N_z
            for a_c=1:N_a
                optd=PolicyKron(a_c,z_c);
                optaprime=Phi_aprimeKron(optd);
                for zprime_c=1:N_z
                    P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
                end
            end
        end
    end
    P=reshape(P,[N_a*N_z,N_a*N_z]);
    Ptran=P';
    
    SteadyStateDistKronOld=zeros(N_a*N_z,1);
    SScurrdist=max(max(abs(StationaryDistKron-SteadyStateDistKronOld)));
    SScounter=0;
    while SScurrdist>simoptions.tolerance && SScounter<simoptions.maxit
        SScurrdist=sum(abs(reshape(StationaryDistKron-SteadyStateDistKronOld, [N_a*N_z,1])));
        SteadyStateDistKronOld=StationaryDistKron;
        StationaryDistKron=Ptran*StationaryDistKron;
        
        SScounter=SScounter+1;
        if rem(SScounter,5000)==0
            SScounter
            SScurrdist
        end
    end
    
else % simoptions.parallel==2
	% First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    
    
    %optdprime=reshape(PolicyKron,[1,N_a*N_z]);
    if Case2_Type==1
        disp('ERROR: StationaryDist_Case2_Iteration_raw() not yet implemented for Case2_Type==1')
        % Create P matrix
    elseif Case2_Type==11
        disp('ERROR: StationaryDist_Case2_Iteration_raw() not yet implemented for Case2_Type==11')
        % Create P matrix
    elseif Case2_Type==12
        disp('ERROR: StationaryDist_Case2_Iteration_raw() not yet implemented for Case2_Type==12')
        % Create P matrix
    elseif Case2_Type==2
        % optaprime is here replaced by Phi_of_Policy, which is a different shape
        Phi_of_Policy=zeros(N_a,N_z,N_z,'gpuArray'); %a'(a,z',z)
        for z_c=1:N_z
            Phi_of_Policy(:,:,z_c)=Phi_aprimeKron(PolicyKron(:,z_c),:,z_c);
        end
%        Phi_aprimeKron % aprime(d,zprime,z)
        Ptemp=zeros(N_a,N_a*N_z*N_z,'gpuArray');
        Ptemp(reshape(permute(Phi_of_Policy,[2,1,3]),[1,N_a*N_z*N_z])+N_a*(gpuArray(0:1:N_a*N_z*N_z-1)))=1;
%        Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
        Ptran=kron(pi_z',ones(N_a,N_a,'gpuArray')).*reshape(Ptemp,[N_a*N_z,N_a*N_z]);
    end
    
    SteadyStateDistKronOld=zeros(N_a*N_z,1,'gpuArray');
    SScurrdist=sum(abs(StationaryDistKron-SteadyStateDistKronOld));
    SScounter=0;
    
    while SScurrdist>simoptions.tolerance && (100*SScounter)<simoptions.maxit
    
%         SteadyStateDistKronOld=SteadyStateDistKron;
%         SteadyStateDistKron=P*SteadyStateDistKron;
        for jj=1:100
            StationaryDistKron=Ptran*StationaryDistKron; %No point checking distance every single iteration. Do 100, then check.
        end
        
        SteadyStateDistKronOld=StationaryDistKron;
        StationaryDistKron=Ptran*StationaryDistKron;
        SScurrdist=sum(abs(StationaryDistKron-SteadyStateDistKronOld));
%         SScurrdist=sum(abs(reshape(SteadyStateDistKron-SteadyStateDistKronOld, [N_a*N_z,1])));
    
        SScounter=SScounter+1;
        if rem(SScounter,50)==0
            SScounter
            SScurrdist
        end
    end
end
    
if ~(SScounter<simoptions.maxit)
    disp('WARNING: SteadyState_Case2 stopped due to reacing simoptions.maxit, this might be causing a problem')
end 

% if simoptions-nagents~=0
%    SteadyStateDistKronTemp=reshape(SteadyStateDistKron, [N_a*N_z,1]);
%    SteadyStateDistKronTemp=cumsum(SteadyStateDistKronTemp);
%    SteadyStateDistKronTemp=SteadyStateDistKronTemp*simoptions.nagents;
%    SteadyStateDistKronTemp=round(SteadyStateDistKronTemp);
%    
%    SteadyStateDistKronTemp2=SteadyStateDistKronTemp;
%    for i=2:length(SteadyStateDistKronTemp)
%        SteadyStateDistKronTemp2(i)=SteadyStateDistKronTemp(i)-SteadyStateDistKronTemp(i-1);
%    end
%    SteadyStateDistKron=reshape(SteadyStateDistKronTemp2,[N_a,N_z]);
% end

end