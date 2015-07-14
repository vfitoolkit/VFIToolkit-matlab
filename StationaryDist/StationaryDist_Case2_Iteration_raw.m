function SteadyStateDistKron=SteadyState_Case2_raw(SteadyStateDistKron,PolicyKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z,simoptions)
%If Nagents=0, then it will treat the agents as being on a continuum of
%weight 1.
%If Nagents is any other (integer), it will give the most likely of the
%distributions of that many agents across the various steady-states; this
%is for use with models that have a finite number of agents, rather than a
%continuum.
%Note: N_d is not actually needed, it is included to make it more like
%Case1 code.

if nargin<10
    simoptions.nagents=0;
    simoptions.maxit=5*10^4; %In my experience, after a simulation, if you need more that 5*10^4 iterations to reach the steady-state it is because something has gone wrong
%     Nagents=0;
else
    eval('fieldexists=1;simoptions.nagents;','fieldexists=0;')
    if fieldexists==0
        simoptions.nagents=0;
    end
    eval('fieldexists=1;simoptions.maxit;','fieldexists=0;')
    if fieldexists==0
        simoptions.maxit=5*10^4;
    end
end

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
        for z_c=1:N_z
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
    SScurrdist=max(max(abs(SteadyStateDistKron-SteadyStateDistKronOld)));
    SScounter=0;
    while SScurrdist>simoptions.tolerance && SScounter<simoptions.maxit
        SScurrdist=sum(abs(reshape(SteadyStateDistKron-SteadyStateDistKronOld, [N_a*N_z,1])));
        SteadyStateDistKronOld=SteadyStateDistKron;
        SteadyStateDistKron=Ptran*SteadyStateDistKron;
        
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
        disp('ERROR: SteadyState_Case2_raw() not yet implemented for Case2_Type==1')
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
    SScurrdist=sum(abs(SteadyStateDistKron-SteadyStateDistKronOld));
    SScounter=0;
    
    while SScurrdist>simoptions.tolerance && (100*SScounter)<simoptions.maxit
    
%         SteadyStateDistKronOld=SteadyStateDistKron;
%         SteadyStateDistKron=P*SteadyStateDistKron;
        for jj=1:100
            SteadyStateDistKron=Ptran*SteadyStateDistKron; %No point checking distance every single iteration. Do 100, then check.
        end
        
        SteadyStateDistKronOld=SteadyStateDistKron;
        SteadyStateDistKron=Ptran*SteadyStateDistKron;
        SScurrdist=sum(abs(SteadyStateDistKron-SteadyStateDistKronOld));
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