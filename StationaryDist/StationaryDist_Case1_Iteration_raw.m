function StationaryDistKron=StationaryDist_Case1_Iteration_raw(StationaryDistKron,PolicyIndexesKron,N_d,N_a,N_z,pi_z,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

%kstep tells the code to use the k-step transition matrix P^k, instead of
%P, when calculating the steady state distn
%kstep=100;
%THIS does not seem to be a good idea as it uses way to much memory and
%appears to in fact slow the code down.

if simoptions.parallel~=2
    %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
    P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
    for a_c=1:N_a
        for z_c=1:N_z
            if N_d==0 %length(n_d)==1 && n_d(1)==0
                optaprime=PolicyIndexesKron(a_c,z_c);
            else
                optaprime=PolicyIndexesKron(2,a_c,z_c);
            end
            for zprime_c=1:N_z
                P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
            end
        end
    end
    P=reshape(P,[N_a*N_z,N_a*N_z]);
    P=P';

    %SteadyStateDistKron=ones(N_a*N_z,1)/(N_a*N_z);
    SteadyStateDistKronOld=zeros(N_a*N_z,1);
    SScurrdist=sum(abs(StationaryDistKron-SteadyStateDistKronOld));
    SScounter=0;
    while SScurrdist>simoptions.tolerance && (100*SScounter)<simoptions.maxit
        
        for jj=1:100
            StationaryDistKron=P*StationaryDistKron; %No point checking distance every single iteration. Do 100, then check.
        end
        
        SteadyStateDistKronOld=StationaryDistKron;
        StationaryDistKron=P*StationaryDistKron;
        SScurrdist=sum(abs(StationaryDistKron-SteadyStateDistKronOld));
        
        SScounter=SScounter+1;
        if simoptions.verbose==1
            if rem(SScounter,50)==0
                SScounter
                SScurrdist
            end
        end
    end
elseif simoptions.parallel==2 % Using the GPU
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)

%     tic;
%    P=zeros(N_a,N_z,N_a,N_z,'gpuArray');

    if N_d==0 %length(n_d)==1 && n_d(1)==0
        optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
    else
        optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
    end
    Ptran=zeros(N_a,N_a*N_z,'gpuArray');
    Ptran(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
    Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptran));
%     timeP=toc

%     tic;
%     P=zeros(N_a,N_z,N_a,N_z,'gpuArray');
%     for a_c=1:N_a
%         for z_c=1:N_z
%             if N_d==0 %length(n_d)==1 && n_d(1)==0
%                 optaprime=PolicyIndexesKron(a_c,z_c);
%             else
%                 optaprime=PolicyIndexesKron(2,a_c,z_c);
%             end
%             for zprime_c=1:N_z
%                 P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%             end
%         end
%     end
%     P=reshape(P,[N_a*N_z,N_a*N_z]);
%     P=P';
%     timeP2=toc
    
%     tic;
    %SteadyStateDistKron=ones(N_a*N_z,1)/(N_a*N_z);
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
        if simoptions.verbose==1
            if rem(SScounter,50)==0
                SScounter
                SScurrdist
            end
        end
    end
%     time2=toc
end

if ~(SScounter<simoptions.maxit)
    disp('WARNING: SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end 

% if simoptions.nagents~=0
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
