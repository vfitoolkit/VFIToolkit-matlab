function StationaryDistKron=StationaryDist_Case1_Iteration_EntryExit2_raw(StationaryDistKron,PolicyIndexesKron,N_d,N_a,N_z,pi_z, ExitProb, EntryDist, simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

%kstep tells the code to use the k-step transition matrix P^k, instead of
%P, when calculating the steady state distn
%kstep=100;
%THIS does not seem to be a good idea as it uses way to much memory and
%appears to in fact slow the code down. NOTE: this is no longer used
%anywhere in code, I leave it here as a reminder that I tried this and it
%did not work well.

% Note that EntryDist is of size N_a*N_z-by-1. So is already like 1 column
% of Ptranspose.

if simoptions.parallel<2
%     % The following commented out version was producing machine precision
%     % level errors in Ptranspose that lead to machine precision level
%     % errors in the individual points of the StationaryDist, but that
%     % summed up accross the StationaryDist this errors were numerically
%     % important and causing problems.
%     %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
%     P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
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
%     Ptranspose=P';

    if N_d==0 %length(n_d)==1 && n_d(1)==0
        optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
    else
        optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
    end
    Ptranspose=zeros(N_a,N_a*N_z);
    Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
    Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(ones(N_z,1),Ptranspose));
    
    % Now add the part relating to EntryExit2, namely remove the ExitProb
    % share, and add them in based on EntryDist.
    Ptranspose=(1-ExitProb)*Ptranspose+ExitProb*(EntryDist*ones(1,N_a*N_z));
    % Note that ExitProb is also the mass of the entrants.
    
elseif simoptions.parallel==2 % Using the GPU
    % First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)

    if N_d==0 %length(n_d)==1 && n_d(1)==0
        optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
    else
        optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
    end
    Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
    Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
    Ptranspose=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptranspose));
    
    % Now add the part relating to EntryExit2, namely remove the ExitProb
    % share, and add them in based on EntryDist.
    Ptranspose=(1-ExitProb)*Ptranspose+ExitProb*(gpuArray(EntryDist)*ones(1,N_a*N_z,'gpuArray'));
    % Note that ExitProb is also the mass of the entrants.
    
elseif simoptions.parallel>2
%     % The following commented out version was producing machine precision
%     % level errors in Ptranspose that lead to machine precision level
%     % errors in the individual points of the StationaryDist, but that
%     % summed up accross the StationaryDist this errors were numerically
%     % important and causing problems.
%     tic;
%     %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
%     Ptranspose=sparse(N_a*N_z,N_a*N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z) [ So Ptranspose(aprime,zprime,a,z)=proby of going to (a',z') given in (a,z)]
%     if N_d==0 %length(n_d)==1 && n_d(1)==0
%         PolicyIndexesKron=reshape(PolicyIndexesKron(:,:),[N_a*N_z,1]);
%     else
%         PolicyIndexesKron=reshape(PolicyIndexesKron(2,:,:),[N_a*N_z,1]);
%     end
%     
%     parfor az_c=1:N_a*N_z
%         Ptranspose_az=sparse(N_a*N_z,1);
%         optaprime=PolicyIndexesKron(az_c);
%         az_sub=ind2sub_homemade([N_a,N_z],az_c);
%         pi_z_temp=pi_z(az_sub(2),:);
%         sum_pi_z_temp=sum(pi_z_temp); % COULD PRECOMPUTE THIS OUTSIDE THE PARFOR-LOOP (other than rounding error this will be 1, but just to be sure; I added this line as the rounding error turned out to matter for larger models)
%         for zprime_c=1:N_z % THIS COULD BE DONE AS MATRIX OPERATION, would be faster than this for-loop.
%             optaprimezprime=sub2ind_homemade([N_a,N_z],[optaprime,zprime_c]);
%             Ptranspose_az(optaprimezprime)=pi_z_temp(zprime_c)/sum_pi_z_temp;
%         end
%         Ptranspose(:,az_c)=Ptranspose_az;
%     end
%     toc
    pi_z=sparse(gather(pi_z));
    
    if N_d==0 %length(n_d)==1 && n_d(1)==0
        optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
    else
        optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
    end
    Ptranspose=sparse(N_a,N_a*N_z);
    Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
    
%     whos PolicyIndexesKron optaprime Ptranspose pi_z
    Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(ones(N_z,1),Ptranspose));
    
    % Now add the part relating to EntryExit2, namely remove the ExitProb
    % share, and add them in based on EntryDist.
    Ptranspose=(1-ExitProb)*Ptranspose+ExitProb*(sparse(EntryDist)*ones(1,N_a*N_z));
    % Note that ExitProb is also the mass of the entrants.
end


%% The rest is essentially the same regardless of which simoption.parallel is being used
%SteadyStateDistKron=ones(N_a*N_z,1)/(N_a*N_z); % This line was handy when checking/debugging. Have left it here.
if simoptions.parallel==2
    SteadyStateDistKronOld=zeros(N_a*N_z,1,'gpuArray');
else
    SteadyStateDistKronOld=zeros(N_a*N_z,1);
end
SScurrdist=sum(abs(StationaryDistKron-SteadyStateDistKronOld));
SScounter=0;
while SScurrdist>simoptions.tolerance && (100*SScounter)<simoptions.maxit
    
    for jj=1:100
        StationaryDistKron=Ptranspose*StationaryDistKron; %No point checking distance every single iteration. Do 100, then check.
    end
    SteadyStateDistKronOld=StationaryDistKron;
    StationaryDistKron=Ptranspose*StationaryDistKron; % Base the tolerance on 10 iterations. (For some reason just using one iteration worked perfect on gpu, but was not accurate enough on cpu)
    SScurrdist=sum(abs(StationaryDistKron-SteadyStateDistKronOld));
    
    SScounter=SScounter+1;
    if simoptions.verbose==1
        if rem(SScounter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance is %8.4f (tolerance=%8.4f) \n', SScounter, SScurrdist, simoptions.tolerance)            
        end
    end
end

if simoptions.parallel>=3 % Solve with sparse matrix
    StationaryDistKron=full(StationaryDistKron);
    if simoptions.parallel==4 % Solve with sparse matrix, but return answer on gpu.
        StationaryDistKron=gpuArray(StationaryDistKron);
    end
end

if ~(SScounter<simoptions.maxit)
    disp('WARNING: SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end 

%% Leftovers from a now extinct simoption
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
