function AgentDistPath=AgentDistOnTransPath_Case1(AgentDist_initial, PolicyPath,n_d,n_a,n_z,pi_z,T,transpathoptions,simoptions)
n_e=0; % NOT YET IMPLEMENTED FOR TRANSITION PATHS

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);

%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=2;
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=2;
    end
    if isfield(transpathoptions,'verbose')==0
        transpathoptions.verbose=0;
    end
end

%% Check which simoptions have been used, set all others to defaults 
if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    simoptions.ncores=feature('numcores'); % Number of CPU cores
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if isfield(simoptions,'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
    if isfield(simoptions,'nsims')==0
        simoptions.nsims=10^4;
    end
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ncores')==0
        simoptions.ncores=feature('numcores'); % Number of CPU cores
    end
    if isfield(simoptions,'iterate')==0
        simoptions.iterate=1;
    end
end

%%
if transpathoptions.parallel==2 
       pi_z=gpuArray(pi_z);
       
       PolicyPath=KronPolicyIndexes_TransPath_Case1(PolicyPath, n_d, n_a, n_z,T);
       AgentDistPath=zeros(N_a*N_z,T,'gpuArray');
       
       % Now we have the full PolicyIndexesPath, we go forward in time from 1
       % to T using the policies to update the agents distribution generating anew price path
       
       % Call AgentDist the current periods distn
       AgentDist=reshape(AgentDist_initial,[N_a*N_z,1]);
       AgentDistPath(:,1)=AgentDist;
       if N_d>0
           for tt=1:T-1
               %Get the current optimal policy
               optaprime=reshape(PolicyPath(2,:,:,tt),[1,N_a*N_z]);
               
               Ptemp=zeros(N_a,N_a*N_z,'gpuArray');
               Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
               Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptemp));
               AgentDist=Ptran*AgentDist;
               
               AgentDistPath(:,tt+1)=AgentDist;
           end
       else
           for tt=1:T-1
               %Get the current optimal policy
               optaprime=reshape(PolicyPath(:,:,tt),[1,N_a*N_z]);
               
               Ptemp=zeros(N_a,N_a*N_z,'gpuArray');
               Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
               Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptemp));
               AgentDist=Ptran*AgentDist;
               
               AgentDistPath(:,tt+1)=AgentDist;
           end
       end

else
    pi_z=gather(pi_z);
    
    PolicyPath=KronPolicyIndexes_TransPath_Case1(PolicyPath, n_d, n_a, n_z,T);
    AgentDistPath=zeros(N_a,N_z,T);
    
    % Now we have the full PolicyIndexesPath, we go forward in time from 1
    % to T using the policies to update the agents distribution generating anew price path
    
    % Call AgentDist the current periods distn
    AgentDist=reshape(AgentDist_initial,[N_a*N_z,1]);
    AgentDistPath(:,1)=AgentDist;
    if N_d>0
        for tt=1:T-1
            %Get the current optimal policy
            optaprime=reshape(PolicyPath(2,:,:,tt),[1,N_a*N_z]);
            
            Ptemp=zeros(N_a,N_a*N_z);
            Ptemp(optaprime+N_a*(0:1:N_a*N_z-1))=1;
            Ptran=(kron(pi_z',ones(N_a,N_a))).*(kron(ones(N_z,1),Ptemp));
            AgentDist=Ptran*AgentDist;
            
            AgentDistPath(:,tt+1)=AgentDist;
        end
    else
        for tt=1:T-1
            %Get the current optimal policy
            optaprime=reshape(PolicyPath(:,:,tt),[1,N_a*N_z]);
            
            Ptemp=zeros(N_a,N_a*N_z);
            Ptemp(optaprime+N_a*(0:1:N_a*N_z-1))=1;
            Ptran=(kron(pi_z',ones(N_a,N_a))).*(kron(ones(N_z,1),Ptemp));
            AgentDist=Ptran*AgentDist;
            
            AgentDistPath(:,tt+1)=AgentDist;
        end
    end
end


AgentDistPath=reshape(AgentDistPath,[n_a,n_z,T]);


end