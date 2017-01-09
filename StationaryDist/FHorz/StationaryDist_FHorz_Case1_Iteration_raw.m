function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_raw(jequaloneDistKron,PolicyIndexesKron,N_d,N_a,N_z,N_j,pi_z,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel


if simoptions.parallel~=2
    
    StationaryDistKron=zeros(N_a*N_z,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
        for a_c=1:N_a
            for z_c=1:N_z
                if N_d==0 %length(n_d)==1 && n_d(1)==0
                    optaprime=PolicyIndexesKron(a_c,z_c,jj);
                else
                    optaprime=PolicyIndexesKron(2,a_c,z_c,jj);
                end
                for zprime_c=1:N_z
                    P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
                end
            end
        end
        P=reshape(P,[N_a*N_z,N_a*N_z]);
        P=P';
        
        StationaryDistKron(:,jj+1)=P*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==2 % Using the GPU
    
    StationaryDistKron=zeros(N_a*N_z,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            optaprime=reshape(PolicyIndexesKron(:,:,jj),[1,N_a*N_z]);
        else
            optaprime=reshape(PolicyIndexesKron(2,:,:,jj),[1,N_a*N_z]);
        end
        Ptran=zeros(N_a,N_a*N_z,'gpuArray');
        Ptran(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
        Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptran));
        
        StationaryDistKron(:,jj+1)=Ptran*StationaryDistKron(:,jj);
    end
end

end
