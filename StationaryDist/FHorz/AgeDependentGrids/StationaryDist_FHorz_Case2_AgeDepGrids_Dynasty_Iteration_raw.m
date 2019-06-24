function StationaryDistKron=StationaryDist_FHorz_Case2_AgeDepGrids_Dynasty_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,daz_gridstructure,N_j,Phi_aprime,Case2_Type,Parameters,PhiaprimeParamNames,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

if simoptions.verbose==1
    fprintf('StationaryDist_Case2_FHorz_AgeDepGrids_Dynasty_Iteration_raw() \n')
end

%%
StationaryDistKron=struct();
if simoptions.parallel<=2
    StationaryDistKron.j001=jequaloneDistKron;
elseif simoptions.parallel>2
    % Move StationaryDist to cpu temporarily so can use sparse matrices,
    % then move the solutions back to the gpu at the end
    StationaryDistKron.j001=sparse(gather(jequaloneDistKron));
    % Note to self: it appears that Matlab should automatically 'choose' to
    % use parallel cpus when multiplying sparse cpus. Not clear if this
    % behaviour can/should be forced.
end

if simoptions.agedependentgrids(1)==0
    n_d_j=daz_gridstructure.n_d.j001;
    N_d_j=prod(n_d_j);
    d_grid_j=daz_gridstructure.d_grid.j001;
end
if simoptions.agedependentgrids(2)==0
    n_a_j=daz_gridstructure.n_a.j001;
    N_a_j=prod(n_a_j);
    a_grid_j=daz_gridstructure.a_grid.j001;
    N_aprime_j=N_a_j;
end
if simoptions.agedependentgrids(3)==0
    n_z_j=daz_gridstructure.n_z.j001;
    N_z_j=prod(n_z_j);
    z_grid_j=daz_gridstructure.z_grid.j001;
    pi_z_j=pi_z;
    if simoptions.parallel>2
        pi_z_j_cpu=daz_gridstructure.pi_z_cpu.j001;
    end
    N_zprime_j=N_z_j;
end

tempcounter=1;
currdist=Inf;
while currdist>simoptions.tolerance
    
    tempcounter
    [currdist, simoptions.tolerance]
        
    % Make a three digit number out of jj=1. Needed so that the first time
    % codes gets Vnextj=V.(jstr); this will be the age 1 due to dynasty.
    jstr='j001';


    if simoptions.parallel<2
        
        %     StationaryDistKron=struct();
        %     StationaryDistKron.j001=jequaloneDistKron;
        
        if vfoptions.phiaprimedependsonage==0
            fprintf('ERROR: state dependent grids and Case2 mean that you must have vfoptions.phiaprimedependsonage==1')
        end
        
        if simoptions.phiaprimedependsonage==0
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
            if simoptions.lowmemory==0
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid_j, a_grid_j, z_grid_j,PhiaprimeParamsVec);
            end
        end
        
        for jj=1:N_j
            
            % Make a three digit number out of jj
            jstr=daz_gridstructure.jstr{jj};
            jjplus1=jj+1;
            if jj==N_j
                jjplus1=1;
            end
            % Make a three digit number out of jjplus1
            jplus1str=daz_gridstructure.jstr{jjplus1};
            % Get the relevant grid and transition matrix
            if simoptions.agedependentgrids(1)==1
                N_d_j=daz_gridstructure.N_d.(jstr(:));
                n_d_j=daz_gridstructure.n_d.(jstr(:));
            end
            if simoptions.agedependentgrids(2)==1
                N_a_j=daz_gridstructure.N_a.(jstr(:));
                n_a_j=daz_gridstructure.n_a.(jstr(:));
                N_aprime_j=daz_gridstructure.N_aprime.(jstr(:));
            end
            if simoptions.agedependentgrids(3)==1
                N_z_j=daz_gridstructure.N_z.(jstr(:));
                n_z_j=daz_gridstructure.n_z.(jstr(:));
                pi_z_j=daz_gridstructure.pi_z.(jstr(:));
                N_zprime_j=daz_gridstructure.N_z.(jstr(:));
            end
            
            if simoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d_j, n_a_j, n_z_j, d_grid_j, a_grid_j, z_grid_j,PhiaprimeParamsVec);
            end
            
            PolicyKron_j=PolicyKron.(jstr);
            %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
            P=zeros(N_a_j,N_z_j,N_aprime_j,N_zprime_j); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
            for a_c=1:N_a_j
                for z_c=1:N_z_j
                    optd=PolicyKron_j(a_c,z_c);
                    if Case2_Type==1 % a'(d,a,z,z')
                        for zprime_c=1:N_zprime_j
                            optaprime=Phi_aprimeMatrix(optd,a_c,z_c,zprime_c);
                        end
                    elseif Case2_Type==11 % a'(d,a,z')
                        for zprime_c=1:N_zprime_j
                            optaprime=Phi_aprimeMatrix(optd,a_c,zprime_c);
                        end
                    elseif Case2_Type==12 % a'(d,a,z)
                        optaprime=Phi_aprimeMatrix(optd,a_c,z_c);
                    elseif Case2_Type==2 % a'(d,z,z')
                        for zprime_c=1:N_zprime_j
                            optaprime=Phi_aprimeMatrix(optd,z_c,zprime_c);
                        end
                    end
                    for zprime_c=1:N_zprime_j
                        P(a_c,z_c,optaprime,zprime_c)=pi_z_j(z_c,zprime_c)/sum(pi_z_j(z_c,:));
                    end
                end
            end
            P=reshape(P,[N_a_j*N_z_j,N_aprime_j*N_zprime_j]);
            P=P';
            
            StationaryDistKron.(jplus1str)=P*StationaryDistKron.(jstr);
        end
        
    elseif simoptions.parallel==2 % GPU
        
%         StationaryDistKron=struct();
%         StationaryDistKron.j001=jequaloneDistKron;
        
        if simoptions.phiaprimedependsonage==0
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
            if simoptions.lowmemory==0
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d_j, n_a_j, n_z_j, d_grid_j, a_grid_j, z_grid_j,PhiaprimeParamsVec);
            end
        end
        
        % First, generate the transition matrix P=g of Q (the convolution of the
        % optimal policy function and the transition fn for exogenous shocks)
        for jj=1:N_j
            
            % Make a three digit number out of jj
            jstr=daz_gridstructure.jstr{jj};
            jjplus1=jj+1;
            if jjplus1==N_j+1
                jjplus1=1;
            end
            % Make a three digit number out of jjplus1
            jplus1str=daz_gridstructure.jstr{jjplus1};
            % Get the relevant grid and transition matrix
            if simoptions.agedependentgrids(1)==1
                N_d_j=daz_gridstructure.N_d.(jstr(:));
                n_d_j=daz_gridstructure.n_d.(jstr(:));
                d_grid_j=daz_gridstructure.d_grid.(jstr(:));
            end
            if simoptions.agedependentgrids(2)==1
                N_a_j=daz_gridstructure.N_a.(jstr(:));
                n_a_j=daz_gridstructure.n_a.(jstr(:));
                N_aprime_j=daz_gridstructure.N_aprime.(jstr(:));
                a_grid_j=daz_gridstructure.a_grid.(jstr(:));
            end
            if simoptions.agedependentgrids(3)==1
                N_z_j=daz_gridstructure.N_z.(jstr(:));
                n_z_j=daz_gridstructure.n_z.(jstr(:));
                pi_z_j=daz_gridstructure.pi_z.(jstr(:));
                z_grid_j=daz_gridstructure.z_grid.(jstr(:));
                N_zprime_j=daz_gridstructure.N_z.(jstr(:));
            end
            
            if simoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d_j, n_a_j, n_z_j, d_grid_j, a_grid_j, z_grid_j,PhiaprimeParamsVec);
            end
            
            PolicyKron_j=PolicyKron.(jstr);
            
            % Phi_of_Policy combines Phi_aprimeMatrix together with PolicyKron.
            % It gives the index for aprime as a function of (whatever
            % Phi_aprime depends on) together with (a,z).
            if Case2_Type==1 % phi(d,a,z,z')
                disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==1')
                % Phi_of_policy: a'(a,z,z')
                Phi_of_Policy=zeros(N_a_j,N_zprime_j,N_z_j,'gpuArray'); %a'(d,z',z)
                %             for z_c=1:N_z
                %                 Phi_of_Policy(:,:,z_c)=Phi_aprimeMatrix(PolicyKron_j(:,z_c),:,:,z_c);
                %             end
                % Create P matrix
                Ptemp=zeros(N_a_j,N_aprime_j*N_z_j*N_zprime_j,'gpuArray');
                Ptemp(reshape(permute(Phi_of_Policy,[2,1,3]),[1,N_a_j*N_z_j*N_zprime_j])+N_aprime_j*(gpuArray(0:1:N_a_j*N_z_j*N_zprime_j-1)))=1;
                Ptran=kron(pi_z_j',ones(N_aprime_j,N_a_j,'gpuArray')).*reshape(Ptemp,[N_aprime_j*N_zprime_j,N_a_j*N_z_j]);
            elseif Case2_Type==11 % phi(d,a,z')
                disp('WARNING: StationaryDist_FHorz_Case2_Iteration_raw() is implemented but not been tested for Case2_Type==11')
                % Phi_of_policy: a'(a,z,z')
                Phi_of_Policy=zeros(N_a_j*N_zprime_j,N_z_j,'gpuArray'); %a'(d,z',z)
                for z_c=1:N_z_j
                    Phi_of_Policy(:,z_c)=Phi_aprimeMatrix(PolicyKron_j(:,z_c),:,z_c);
                end
                % Create P matrix
                Ptemp=zeros(N_a_j,N_aprime_j*N_z_j*N_zprime_j,'gpuArray');
                Ptemp(reshape(permute(Phi_of_Policy,[2,1,3]),[1,N_a_j*N_z_j*N_zprime_j])+N_aprime_j*(gpuArray(0:1:N_a_j*N_z_j*N_zprime_j-1)))=1;
                Ptran=kron(pi_z_j',ones(N_aprime_j,N_a_j,'gpuArray')).*reshape(Ptemp,[N_aprime_j*N_zprime_j,N_a_j*N_z_j]);
            elseif Case2_Type==12 % phi(d,a,z)
                % Phi_of_policy: a'(a,z)
                Phi_of_Policy=zeros(N_a_j*N_z_j,1,'gpuArray'); %a'(a,z)
                aindexes=kron((1:1:N_a_j)',ones(N_z_j,1));
                zindexes=kron(ones(N_a_j,1),(1:1:N_z_j)');
%                 size(PolicyKron_j)
%                 size(PolicyKron_j(:))
%                 size(aindexes)
%                 size(zindexes)
%                 max(PolicyKron_j(:))
%                 max(aindexes)
%                 max(zindexes)
                Phi_of_Policy=Phi_aprimeMatrix(PolicyKron_j(:)+N_d_j*(aindexes-1)+N_d_j*N_a_j*(zindexes-1));
                Ptemp=zeros(N_a_j,N_aprime_j*N_z_j*N_zprime_j,'gpuArray');
                Ptemp(reshape(Phi_of_Policy,[1,N_a_j*N_z_j])+N_aprime_j*N_zprime_j*(gpuArray(0:1:N_a_j*N_z_j-1)))=1;
                Ptran=kron(pi_z_j',ones(N_aprime_j,N_a_j,'gpuArray')).*reshape(Ptemp,[N_aprime_j*N_zprime_j,N_a_j*N_z_j]);
            elseif Case2_Type==2  % phi(d,z',z)
                % Phi_of_policy: a'(a,z',z)
                Phi_of_Policy=zeros(N_a_j,N_zprime_j,N_z_j,'gpuArray'); %a'(d,z',z)
                for z_c=1:N_z_j
                    Phi_of_Policy(:,:,z_c)=Phi_aprimeMatrix(PolicyKron_j(:,z_c),:,z_c);
                end
                Ptemp=zeros(N_a_j,N_aprime_j*N_z_j*N_zprime_j,'gpuArray');
                Ptemp(reshape(permute(Phi_of_Policy,[2,1,3]),[1,N_a_j*N_z_j*N_zprime_j])+N_aprime_j*(gpuArray(0:1:N_a_j*N_z_j*N_zprime_j-1)))=1;
                Ptran=kron(pi_z_j',ones(N_aprime_j,N_a_j,'gpuArray')).*reshape(Ptemp,[N_aprime_j*N_zprime_j,N_a_j*N_z_j]);
            end
            
            StationaryDistKron.(jplus1str)=Ptran*StationaryDistKron.(jstr);
        end
        
    elseif simoptions.parallel>2 % Same as <2, but using sparse matrices. To make it faster I do some vectorization.
        
%         StationaryDistKron=struct();
%         StationaryDistKron.j001=jequaloneDistKron;
        
        if simoptions.phiaprimedependsonage==0
            fprintf('ERROR: state dependent grids and Case2 mean that you must have simoptions.phiaprimedependsonage==1')
        end
        
        if simoptions.phiaprimedependsonage==0
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
            if simoptions.lowmemory==0
                Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d_j, n_a_j, n_z_j, d_grid_j, a_grid_j, z_grid_j,PhiaprimeParamsVec);
            end
        end
        
        for jj=1:N_j
            
            % Make a three digit number out of jj
            jstr=daz_gridstructure.jstr{jj};
            jjplus1=jj+1;
            if jjplus1==N_j+1
                jjplus1=1;
            end
            % Make a three digit number out of jjplus1
            jplus1str=daz_gridstructure.jstr{jjplus1};
            % Get the relevant grid and transition matrix
            if simoptions.agedependentgrids(1)==1
                N_d_j=daz_gridstructure.N_d.(jstr(:));
                n_d_j=daz_gridstructure.n_d.(jstr(:));
                d_grid_j=daz_gridstructure.d_grid.(jstr(:));
            end
            if simoptions.agedependentgrids(2)==1
                N_a_j=daz_gridstructure.N_a.(jstr(:));
                n_a_j=daz_gridstructure.n_a.(jstr(:));
                N_aprime_j=daz_gridstructure.N_aprime.(jstr(:));
                a_grid_j=daz_gridstructure.a_grid.(jstr(:));
            end
            if simoptions.agedependentgrids(3)==1
                N_z_j=daz_gridstructure.N_z.(jstr(:));
                n_z_j=daz_gridstructure.n_z.(jstr(:));
                z_grid_j=daz_gridstructure.z_grid.(jstr(:));
                pi_z_j=daz_gridstructure.pi_z.(jstr(:));
                pi_z_j_cpu=daz_gridstructure.pi_z_cpu.(jstr(:));
                N_zprime_j=daz_gridstructure.N_zprime.(jstr(:));
            end
            
            if simoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
                % Because this is simoptions.parallel>2 we want to move Phi_aprimeMatrix to the cpu
                Phi_aprimeMatrix=gather(CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d_j, n_a_j, n_z_j, d_grid_j, a_grid_j, z_grid_j,PhiaprimeParamsVec));
                % HAVE NOT formally tested creating on gpu and moving to
                % cpu against just creating on cpu. Am guessing former is
                % faster.
            end
            
            PolicyKron_j=PolicyKron.(jstr);
            %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
            if Case2_Type==1 %phi(d,a,z,z')
                
            elseif Case2_Type==11 % phi(d,a,z')
                
            elseif Case2_Type==12 % phi(d,a,z)
                Phi_of_Policy=zeros(N_a_j*N_z_j,1); %a'(a,z) % Contains the aprime indexes
                aindexesvec=kron((1:1:N_a_j)',ones(N_z_j,1)); % Size az-by-1
                zindexesvec=kron(ones(N_a_j,1),(1:1:N_z_j)'); % Size az-by-1
                Phi_of_Policy=Phi_aprimeMatrix(PolicyKron_j(:)+N_d_j*(aindexesvec-1)+N_d_j*N_a_j*(zindexesvec-1)); % Size az-by-1
                % The following lines of code are just about creating the sparse matrix in a fast way: https://au.mathworks.com/matlabcentral/answers/203734-most-efficient-way-to-add-multiple-sparse-matrices-in-a-loop-in-matlab
                % Calculate number of non-zero entries in P. It will be
                % N_a*(number of non-zero entries in pi_z)
                % Notice that the roles of both z and zprime are captured by pi_z
                
                binary_pi_z=pi_z_j_cpu;
                binary_pi_z(binary_pi_z>0)=1;
                
                % Create a version of Phi_of_Policy which has a'(a,z,z'), but only for the z' which are non-zero transition probabilities.
                %             tic;
                PaprimeIndex=Phi_of_Policy.*kron(binary_pi_z,ones(N_a_j,1)); % There should be a better way to do this and the next step
                PaprimeIndex=PaprimeIndex(PaprimeIndex>0);
                %             time1=toc
                
                % SLOWER LOWER MEMORY VERSION THAT TAKES GREATER ADVANTAGE OF SPARSENESS
                %             tic;
                %             nonzerotransitions=sum(binary_pi_z,2);
                %             PaprimeIndex2=repelem(reshape(Phi_of_Policy,[N_a,N_z]),1,nonzerotransitions');
                %             PaprimeIndex2=reshape(PaprimeIndex2,[N_a*sum(nonzerotransitions),1]);
                %             time2=toc
                
                % The non-zero-elements-only version of PaprimezprimeIndex is
                % roughly one-fifthtieth the size. Can I skip straight to it in
                % a memory efficient way?
                
                % Really should work on the gpu till here and only go to cpu when switching to sparse.
                
                % The transition probabilities themselves can be gotten from kron(ones(N_a,1),pi_z)
%                 Pvalues=kron(ones(N_a_j,1),pi_z_j_cpu); % THIS WAS INCORRECT
                Pvalues=kron(pi_z_j_cpu,ones(N_a_j,1));
                Pvalues=Pvalues(Pvalues>0);
                
                PzprimeIndex=kron(ones(N_a_j,1),(binary_pi_z.*(ones(N_z_j,1)*1:1:N_zprime_j)));
                PzprimeIndex=PzprimeIndex(PzprimeIndex>0);
                
                PaprimezprimeIndex=PaprimeIndex+N_aprime_j*(PzprimeIndex-1);
                PaprimezprimeIndex=PaprimezprimeIndex(PaprimezprimeIndex>0);
                
                % Phi_of_Policy gives me the aprime indexes
                % kron(ones(N_a,1),binary_pi_z) gives me the zprime indexes (once I remove the zeros
                % And kron(ones(N_a,1),pi_z) gives me the transition probabilities (once I remove the zeros)
                Paz1=(kron(ones(N_z_j,1),(1:1:N_a_j)')+N_a_j*kron(((1:1:N_z_j)'-1),ones(N_a_j,1)));
                PazIndex=Paz1.*kron(binary_pi_z,ones(N_a_j,1));
                PazIndex=PazIndex(PazIndex>0);
                
                % So now it is just a matter of putting them together as a sparse matrix
                Ptranspose=sparse(PaprimezprimeIndex,PazIndex,Pvalues,N_aprime_j*N_zprime_j,N_a_j*N_z_j); % I is the (aprime-by-zprime)-index, J is the (a-by-z)-index.
                
%                 fprintf('Ptranspose stuff that should both evaluate to one \n')
%                 [min(sum(Ptranspose)), max(sum(Ptranspose))]
                
            elseif Case2_Type==2 % phi(d,z',z)
                
            end

            StationaryDistKron.(jplus1str)=Ptranspose*(StationaryDistKron.(jstr));
        end
        
    end
    
    % For dynasty, reweighting based on age weights is done outside the while loop once
    % convergence is reached.
    
    if tempcounter>=2 % I simply assume you won't converge on the first try when using dynasty
        % No need to check convergence for the whole agents distribution, if the
        % 'youngest', j==1, has converged then necessarily so have all the others.
        N_a_j=daz_gridstructure.N_a.j001;
        N_z_j=daz_gridstructure.N_z.j001;
        StationaryDistdist=reshape(StationaryDistKron.j001-StationaryDistKronold,[N_a_j*N_z_j,1]); StationaryDistdist(isnan(StationaryDistdist))=0;
        % Note that with sparse matrices need to convert currdist into a full matrix
        currdist=full(max(abs(StationaryDistdist))); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?
    end
    StationaryDistKronold=StationaryDistKron.j001; % Only need to keep the old j==1, no need for the rest
    
    tempcounter=tempcounter+1;
    if simoptions.verbose==1 && rem(tempcounter,10)==0
        fprintf('Stationary Distribution (Iteration): After %d steps, current distance is %8.2f \n', tempcounter, currdist);
    end
end

if simoptions.parallel>2
    % Move StationaryDist to cpu temporarily so can use sparse matrices,
    % then move the solutions back to the gpu at the end.
    for jj=1:N_j
        jstr=daz_gridstructure.jstr{jj};
        StationaryDistKron.(jstr)=gpuArray(full(StationaryDistKron.(jstr)));
    end
end

% Reweight the different ages based on 'AgeWeightParamNames'. (it is
% assumed there is only one Age Weight Parameter (name))
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);
found=0;
for iField=1:nFields
    if strcmp(AgeWeightParamNames{1},FullParamNames{iField})
        AgeWeights=Parameters.(FullParamNames{iField});
        found=1;
    end
end
if found==0 % Have added this check so that user can see if they are missing a parameter
    fprintf(['FAILED TO FIND PARAMETER ',AgeWeightParamNames{1}])
end
% I assume AgeWeights is a row vector
for jj=1:N_j
    % Make a three digit number out of jj
    jstr=daz_gridstructure.jstr{jj};    
    StationaryDistKron.(jstr)=StationaryDistKron.(jstr)*AgeWeights(jj);
end
StationaryDistKron.AgeWeights=AgeWeights; % This will be helpful to have precomputed for any time I want to draw randomly from the stationary distribution


end