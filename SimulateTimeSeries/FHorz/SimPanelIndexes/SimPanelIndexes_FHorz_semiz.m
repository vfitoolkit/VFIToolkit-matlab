function SimPanel=SimPanelIndexes_FHorz_semiz(InitialDist,Policy,n_d,n_a,n_z,N_j,pi_z_J, Parameters,simoptions)
% Inputs should already be on cpu, output is on cpu
% 
% Intended to be called from SimPanelValues_FHorz_Case1()

N_d=prod(n_d);
if N_d>0
    l_d=length(n_d);
else
    l_d=0;
end

N_a=prod(n_a);
l_a=length(n_a);

N_z=prod(n_z);
if N_z>0
    l_z=length(n_z);
    cumsumpi_z_J=cumsum(pi_z_J,2);
else
    l_z=0;
end
N_e=prod(simoptions.n_e);
if N_e>0
    l_e=length(simoptions.n_e);
    cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));
else
    l_e=0;
end

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were


%% Setup related to semi-exogenous state (an exogenous state whose transition probabilities depend on a decision variable)
if ~isfield(simoptions,'n_semiz')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.n_semiz')
end
if ~isfield(simoptions,'semiz_grid')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.semiz_grid')
end
if ~isfield(simoptions,'d_grid')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.d_grid')
else
    simoptions.d_grid=gpuArray(simoptions.d_grid);
end
if ~isfield(simoptions,'numd_semiz')
    simoptions.numd_semiz=1; % by default, only one decision variable influences the semi-exogenous state
end
if length(n_d)>simoptions.numd_semiz
    n_d1=n_d(1:end-simoptions.numd_semiz);
    d1_grid=simoptions.d_grid(1:sum(n_d1));
    l_d1=length(n_d1);
else
    n_d1=0; 
    d1_grid=[];
    l_d1=0;
end
N_d1=prod(n_d1);
n_d2=n_d(end-simoptions.numd_semiz+1:end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
l_d2=length(n_d2);
d2_grid=simoptions.d_grid(sum(n_d1)+1:end);


% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
simoptions=SemiExogShockSetup_FHorz(n_d,N_j,simoptions.d_grid,Parameters,simoptions,2);
% output: vfoptions.semiz_gridvals_J, vfoptions.pi_semiz_J
% size(semiz_gridvals_J)=[prod(n_z),length(n_z),N_j]
% size(pi_semiz_J)=[prod(n_semiz),prod(n_semiz),prod(n_dsemiz),N_j]
% If no semiz, then vfoptions just does not contain these field


% Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
N_semiz=prod(simoptions.n_semiz);
l_semiz=length(simoptions.n_semiz);
cumsumpi_semiz_J=gather(cumsum(simoptions.pi_semiz_J,2));


%%
if N_z==0
    if N_e==0
        N_semizze=N_semiz;
    else
        N_semizze=N_semiz*N_e;
    end
else
    if N_e==0
        N_semizze=N_semiz*N_z;
    else
        N_semizze=N_semiz*N_z*N_e;
    end
end


if N_semizze==0
    Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);
    Policy_aprime=Policy(l_d+1:end,:,:);
else
    Policy=reshape(Policy,[size(Policy,1),N_a,N_semizze,N_j]);
    Policy_aprime=Policy(l_d+1:end,:,:,:);
end

Policy_aprime=KronPolicyIndexes_FHorz_Case1(Policy_aprime,0,n_a,N_semizze,N_j,simoptions);
if simoptions.gridinterplayer==1
    CumPolicyProbs=ones([N_a,N_semizze,2,N_j]);
    CumPolicyProbs(:,:,1,:)=reshape(Policy_aprime(2,:,:,:),[N_a,N_semizze,1,N_j]); % L2 index
    CumPolicyProbs(:,:,1,:)=1-(CumPolicyProbs(:,:,1,:)-1)/(simoptions.ngridinterp+1); % prob of lower index
    % CumPolicyProbs(:,:,2,:) just leave this as ones
    Policy_aprime=repmat(reshape(Policy_aprime(1,:,:,:),[N_a,N_semizze,1,N_j]),1,1,2,1); % lower grid index
    Policy_aprime(:,:,2,:)=Policy_aprime(:,:,2,:)+1; % upper grid index
end

%% Policy_dsemiexo

% d2 is the variable relevant for the semi-exogenous asset. 
if l_d2==1
    Policy_dsemiexo=Policy(l_d1+1,:,:,:);
elseif l_d2==2
    Policy_dsemiexo=Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*(Policy(l_d1+2,:,:,:)-1);
elseif l_d2==3
    Policy_dsemiexo=Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*(Policy(l_d1+2,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*(Policy(l_d1+3,:,:,:)-1); 
elseif l_d2==4
    Policy_dsemiexo=Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*(Policy(l_d1+2,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*(Policy(l_d1+3,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*n_d(l_d1+3)*(Policy(l_d1+4,:,:,:)-1);
end
Policy_dsemiexo=shiftdim(Policy_dsemiexo,1);



%% Do the simulations themselves

if N_z==0
    if N_e==0  % No z, No e
        if simoptions.gridinterplayer==0
            Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_j]);
        elseif simoptions.gridinterplayer==1
            Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,2,N_j]);
            CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,2,N_j]);
        end
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_j]);

        % Get seedpoints from InitialDist
        [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
        if numel(InitialDist)==N_a*N_semiz % Has just been given for age j=1
            seedpoints=[ind2sub_vec_homemade([N_a,N_semiz],seedpointind'),ones(simoptions.numbersims,1)];
        else  % Distribution across ages as well
            seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_j],seedpointind'),ones(simoptions.numbersims,1)];
        end
        seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

        SimPanel=nan(3,N_j,simoptions.numbersims); % (a,semiz,j)
        if simoptions.gridinterplayer==0
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_semiz_noz_raw(Policy_aprime,Policy_dsemiexo,N_j,cumsumpi_semiz_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        elseif simoptions.gridinterplayer==1
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_noz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[3,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+1,N_j*simoptions.numbersims); % (a,semiz,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
            SimPanel(end,:)=SimPanelKron(3,:); % j

            SimPanel=reshape(SimPanel,[3,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            % Only semiz, so nothing to do
        end

    else % No z, e
        if simoptions.gridinterplayer==0
            Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_e,N_j]);
        elseif simoptions.gridinterplayer==1
            Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_e,2,N_j]);
            CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,N_e,2,N_j]);
        end
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_e,N_j]);

        % Get seedpoints from InitialDist
        [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
        if numel(InitialDist)==N_a*N_semiz*N_e % Has just been given for age j=1
            seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_e],seedpointind'),ones(simoptions.numbersims,1)];
        else  % Distribution across ages as well
            seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_e,N_j],seedpointind'),ones(simoptions.numbersims,1)];
        end
        seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

        SimPanel=nan(4,N_j,simoptions.numbersims); % (a,semiz,e,j)
        if simoptions.gridinterplayer==0
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_semiz_noz_e_raw(Policy_aprime,Policy_dsemiexo,N_j,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        elseif simoptions.gridinterplayer==1
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_noz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+l_e+1,N_j*simoptions.numbersims); % (a,semiz,e,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
            SimPanel(l_a+l_semiz+l_z+1:l_a+l_semiz+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(3,:)); % e
            SimPanel(end,:)=SimPanelKron(4,:); % j

            SimPanel=reshape(SimPanel,[4,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            SimPanel(2,:,:)=SimPanel(2,:,:)+N_semiz*(SimPanel(3,:,:)-1); % put semiz and e together
            SimPanel(3,:,:)=SimPanel(4,:,:); % move j forward
            SimPanel=SimPanel(1:3,:,:);
        end
    end

else % N_z>0
    if N_e==0  % z, No e
        if simoptions.gridinterplayer==0
            Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_z,N_j]);
        elseif simoptions.gridinterplayer==1
            Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_z,2,N_j]);
            CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,N_z,2,N_j]);
        end
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_z,N_j]);

        % Get seedpoints from InitialDist
        [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
        if numel(InitialDist)==N_a*N_semiz*N_z % Has just been given for age j=1
            seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_z],seedpointind'),ones(simoptions.numbersims,1)];
        else  % Distribution across ages as well
            seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_z,N_j],seedpointind'),ones(simoptions.numbersims,1)];
        end
        seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
                
        SimPanel=nan(4,N_j,simoptions.numbersims); % (a,semiz,z,j)
        if simoptions.gridinterplayer==0
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_semiz_raw(Policy_aprime,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        elseif simoptions.gridinterplayer==1
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+l_z+1,N_j*simoptions.numbersims); % (a,semiz,z,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
            SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(3,:)); % z
            SimPanel(end,:)=SimPanelKron(4,:); % j

            SimPanel=reshape(SimPanel,[4,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            SimPanel(2,:,:)=SimPanel(2,:,:)+N_semiz*(SimPanel(3,:,:)-1); % put semiz and z together
            SimPanel(3,:,:)=SimPanel(4,:,:); % move j forward
            SimPanel=SimPanel(1:3,:,:);
        end
        
    else % z, e
        if simoptions.gridinterplayer==0
            Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_z,N_e,N_j]);
        elseif simoptions.gridinterplayer==1
            Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_z,N_e,2,N_j]);
            CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,N_z,N_e,2,N_j]);
        end
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_z,N_e,N_j]);

        % Get seedpoints from InitialDist
        [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
        if numel(InitialDist)==N_a*N_semiz*N_z*N_e % Has just been given for age j=1
            seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_z,N_e],seedpointind'),ones(simoptions.numbersims,1)];
        else  % Distribution across ages as well
            seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_z,N_e,N_j],seedpointind'),ones(simoptions.numbersims,1)];
        end
        seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

        SimPanel=nan(5,N_j,simoptions.numbersims); % (a,semiz,z,e,j)
        if simoptions.gridinterplayer==0
            parfor ii=1:simoptions.numbersims
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_semiz_e_raw(Policy_aprime,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        elseif simoptions.gridinterplayer==1
            parfor ii=1:simoptions.numbersims
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        end
        
        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[5,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+l_z+l_e+1,N_j*simoptions.numbersims); % (a,semiz,z,e,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
            SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(3,:)); % z
            SimPanel(l_a+l_semiz+l_z+1:l_a+l_semiz+l_z+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(4,:)); % e
            SimPanel(end,:)=SimPanelKron(5,:); % j

            SimPanel=reshape(SimPanel,[5,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            SimPanel(2,:,:)=SimPanel(2,:,:)+N_semiz*(SimPanel(3,:,:)-1)+N_semiz*N_z*(SimPanel(4,:,:)-1); % put semiz and z and e together
            SimPanel(3,:,:)=SimPanel(5,:,:); % move j forward
            SimPanel=SimPanel(1:3,:,:);
        end
    end
end



end



