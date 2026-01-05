function SimPanel=SimPanelIndexes_InfHorz(InitialDist,PolicyIndexesKron,n_d,n_a,n_z,pi_z, simoptions, CondlProbOfSurvival, Parameters)
% Input must already be on CPU
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length 'simperiods' beginning from randomly drawn InitialDist. 
% CondlProbOfSurvival is an optional input. (only needed when using: simoptions.exitinpanel=1, there there is exit, either exog, endog or mix of both)
%
% InitialDist is n_a-by-n_z
%
% Parameters is only needed as an input when you have mixed (endogenous and exogenous) exit. It is otherwise not required to be inputed.
%
% simoptions are already set, this is an internal use only command
% (SimPanelIndexes is called by SimPanelValues)

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

simoptions.numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()
simoptions.simperiods=gather(simoptions.simperiods);
simoptions.burnin=gather(simoptions.burnin);

%%
if exist('CondlProbOfSurvival','var')==1
    simoptions.exitinpanel=1;
    CondlProbOfSurvivalKron=reshape(CondlProbOfSurvival,[N_a,N_z]);
    if ~isfield(simoptions, 'endogenousexit')
        simoptions.endogenousexit=0;  % Note: this will only be relevant if exitinpanel=1
    end
else
    CondlProbOfSurvivalKron=0; % will be unused, but otherwise there was an error that it wasnt recognized
end


%%
% Get seedpoints from InitialDist
cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z,1]));
[~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1));

cumsumpi_z=cumsum(pi_z,2);

SimPanel=nan(l_a+l_z,simoptions.simperiods,simoptions.numbersims); % (a,z)

exitinpanel=simoptions.exitinpanel; % reduce overhead with parfor
endogenousexit=simoptions.endogenousexit; % reduce overhead with parfor
if endogenousexit==2
    exitprobabilities=CreateVectorFromParams(Parameters, simoptions.exitprobabilities);
    exitprobs=[1-sum(exitprobabilities),exitprobabilities];
else
    exitprobs=0; % Not sure why, but Matlab was throwing error if this did not exist even when endogenousexit~=2, presumably something to do with figuring out the parallelization for parfor???
end
% parfor ii=1:simoptions.numbersims % Parallel CPUs for the simulations
for ii=1:simoptions.numbersims % Parallel CPUs for the simulations
    seedpoint_ii=ind2sub_homemade([N_a,N_z],seedpointvec(ii));
    seedpoint_ii=round(seedpoint_ii); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

    if exitinpanel==0
        SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_raw(PolicyIndexesKron,l_d,n_a,cumsumpi_z,seedpoint_ii,simoptions);
    else
        if endogenousexit==2 % Mixture of endogenous and exogenous exit
            SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_Exit2_raw(PolicyIndexesKron, CondlProbOfSurvivalKron,N_d,N_a,N_z,cumsumpi_z,simoptions.burnin,seedpoint_ii,simoptions.simperiods,exitprobs,0); % 0: burnin, 0: use single CPU
        else % Otherwise (either one of endogenous or exogenous exit; but not mixture)
            SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_Exit_raw(PolicyIndexesKron, CondlProbOfSurvivalKron,N_d,N_a,N_z,cumsumpi_z,simoptions.burnin,seedpoint_ii,simoptions.simperiods,0); % 0: burnin, 0: use single CPU
        end
    end

    SimPanel_ii=nan(l_a+l_z,simoptions.simperiods);

    for t=1:simoptions.simperiods
        temp=SimTimeSeriesKron(:,t);
        if ~isnan(temp)
            a_c_vec=ind2sub_homemade([n_a],temp(1));
            z_c_vec=ind2sub_homemade([n_z],temp(2));
            for kk=1:l_a
                SimPanel_ii(kk,t)=a_c_vec(kk);
            end
            for kk=1:l_z
                SimPanel_ii(l_a+kk,t)=z_c_vec(kk);
            end
        end
    end
    SimPanel(:,:,ii)=SimPanel_ii;
end


end



