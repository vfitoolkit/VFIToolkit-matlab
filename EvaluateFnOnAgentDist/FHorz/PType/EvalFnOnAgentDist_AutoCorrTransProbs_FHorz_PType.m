function CorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType(StationaryDist, Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid, z_grid, pi_z, simoptions)
% Auto-covariances/-correlations (and transition probabilities) with permanent types.
% Calls EvalFnOnAgentDist_AutoCorrTransProbs_FHorz() for each permanent type, and reports
% the results by type and (by default) grouped over the types.
%
% Inputs follow LifeCycleProfiles_FHorz_Case1_PType: anything that depends on the
% permanent type is given as a structure with a field per type (Parameters, pi_z, z_grid,
% simoptions fields, FnsToEvaluate fields, conditional restrictions) or with a trailing
% dimension of length N_i where that form is accepted. Names_i is the cell of type names,
% or just the number of types. StationaryDist is the structure output of
% StationaryDist_Case1_FHorz_PType (it includes StationaryDist.ptweights).
% N_j must be the same for all types (no per-type N_j, no agejshifter).
%
% See EvalFnOnAgentDist_AutoCorrTransProbs_FHorz for the simoptions (timehorizons,
% conditionalrestrictions, transprobs, transprobquantiles) and the outputs.
%
% Output:
%   CorrTransProbs.(fnname).(typename)   the per-type output of EvalFnOnAgentDist_AutoCorrTransProbs_FHorz
%   CorrTransProbs.(fnname).Mean, .StdDeviation, .AutoCovariance, .AutoCorrelation (and _kK)  grouped
%   CorrTransProbs.(restriction).(fnname).(typename)   per-type restricted output
%   CorrTransProbs.(restriction).(fnname).Mean, ... , .PairMass, .PairMean_j, ...   grouped restricted output
%   CorrTransProbs.(restriction).RestrictedSampleMass.(typename), .ByAge, .ByPType, .Total
%   TransitionProbs are only reported by type.
% Grouped means and std deviations at age j pool the types with weights
% ptweights(ii)*(mass of type ii at age j). Grouped auto-covariances at horizon k pool the
% types' PAIR populations (age j and age j+k) with weights ptweights(ii)*(pair mass of type ii),
% about the pooled pair means; the grouped AutoCorrelation is the correlation of that pooled
% pair population (so when the types have the same age weights it is exactly what a
% single-type model with an extra 'type' state would give). A type of zero weight, or with
% zero mass at an age, is simply not in the pool; if the pool is empty the output is NaN.
% simoptions.groupptypesforstats=0 skips the grouped outputs.

if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i=cell(1,N_i);
    for ii=1:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

if ~exist('simoptions','var')
    simoptions.groupptypesforstats=1;
    simoptions.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu (off by default)
    simoptions.verbose=0;
    simoptions.verboseparams=0;
    simoptions.timehorizons=[];
else
    if ~isfield(simoptions,'groupptypesforstats')
        simoptions.groupptypesforstats=1;
    end
    if ~isfield(simoptions,'ptypestorecpu')
        simoptions.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu (off by default)
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=0;
    end
    if ~isfield(simoptions,'timehorizons')
        simoptions.timehorizons=[];
    end
end

if isstruct(N_j)
    error('EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType: N_j must be the same for all permanent types (N_j cannot be a structure)')
end
if isfield(simoptions,'agejshifter')
    if isstruct(simoptions.agejshifter) || any(simoptions.agejshifter~=0)
        error('EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType: simoptions.agejshifter is not implemented')
    end
end
if isfield(simoptions,'agegroupings')
    if isstruct(simoptions.agegroupings) || ~isequal(simoptions.agegroupings,1:1:N_j)
        error('EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType: simoptions.agegroupings is not implemented')
    end
end

if isstruct(FnsToEvaluate)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnsToEvalNames);
else
    error('You can only use PType when FnsToEvaluate is a structure')
end

useCondlRest=0;
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);
end

% The horizons and the suffixes of the output fields, exactly as in EvalFnOnAgentDist_AutoCorrTransProbs_FHorz
horizons=unique([1,gather(simoptions.timehorizons(:)')]);
nhorizons=length(horizons);
horizonstr=cell(1,nhorizons);
horizonstr{1}='';
for hh=2:nhorizons
    horizonstr{hh}=['_k',num2str(horizons(hh))];
end

ptweights=gather(reshape(StationaryDist.ptweights,[N_i,1]));

CorrTransProbs=struct();
CorrTransProbs_byType=cell(N_i,1); % the per-type outputs, kept for the grouping
FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i);
AgeMasses=zeros(N_i,N_j); % mass of each type at each age (the age weights of that type)

%% Loop over the permanent types
for ii=1:N_i
    iistr=Names_i{ii};

    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,iistr); % Note: already check for existence of simoptions and created it if it was not inputted
    % PType_Options only keeps a structure-valued option when it has a field for this type, so the
    % conditional restrictions (a structure of functions) have to be put back; a restriction may
    % itself be a structure with a field per type.
    if useCondlRest==1
        simoptions_temp.conditionalrestrictions=struct();
        for rr=1:length(CondlRestnFnNames)
            if isstruct(simoptions.conditionalrestrictions.(CondlRestnFnNames{rr}))
                simoptions_temp.conditionalrestrictions.(CondlRestnFnNames{rr})=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr}).(iistr);
            else
                simoptions_temp.conditionalrestrictions.(CondlRestnFnNames{rr})=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
            end
        end
    end

    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end

    if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
        PolicyIndexes_temp=gpuArray(Policy.(iistr));
        StationaryDist_temp=gpuArray(StationaryDist.(iistr));
    else
        PolicyIndexes_temp=Policy.(iistr);
        StationaryDist_temp=StationaryDist.(iistr);
    end

    %% Go through everything which might be dependent on fixed type (PType)
    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

    % Exogenous shocks
    [n_z_temp,z_grid_temp,pi_z_temp,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,pi_z,simoptions_temp,3);

    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);

    if simoptions_temp.verboseparams==1
        fprintf('Parameter values for the current permanent type \n')
        Parameters_temp
    end

    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=length(n_d_temp);
    end
    l_a_temp=length(n_a_temp);
    if prod(n_z_temp)==0
        l_z_temp=0;
    else
        l_z_temp=length(n_z_temp);
    end

    % Which of the FnsToEvaluate are relevant to this type (kept as a structure)
    [FnsToEvaluate_temp,~,~,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;

    AgeMasses(ii,:)=gather(sum(reshape(StationaryDist_temp,[numel(StationaryDist_temp)/N_j,N_j]),1));

    if sum(FnsAndPTypeIndicator_ii)==0
        continue % none of the FnsToEvaluate are relevant to this type
    end

    %% Compute for this type
    CorrTransProbs_ii=EvalFnOnAgentDist_AutoCorrTransProbs_FHorz(StationaryDist_temp,PolicyIndexes_temp,FnsToEvaluate_temp,Parameters_temp,[],n_d_temp,n_a_temp,n_z_temp,N_j,d_grid_temp,a_grid_temp,z_grid_temp,pi_z_temp,simoptions_temp);
    CorrTransProbs_byType{ii}=CorrTransProbs_ii;

    % Store by type
    for ff=1:numFnsToEvaluate
        if FnsAndPTypeIndicator_ii(ff)==1
            CorrTransProbs.(FnsToEvalNames{ff}).(iistr)=CorrTransProbs_ii.(FnsToEvalNames{ff});
        end
    end
    if useCondlRest==1
        for rr=1:length(CondlRestnFnNames)
            CorrTransProbs.(CondlRestnFnNames{rr}).RestrictedSampleMass.(iistr)=CorrTransProbs_ii.(CondlRestnFnNames{rr}).RestrictedSampleMass;
            for ff=1:numFnsToEvaluate
                if FnsAndPTypeIndicator_ii(ff)==1
                    CorrTransProbs.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr)=CorrTransProbs_ii.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff});
                end
            end
        end
    end
end

%% Grouped: pool the types
if simoptions.groupptypesforstats==1
    if simoptions.verbose==1
        fprintf('Permanent type: Grouped Stats \n')
    end

    for ff=1:numFnsToEvaluate
        fn=FnsToEvalNames{ff};

        % Per-type age-j means and std devs, and the pair covariances at each horizon, into arrays (NaN where not relevant/no mass)
        MeanVec=nan(N_i,N_j);
        StdDevVec=nan(N_i,N_j);
        for ii=1:N_i
            if FnsAndPTypeIndicator(ff,ii)==1
                MeanVec(ii,:)=gather(CorrTransProbs_byType{ii}.(fn).Mean);
                StdDevVec(ii,:)=gather(CorrTransProbs_byType{ii}.(fn).StdDeviation);
            end
        end

        % Grouped Mean and StdDeviation at each age: pool over the types with weight ptweights(ii)*AgeMasses(ii,jj)
        MeanG=nan(1,N_j);
        StdDevG=nan(1,N_j);
        for jj=1:N_j
            w=FnsAndPTypeIndicator(ff,:)'.*ptweights.*AgeMasses(:,jj);
            if sum(w)>0
                p=w/sum(w);
                relevant=(w>0); % a type with no weight here may carry NaN, so it is left out explicitly
                MeanG(jj)=sum(p(relevant).*MeanVec(relevant,jj));
                StdDevG(jj)=sqrt(sum(p(relevant).*(StdDevVec(relevant,jj).^2+(MeanVec(relevant,jj)-MeanG(jj)).^2)));
            end
        end
        CorrTransProbs.(fn).Mean=MeanG;
        CorrTransProbs.(fn).StdDeviation=StdDevG;

        % Grouped AutoCovariance and AutoCorrelation at each horizon: pool the pair populations of the types
        % (unrestricted: the pairs of type ii started at age j0 are all of type ii at age j0, mass ptweights(ii)*AgeMasses(ii,j0);
        % their x_j0 has mean Mean(j0) and std dev StdDeviation(j0), their x_{j0+k} has mean Mean(j0+k) and std dev StdDeviation(j0+k))
        for hh=1:nhorizons
            kk=horizons(hh);
            AutoCovG=nan(1,N_j-kk);
            AutoCorrG=nan(1,N_j-kk);
            AutoCovVec=nan(N_i,N_j-kk);
            for ii=1:N_i
                if FnsAndPTypeIndicator(ff,ii)==1
                    AutoCovVec(ii,:)=gather(CorrTransProbs_byType{ii}.(fn).(['AutoCovariance',horizonstr{hh}]));
                end
            end
            for j0=1:N_j-kk
                w=FnsAndPTypeIndicator(ff,:)'.*ptweights.*AgeMasses(:,j0);
                if sum(w)>0
                    p=w/sum(w);
                    relevant=(w>0);
                    mux=sum(p(relevant).*MeanVec(relevant,j0));
                    muy=sum(p(relevant).*MeanVec(relevant,j0+kk));
                    varx=sum(p(relevant).*(StdDevVec(relevant,j0).^2+(MeanVec(relevant,j0)-mux).^2));
                    vary=sum(p(relevant).*(StdDevVec(relevant,j0+kk).^2+(MeanVec(relevant,j0+kk)-muy).^2));
                    AutoCovG(j0)=sum(p(relevant).*(AutoCovVec(relevant,j0)+(MeanVec(relevant,j0)-mux).*(MeanVec(relevant,j0+kk)-muy)));
                    denom=sqrt(varx*vary);
                    if denom>1e-15
                        AutoCorrG(j0)=AutoCovG(j0)/denom;
                    end
                end
            end
            CorrTransProbs.(fn).(['AutoCovariance',horizonstr{hh}])=AutoCovG;
            CorrTransProbs.(fn).(['AutoCorrelation',horizonstr{hh}])=AutoCorrG;
        end

        %% Grouped restricted
        if useCondlRest==1
            for rr=1:length(CondlRestnFnNames)
                rname=CondlRestnFnNames{rr};
                % Per-type restricted masses, means, std devs
                RSMVec=zeros(N_i,N_j); % restricted mass of each type at each age (includes the type's age weights)
                MeanRVec=nan(N_i,N_j);
                StdDevRVec=nan(N_i,N_j);
                for ii=1:N_i
                    if FnsAndPTypeIndicator(ff,ii)==1
                        RSMVec(ii,:)=gather(CorrTransProbs_byType{ii}.(rname).RestrictedSampleMass);
                        MeanRVec(ii,:)=gather(CorrTransProbs_byType{ii}.(rname).(fn).Mean);
                        StdDevRVec(ii,:)=gather(CorrTransProbs_byType{ii}.(rname).(fn).StdDeviation);
                    end
                end
                MeanRG=nan(1,N_j);
                StdDevRG=nan(1,N_j);
                for jj=1:N_j
                    w=FnsAndPTypeIndicator(ff,:)'.*ptweights.*RSMVec(:,jj);
                    if sum(w)>0
                        p=w/sum(w);
                        relevant=(w>0);
                        MeanRG(jj)=sum(p(relevant).*MeanRVec(relevant,jj));
                        StdDevRG(jj)=sqrt(sum(p(relevant).*(StdDevRVec(relevant,jj).^2+(MeanRVec(relevant,jj)-MeanRG(jj)).^2)));
                    end
                end
                CorrTransProbs.(rname).(fn).Mean=MeanRG;
                CorrTransProbs.(rname).(fn).StdDeviation=StdDevRG;

                % Pairs at each horizon: pool with weight ptweights(ii)*PairMass_ii(j0), about the pooled pair means
                for hh=1:nhorizons
                    kk=horizons(hh);
                    PairMassVec=zeros(N_i,N_j-kk);
                    PairMeanxVec=nan(N_i,N_j-kk);
                    PairMeanyVec=nan(N_i,N_j-kk);
                    PairStdxVec=nan(N_i,N_j-kk);
                    PairStdyVec=nan(N_i,N_j-kk);
                    AutoCovRVec=nan(N_i,N_j-kk);
                    for ii=1:N_i
                        if FnsAndPTypeIndicator(ff,ii)==1
                            PairMassVec(ii,:)=gather(CorrTransProbs_byType{ii}.(rname).(fn).(['PairMass',horizonstr{hh}]));
                            PairMeanxVec(ii,:)=gather(CorrTransProbs_byType{ii}.(rname).(fn).(['PairMean_j',horizonstr{hh}]));
                            PairMeanyVec(ii,:)=gather(CorrTransProbs_byType{ii}.(rname).(fn).(['PairMean_jplusk',horizonstr{hh}]));
                            PairStdxVec(ii,:)=gather(CorrTransProbs_byType{ii}.(rname).(fn).(['PairStdDeviation_j',horizonstr{hh}]));
                            PairStdyVec(ii,:)=gather(CorrTransProbs_byType{ii}.(rname).(fn).(['PairStdDeviation_jplusk',horizonstr{hh}]));
                            AutoCovRVec(ii,:)=gather(CorrTransProbs_byType{ii}.(rname).(fn).(['AutoCovariance',horizonstr{hh}]));
                        end
                    end
                    PairMassNaN=isnan(PairMassVec); % NaN where the type has no mass at j0: not in the pool
                    PairMassVec(PairMassNaN)=0;
                    PairMassG=nan(1,N_j-kk);
                    PairMeanxG=nan(1,N_j-kk);
                    PairMeanyG=nan(1,N_j-kk);
                    PairStdxG=nan(1,N_j-kk);
                    PairStdyG=nan(1,N_j-kk);
                    AutoCovRG=nan(1,N_j-kk);
                    AutoCorrRG=nan(1,N_j-kk);
                    for j0=1:N_j-kk
                        w=FnsAndPTypeIndicator(ff,:)'.*ptweights.*PairMassVec(:,j0);
                        if all(PairMassNaN(FnsAndPTypeIndicator(ff,:)==1,j0))
                            PairMassG(j0)=NaN; % no relevant type has any mass at j0
                        else
                            PairMassG(j0)=sum(w);
                        end
                        if sum(w)>0
                            p=w/sum(w);
                            relevant=(w>0);
                            mux=sum(p(relevant).*PairMeanxVec(relevant,j0));
                            muy=sum(p(relevant).*PairMeanyVec(relevant,j0));
                            varx=sum(p(relevant).*(PairStdxVec(relevant,j0).^2+(PairMeanxVec(relevant,j0)-mux).^2));
                            vary=sum(p(relevant).*(PairStdyVec(relevant,j0).^2+(PairMeanyVec(relevant,j0)-muy).^2));
                            PairMeanxG(j0)=mux;
                            PairMeanyG(j0)=muy;
                            PairStdxG(j0)=sqrt(varx);
                            PairStdyG(j0)=sqrt(vary);
                            AutoCovRG(j0)=sum(p(relevant).*(AutoCovRVec(relevant,j0)+(PairMeanxVec(relevant,j0)-mux).*(PairMeanyVec(relevant,j0)-muy)));
                            denom=sqrt(varx*vary);
                            if denom>1e-15
                                AutoCorrRG(j0)=AutoCovRG(j0)/denom;
                            end
                        end
                    end
                    CorrTransProbs.(rname).(fn).(['AutoCovariance',horizonstr{hh}])=AutoCovRG;
                    CorrTransProbs.(rname).(fn).(['AutoCorrelation',horizonstr{hh}])=AutoCorrRG;
                    CorrTransProbs.(rname).(fn).(['PairMass',horizonstr{hh}])=PairMassG;
                    CorrTransProbs.(rname).(fn).(['PairMean_j',horizonstr{hh}])=PairMeanxG;
                    CorrTransProbs.(rname).(fn).(['PairMean_jplusk',horizonstr{hh}])=PairMeanyG;
                    CorrTransProbs.(rname).(fn).(['PairStdDeviation_j',horizonstr{hh}])=PairStdxG;
                    CorrTransProbs.(rname).(fn).(['PairStdDeviation_jplusk',horizonstr{hh}])=PairStdyG;
                end
            end
        end
    end

    % Restricted sample masses of the whole population (as LifeCycleProfiles_FHorz_Case1_PType reports them)
    if useCondlRest==1
        for rr=1:length(CondlRestnFnNames)
            rname=CondlRestnFnNames{rr};
            RSMVec=zeros(N_i,N_j);
            for ii=1:N_i
                if ~isempty(CorrTransProbs_byType{ii})
                    RSMVec(ii,:)=gather(CorrTransProbs_byType{ii}.(rname).RestrictedSampleMass);
                end
            end
            CorrTransProbs.(rname).RestrictedSampleMass.ByAge=sum(RSMVec.*ptweights,1); % mass at each age that satisfies the restriction, as a share of the whole population
            CorrTransProbs.(rname).RestrictedSampleMass.ByPType=sum(RSMVec,2); % conditional on type, what fraction satisfy the restriction
            CorrTransProbs.(rname).RestrictedSampleMass.Total=sum(ptweights.*sum(RSMVec,2)); % what fraction of the population satisfy the restriction
        end
    end
end

CorrTransProbs.Notes='Per type: CorrTransProbs.(fn).(typename) is the output of EvalFnOnAgentDist_AutoCorrTransProbs_FHorz for that type (see its Notes). Grouped (CorrTransProbs.(fn).Mean etc.): Mean and StdDeviation at age j pool the types with weights ptweights*(mass of the type at age j); AutoCovariance and AutoCorrelation (and the _kK horizons) pool the pair populations of the types with weights ptweights*(pair mass), about the pooled pair means, so AutoCorrelation is the correlation of the pooled pair population. The same under each conditional restriction, where PairMass is the population mass of the pairs. TransitionProbs are only reported by type.';

end
