function CorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz_PType(StationaryDist, Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, Names_i, d_grid, a_grid, z_grid, pi_z, simoptions)
% Auto-covariances/-correlations (and transition probabilities) with permanent types. Calls
% EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz() for each permanent type, and reports the results by type and (by default) grouped
% over the types. (The InfHorz counterpart of EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType.)
%
% Inputs follow ValueFnIter_InfHorz_PType: anything that depends on the permanent type is given as a structure with a field per
% type (Parameters, n_z, z_grid, pi_z, simoptions fields, FnsToEvaluate fields, conditional restrictions) or with one value per
% type where that form is accepted. Names_i is the cell of type names, or just the number of types. StationaryDist is the
% structure output of StationaryDist_InfHorz_PType (it includes StationaryDist.ptweights).
%
% See EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz for the simoptions (timehorizons, conditionalrestrictions, transprobs,
% transprobquantiles) and the outputs.
%
% Output:
%   CorrTransProbs.(fnname).(typename)   the per-type output of EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz (TransitionProbs
%                                         included, when asked for)
%   CorrTransProbs.(fnname).Mean, .StdDeviation, .AutoCovariance, .AutoCorrelation (and .tperiodsK.AutoCovariance/AutoCorrelation)  grouped
%   CorrTransProbs.(restriction).(fnname).(typename)   per-type restricted output
%   CorrTransProbs.(restriction).(fnname).Mean, .StdDeviation, .AutoCovariance, .AutoCorrelation, .PairMass, .PairMean_t,
%       .PairMean_tplusk, .PairStdDeviation_t, .PairStdDeviation_tplusk (and the same under .tperiodsK)   grouped restricted output
%   CorrTransProbs.(restriction).RestrictedSampleMass.(typename), .ByPType, .Total
% Grouped, unrestricted: the types are pooled with weights ptweights (each type's distribution has mass one). Each type's
% distribution is stationary, so its mean is the same now and k periods on, and the grouped auto-covariance at horizon k is
% sum_i ptweights(i)*(AutoCovariance_i + (Mean_i-Mean)^2) about the pooled Mean; the grouped auto-correlation is that over the
% pooled variance (so when the stationary distributions are exact, this is what a single-type model with the type as an extra,
% never-changing, state would give).
% Grouped, restricted: the Mean and StdDeviation pool the types with weights ptweights times their restricted sample mass; the
% auto-covariance pools the types' pair populations (those satisfying the restriction now and k periods on), weights ptweights
% times their PairMass, about the pooled pair means, so the grouped PairMass is sum_i ptweights(i)*PairMass_i (a share of the whole
% population) and the grouped AutoCorrelation is the correlation of the pooled pair population.
% A type of zero weight, or with no (restricted) mass, is not in the pool; if the pool is empty the output is NaN.
% The TransitionProbs are reported per type only (they are between the type's own value bins).
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
    simoptions.verbose=0;
    simoptions.verboseparams=0;
    simoptions.timehorizons=[];
else
    if ~isfield(simoptions,'groupptypesforstats')
        simoptions.groupptypesforstats=1;
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

% The horizons, and where their outputs are, exactly as in EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz
horizons=unique([1,gather(simoptions.timehorizons(:)')]);
nhorizons=length(horizons);

ptweights=gather(reshape(StationaryDist.ptweights,[N_i,1]));

CorrTransProbs=struct();
CorrTransProbs_byType=cell(N_i,1); % the per-type outputs, kept for the grouping
FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i);

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

    PolicyIndexes_temp=gpuArray(Policy.(iistr)); % (in case the solutions are stored on the cpu)
    StationaryDist_temp=gpuArray(StationaryDist.(iistr));

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
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii(:);
    if sum(FnsAndPTypeIndicator_ii)==0
        continue % none of the FnsToEvaluate are relevant to this type
    end
    % (simoptions.transprobs given as names: keep only those relevant to this type)
    if isfield(simoptions_temp,'transprobs') && iscell(simoptions_temp.transprobs)
        simoptions_temp.transprobs=simoptions_temp.transprobs(isfield(FnsToEvaluate_temp,simoptions_temp.transprobs));
    end

    %% Compute for this type
    CorrTransProbs_ii=EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz(StationaryDist_temp,PolicyIndexes_temp,FnsToEvaluate_temp,Parameters_temp,[],n_d_temp,n_a_temp,n_z_temp,d_grid_temp,a_grid_temp,z_grid_temp,pi_z_temp,simoptions_temp);
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

%% Restricted sample masses across the types
if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        RSMVec=nan(N_i,1);
        for ii=1:N_i
            if ~isempty(CorrTransProbs_byType{ii})
                RSMVec(ii)=gather(CorrTransProbs_byType{ii}.(CondlRestnFnNames{rr}).RestrictedSampleMass);
            end
        end
        CorrTransProbs.(CondlRestnFnNames{rr}).RestrictedSampleMass.ByPType=RSMVec; % conditional on type, what fraction satisfy the restriction
        CorrTransProbs.(CondlRestnFnNames{rr}).RestrictedSampleMass.Total=sum(ptweights.*RSMVec); % what fraction of the population satisfy the restriction
    end
end

%% Grouped: pool the types
if simoptions.groupptypesforstats==1
    if simoptions.verbose==1
        fprintf('Permanent type: Grouped Stats \n')
    end
    for ff=1:numFnsToEvaluate
        fn=FnsToEvalNames{ff};
        %% Unrestricted
        MeanVec=nan(N_i,1); StdDevVec=nan(N_i,1); AutoCovVec=nan(N_i,nhorizons);
        for ii=1:N_i
            if FnsAndPTypeIndicator(ff,ii)==1
                Q=CorrTransProbs_byType{ii}.(fn);
                MeanVec(ii)=gather(Q.Mean); StdDevVec(ii)=gather(Q.StdDeviation);
                for hh=1:nhorizons
                    if horizons(hh)==1
                        AutoCovVec(ii,hh)=gather(Q.AutoCovariance);
                    else
                        AutoCovVec(ii,hh)=gather(Q.(['tperiods',num2str(horizons(hh))]).AutoCovariance);
                    end
                end
            end
        end
        w=FnsAndPTypeIndicator(ff,:)'.*ptweights;
        relevant=(w>0);
        MeanG=NaN; VarG=NaN; AutoCovG=nan(1,nhorizons);
        if any(relevant)
            p=w(relevant)/sum(w(relevant));
            MeanG=sum(p.*MeanVec(relevant));
            VarG=sum(p.*(StdDevVec(relevant).^2+(MeanVec(relevant)-MeanG).^2));
            for hh=1:nhorizons
                AutoCovG(hh)=sum(p.*(AutoCovVec(relevant,hh)+(MeanVec(relevant)-MeanG).^2)); % each type is stationary: its mean is the same now and k periods on
            end
        end
        CorrTransProbs.(fn).Mean=MeanG;
        CorrTransProbs.(fn).StdDeviation=sqrt(VarG);
        for hh=1:nhorizons
            if horizons(hh)==1
                CorrTransProbs.(fn).AutoCovariance=AutoCovG(hh);
                CorrTransProbs.(fn).AutoCorrelation=AutoCovG(hh)/VarG;
            else
                CorrTransProbs.(fn).(['tperiods',num2str(horizons(hh))]).AutoCovariance=AutoCovG(hh);
                CorrTransProbs.(fn).(['tperiods',num2str(horizons(hh))]).AutoCorrelation=AutoCovG(hh)/VarG;
            end
        end

        %% Restricted
        if useCondlRest==1
            for rr=1:length(CondlRestnFnNames)
                rname=CondlRestnFnNames{rr};
                MeanVec=nan(N_i,1); StdDevVec=nan(N_i,1); RSMVec=zeros(N_i,1);
                PairMassVec=zeros(N_i,nhorizons); MXVec=nan(N_i,nhorizons); MYVec=nan(N_i,nhorizons); SXVec=nan(N_i,nhorizons); SYVec=nan(N_i,nhorizons); CVec=nan(N_i,nhorizons);
                for ii=1:N_i
                    if FnsAndPTypeIndicator(ff,ii)==1
                        Q=CorrTransProbs_byType{ii}.(rname).(fn);
                        RSMVec(ii)=gather(CorrTransProbs_byType{ii}.(rname).RestrictedSampleMass);
                        MeanVec(ii)=gather(Q.Mean); StdDevVec(ii)=gather(Q.StdDeviation);
                        for hh=1:nhorizons
                            if horizons(hh)==1
                                Qh=Q;
                            else
                                Qh=Q.(['tperiods',num2str(horizons(hh))]);
                            end
                            PairMassVec(ii,hh)=gather(Qh.PairMass);
                            MXVec(ii,hh)=gather(Qh.PairMean_t); MYVec(ii,hh)=gather(Qh.PairMean_tplusk);
                            SXVec(ii,hh)=gather(Qh.PairStdDeviation_t); SYVec(ii,hh)=gather(Qh.PairStdDeviation_tplusk);
                            CVec(ii,hh)=gather(Qh.AutoCovariance);
                        end
                    end
                end
                % Mean and StdDeviation: weights ptweights times the restricted sample mass
                w=FnsAndPTypeIndicator(ff,:)'.*ptweights.*RSMVec;
                relevant=(w>0);
                MeanG=NaN; StdDevG=NaN;
                if any(relevant)
                    p=w(relevant)/sum(w(relevant));
                    MeanG=sum(p.*MeanVec(relevant));
                    StdDevG=sqrt(sum(p.*(StdDevVec(relevant).^2+(MeanVec(relevant)-MeanG).^2)));
                end
                CorrTransProbs.(rname).(fn).Mean=MeanG;
                CorrTransProbs.(rname).(fn).StdDeviation=StdDevG;
                % The pair populations: weights ptweights times the PairMass
                for hh=1:nhorizons
                    w=FnsAndPTypeIndicator(ff,:)'.*ptweights.*PairMassVec(:,hh);
                    relevant=(w>0);
                    PairMassG=sum(w);
                    mx=NaN; my=NaN; sx=NaN; sy=NaN; covxy=NaN; corrxy=NaN;
                    if any(relevant)
                        p=w(relevant)/sum(w(relevant));
                        mx=sum(p.*MXVec(relevant,hh));
                        my=sum(p.*MYVec(relevant,hh));
                        sx=sqrt(sum(p.*(SXVec(relevant,hh).^2+(MXVec(relevant,hh)-mx).^2)));
                        sy=sqrt(sum(p.*(SYVec(relevant,hh).^2+(MYVec(relevant,hh)-my).^2)));
                        covxy=sum(p.*(CVec(relevant,hh)+(MXVec(relevant,hh)-mx).*(MYVec(relevant,hh)-my)));
                        if sx*sy>1e-15
                            corrxy=covxy/(sx*sy);
                        end
                    end
                    if horizons(hh)==1
                        CorrTransProbs.(rname).(fn).AutoCovariance=covxy;
                        CorrTransProbs.(rname).(fn).AutoCorrelation=corrxy;
                        CorrTransProbs.(rname).(fn).PairMass=PairMassG;
                        CorrTransProbs.(rname).(fn).PairMean_t=mx;
                        CorrTransProbs.(rname).(fn).PairMean_tplusk=my;
                        CorrTransProbs.(rname).(fn).PairStdDeviation_t=sx;
                        CorrTransProbs.(rname).(fn).PairStdDeviation_tplusk=sy;
                    else
                        hstr=['tperiods',num2str(horizons(hh))];
                        CorrTransProbs.(rname).(fn).(hstr).AutoCovariance=covxy;
                        CorrTransProbs.(rname).(fn).(hstr).AutoCorrelation=corrxy;
                        CorrTransProbs.(rname).(fn).(hstr).PairMass=PairMassG;
                        CorrTransProbs.(rname).(fn).(hstr).PairMean_t=mx;
                        CorrTransProbs.(rname).(fn).(hstr).PairMean_tplusk=my;
                        CorrTransProbs.(rname).(fn).(hstr).PairStdDeviation_t=sx;
                        CorrTransProbs.(rname).(fn).(hstr).PairStdDeviation_tplusk=sy;
                    end
                end
            end
        end
    end
end

end
