function CorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType(StationaryDist, Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid, z_grid, pi_z, simoptions)
% simoptions.whichcombos ([numFnsToEvaluate, N_j, 1+number of conditional restrictions], optionally with a trailing type dimension of N_i+1)
% selects which (fn, start age, restriction) combinations are computed, per type and grouped; see below.
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
%   CorrTransProbs.(fnname).TransitionProbs (and TransitionValues_j, TransitionValues_jplus1, TransitionMass_j)  grouped,
%     pooling the types' transitions between value bins (see below)
% Grouped means and std deviations at age j pool the types with weights
% ptweights(ii)*(mass of type ii at age j). Grouped auto-covariances at horizon k pool the
% types' PAIR populations (age j and age j+k) with weights ptweights(ii)*(pair mass of type ii),
% about the pooled pair means; the grouped AutoCorrelation is the correlation of that pooled
% pair population (so when the types have the same age weights it is exactly what a
% single-type model with an extra 'type' state would give). A type of zero weight, or with
% zero mass at an age, is simply not in the pool; if the pool is empty the output is NaN.
% Grouped TransitionProbs at age j pool the types' joint distributions of (value at j, value at
% j+1), weight ptweights(ii)*(mass of type ii at age j), over the union of the types' value bins,
% and divide by the pooled mass of the origin bin; with simoptions.transprobquantiles the pooled
% joint is coarsened to the quantile bins of the pooled age-j and age-(j+1) value distributions,
% which is what the single-type command gives on the pooled population.
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

%% simoptions.whichcombos: which (fn, start age, restriction) combinations to compute
% [numFnsToEvaluate, N_j, 1+number of conditional restrictions] of zeros/ones: page 1 is the unrestricted outputs, pages 2:end the
% restrictions in the fieldnames order of simoptions.conditionalrestrictions; the second dimension is the start age of the
% autocovariances. At this (PType wrapper) level the function and the page act directly: a function with nothing selected is not
% evaluated for any type, and a (fn, page) with nothing selected at any age has no per-type and no grouped output (RestrictedSampleMass
% is always filled). The start-age dimension is passed to the per-type command, which acts on it (a one at (ff,j,page) asks for the
% age-j Mean/StdDeviation and the auto-covariances and pair outputs starting at age j); the grouped outputs pool the types' per-age
% outputs, so a skipped start age is NaN in them too. A [numFnsToEvaluate, 1+number of restrictions] or
% [numFnsToEvaluate, N_j] input is expanded over the missing dimension. Default all ones. Intended for calibration/estimation.
if useCondlRest==1
    nwhichpages=1+length(CondlRestnFnNames);
else
    nwhichpages=1;
end
% A trailing type dimension may be added (2026-10-08): [.., N_i+1] selects per permanent type, in the order of Names_i, with the
% last slot the grouped stats. A type's slot selects that type's own computation; the grouped slot forces every type's computation of
% that combination (the grouped means, std devs, pair outputs and auto-covariances at a start age are built from every type's), and a type's output is reported
% whenever it was computed. The grouped output is reported only where the grouped slot asks (the grouped age-j Mean/StdDeviation of a (fn, page)
% wherever any grouped start age of it is on, as they are computed anyway). An input without the type
% dimension applies to every ptype and to the grouped stats. Intended for calibration/estimation.
if ~isfield(simoptions,'whichcombos')
    whichcombosAll=ones(numFnsToEvaluate,N_j,nwhichpages,N_i+1);
else
    whichcombos=simoptions.whichcombos;
    if isstruct(N_j)
        error('simoptions.whichcombos is not implemented when N_j differs by permanent type (N_j a structure)')
    end
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ismatrix(whichcombos) && isequal(size(whichcombos),[numFnsToEvaluate,nwhichpages]) && nwhichpages~=N_j
        whichcombos=repmat(reshape(whichcombos,[numFnsToEvaluate,1,nwhichpages]),[1,N_j,1]); % (fn, page): apply to every start age
    elseif ismatrix(whichcombos) && isequal(size(whichcombos),[numFnsToEvaluate,N_j])
        whichcombos=repmat(whichcombos,[1,1,nwhichpages]); % (fn, start age): apply to every page
    end
    if ndims(whichcombos)<=3 && isequal(size(whichcombos,1:3),[numFnsToEvaluate,N_j,nwhichpages])
        whichcombos=repmat(whichcombos,[1,1,1,N_i+1]); % no type dimension: apply to every ptype and to the grouped stats
    end
    if ~isequal(size(whichcombos,1:4),[numFnsToEvaluate,N_j,nwhichpages,N_i+1])
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(N_j),',',num2str(nwhichpages),'] (number of FnsToEvaluate, N_j, 1+number of conditional restrictions), optionally with a trailing type dimension of ',num2str(N_i+1),' (the permanent types in the order of Names_i, then the grouped stats)'])
    end
    whichcombosAll=double(whichcombos);
end
whichcombosG=whichcombosAll(:,:,:,N_i+1); % the grouped stats; whichcombosAll(:,:,:,ii) is ptype ii
fnwantedG=any(any(whichcombosG,2),3); % numFnsToEvaluate x 1: the functions with some grouped output wanted
pagewantedG=reshape(any(whichcombosG,2),[numFnsToEvaluate,nwhichpages]); % (fn, page) combinations with some grouped start age selected

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
CorrTransProbs_byType_uv=cell(N_i,1); % the per-type unique-value TransitionProbs (the same outputs unless simoptions.transprobquantiles is set), kept for the grouped TransitionProbs
% Which FnsToEvaluate have transition probabilities requested
transprobnames={};
if isfield(simoptions,'transprobs')
    if iscell(simoptions.transprobs)
        transprobnames=simoptions.transprobs;
    elseif any(simoptions.transprobs(:)>0)
        transprobnames=FnsToEvalNames(simoptions.transprobs(:)>0);
    end
end
usequantiles=isfield(simoptions,'transprobquantiles') && ~isempty(simoptions.transprobquantiles);
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
    % This ptype's selection: its own slot, plus whatever the grouped stats need (a grouped slot forces every type's computation)
    whichcombos_ii=max(whichcombosAll(:,:,:,ii),whichcombosG);
    fnwanted_ii=any(any(whichcombos_ii,2),3); % numFnsToEvaluate x 1: the functions this type evaluates
    pagewanted_ii=reshape(any(whichcombos_ii,2),[numFnsToEvaluate,nwhichpages]); % (fn, page) combinations with any start age selected for this type
    % Drop the functions that whichcombos does not want from this type's evaluation
    FnsAndPTypeIndicator_ii=FnsAndPTypeIndicator_ii(:).*fnwanted_ii;
    for ff=1:numFnsToEvaluate
        if fnwanted_ii(ff)==0 && isfield(FnsToEvaluate_temp,FnsToEvalNames{ff})
            FnsToEvaluate_temp=rmfield(FnsToEvaluate_temp,FnsToEvalNames{ff});
        end
    end
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;

    AgeMasses(ii,:)=gather(sum(reshape(StationaryDist_temp,[numel(StationaryDist_temp)/N_j,N_j]),1));

    if sum(FnsAndPTypeIndicator_ii)==0
        continue % none of the FnsToEvaluate are relevant to this type
    end

    %% Compute for this type
    simoptions_temp.whichcombos=whichcombos_ii(FnsAndPTypeIndicator_ii==1,:,:); % the selection for the functions this type evaluates, in their order (the per-type command acts on the start-age dimension)
    CorrTransProbs_ii=EvalFnOnAgentDist_AutoCorrTransProbs_FHorz(StationaryDist_temp,PolicyIndexes_temp,FnsToEvaluate_temp,Parameters_temp,[],n_d_temp,n_a_temp,n_z_temp,N_j,d_grid_temp,a_grid_temp,z_grid_temp,pi_z_temp,simoptions_temp);
    CorrTransProbs_byType{ii}=CorrTransProbs_ii;
    if simoptions.groupptypesforstats==1 && usequantiles && ~isempty(transprobnames)
        % The grouped TransitionProbs are pooled from the types' transitions between unique values (the quantile bins of the
        % pooled population are a coarsening of its value bins; the types' own quantile bins are not), so get those too, for
        % the requested functions only and without the horizons and restrictions
        simoptions_temp_uv=simoptions_temp;
        simoptions_temp_uv=rmfield(simoptions_temp_uv,'whichcombos'); % the grouped TransitionProbs pool every age
        simoptions_temp_uv.transprobquantiles=[];
        simoptions_temp_uv.timehorizons=[];
        if isfield(simoptions_temp_uv,'conditionalrestrictions')
            simoptions_temp_uv=rmfield(simoptions_temp_uv,'conditionalrestrictions');
        end
        FnsToEvaluate_temp_uv=struct();
        fnames_temp=fieldnames(FnsToEvaluate_temp);
        for ff=1:length(fnames_temp)
            if any(strcmp(transprobnames,fnames_temp{ff}))
                FnsToEvaluate_temp_uv.(fnames_temp{ff})=FnsToEvaluate_temp.(fnames_temp{ff});
            end
        end
        if ~isempty(fieldnames(FnsToEvaluate_temp_uv))
            CorrTransProbs_byType_uv{ii}=EvalFnOnAgentDist_AutoCorrTransProbs_FHorz(StationaryDist_temp,PolicyIndexes_temp,FnsToEvaluate_temp_uv,Parameters_temp,[],n_d_temp,n_a_temp,n_z_temp,N_j,d_grid_temp,a_grid_temp,z_grid_temp,pi_z_temp,simoptions_temp_uv);
        end
    else
        CorrTransProbs_byType_uv{ii}=CorrTransProbs_ii;
    end

    % Store by type
    for ff=1:numFnsToEvaluate
        if FnsAndPTypeIndicator_ii(ff)==1 && pagewanted_ii(ff,1)
            CorrTransProbs.(FnsToEvalNames{ff}).(iistr)=CorrTransProbs_ii.(FnsToEvalNames{ff});
        end
    end
    if useCondlRest==1
        for rr=1:length(CondlRestnFnNames)
            CorrTransProbs.(CondlRestnFnNames{rr}).RestrictedSampleMass.(iistr)=CorrTransProbs_ii.(CondlRestnFnNames{rr}).RestrictedSampleMass;
            for ff=1:numFnsToEvaluate
                if FnsAndPTypeIndicator_ii(ff)==1 && pagewanted_ii(ff,1+rr)
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
        if ~fnwantedG(ff) % no grouped combination of this function is wanted
            continue
        end
        if pagewantedG(ff,1) % the grouped unrestricted outputs of this function are wanted

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
            off=reshape(whichcombosG(ff,1:N_j-kk,1)==0,[1,N_j-kk]); % start ages the grouped slot does not ask for
            AutoCovG(off)=NaN;
            AutoCorrG(off)=NaN;
            CorrTransProbs.(fn).(['AutoCovariance',horizonstr{hh}])=AutoCovG;
            CorrTransProbs.(fn).(['AutoCorrelation',horizonstr{hh}])=AutoCorrG;
        end

        %% Grouped TransitionProbs (when requested for this fn): pool the types' transitions between value bins
        % The joint distribution of (value at age jj, value at age jj+1) of type ii is w_ii*TransitionMass_j{jj}(b)*TransitionProbs{jj}(b,:)
        % with w_ii=ptweights(ii)*AgeMasses(ii,jj). The grouped joint is their sum over the union of the types' value bins, and
        % the grouped TransitionProbs row b is that joint divided by the pooled mass of origin bin b (a zero row when it has no
        % mass, as in the single-type command). With simoptions.transprobquantiles the pooled joint is coarsened to the quantile
        % bins of the pooled age-jj and age-(jj+1) value distributions (bin q holds the values up to the first sorted value whose
        % cumulative mass exceeds q/n, the single-type command's rule), which is what that command gives on the pooled population.
        % [A type's age-(jj+1) value distribution is its pushed age-jj one, the column sums of its joint, since within a type the
        % age weights only rescale; across types the age-(jj+1) weights are ptweights(ii)*AgeMasses(ii,jj+1).]
        if any(strcmp(transprobnames,fn))
            P_G_cell=cell(N_j-1,1);
            fvals_j_G=cell(N_j-1,1);
            fvals_jplus1_G=cell(N_j-1,1);
            massbin_j_G=cell(N_j-1,1);
            if usequantiles
                n_fvals=simoptions.transprobquantiles;
                P_G_3d=nan(n_fvals,n_fvals,N_j-1);
            end
            for jj=1:N_j-1
                if whichcombosG(ff,jj,1)==0
                    continue % the grouped slot does not ask for start age jj: the cell stays empty (NaN slice under quantiles)
                end
                w=FnsAndPTypeIndicator(ff,:)'.*ptweights.*AgeMasses(:,jj);
                wnext=FnsAndPTypeIndicator(ff,:)'.*ptweights.*AgeMasses(:,jj+1);
                inpool=false(N_i,1);
                for ii=1:N_i
                    if w(ii)>0
                        inpool(ii)=~isempty(CorrTransProbs_byType_uv{ii}.(fn).TransitionProbs{jj});
                    end
                end
                if ~any(inpool)
                    continue % no relevant type has mass at age jj: the cell stays empty (NaN slice under quantiles), as in the single-type command
                end
                % the union of the types' value bins at jj and at jj+1
                vals_j_cell=cell(N_i,1);
                vals_jp_cell=cell(N_i,1);
                for ii=1:N_i
                    if inpool(ii)
                        vals_j_cell{ii}=gather(CorrTransProbs_byType_uv{ii}.(fn).TransitionValues_j{jj}(:));
                        vals_jp_cell{ii}=gather(CorrTransProbs_byType_uv{ii}.(fn).TransitionValues_jplus1{jj}(:));
                    end
                end
                vals_j=unique(vertcat(vals_j_cell{:}));
                vals_jp=unique(vertcat(vals_jp_cell{:}));
                J_G=zeros(length(vals_j),length(vals_jp)); % the pooled joint distribution (mass sum(w))
                mass_j_G=zeros(length(vals_j),1); % pooled mass of each origin bin at age jj
                mass_jp_G=zeros(length(vals_jp),1); % pooled mass of each value bin at age jj+1 (with the age-(jj+1) weights)
                for ii=1:N_i
                    if inpool(ii)
                        [~,rowmap]=ismember(vals_j_cell{ii},vals_j);
                        [~,colmap]=ismember(vals_jp_cell{ii},vals_jp);
                        m_ii=gather(CorrTransProbs_byType_uv{ii}.(fn).TransitionMass_j{jj}(:));
                        J_ii=m_ii.*gather(CorrTransProbs_byType_uv{ii}.(fn).TransitionProbs{jj}); % the type's joint, mass one
                        J_G(rowmap,colmap)=J_G(rowmap,colmap)+w(ii)*J_ii;
                        mass_j_G(rowmap)=mass_j_G(rowmap)+w(ii)*m_ii;
                        mass_jp_G(colmap)=mass_jp_G(colmap)+wnext(ii)*sum(J_ii,1)';
                    end
                end
                mass_j_safe=mass_j_G;
                mass_j_safe(mass_j_G==0)=1; % an origin bin with no mass gets a row of zeros
                P_G_cell{jj}=J_G./mass_j_safe;
                fvals_j_G{jj}=vals_j;
                fvals_jplus1_G{jj}=vals_jp;
                massbin_j_G{jj}=mass_j_G/sum(w); % within-age share of the pooled population
                if usequantiles && sum(wnext)>0
                    cum_j=cumsum(mass_j_G)/sum(w); % vals_j is sorted
                    idx_j=ones(length(vals_j),1);
                    for qq=1:n_fvals-1
                        cutoff=vals_j(find(cum_j>qq/n_fvals,1,'first'));
                        idx_j(vals_j>cutoff)=qq+1;
                    end
                    cum_jp=cumsum(mass_jp_G)/sum(wnext);
                    idx_jp=ones(length(vals_jp),1);
                    for qq=1:n_fvals-1
                        cutoff=vals_jp(find(cum_jp>qq/n_fvals,1,'first'));
                        idx_jp(vals_jp>cutoff)=qq+1;
                    end
                    S_j=sparse(1:length(vals_j),idx_j,1,length(vals_j),n_fvals);
                    S_jp=sparse(1:length(vals_jp),idx_jp,1,length(vals_jp),n_fvals);
                    massq=full(S_j'*mass_j_G);
                    massq(massq==0)=1; % a quantile bin with no mass gets a row of zeros
                    P_G_3d(:,:,jj)=full(S_j'*J_G*S_jp)./massq;
                end
            end
            if usequantiles
                CorrTransProbs.(fn).TransitionProbs=P_G_3d;
            else
                CorrTransProbs.(fn).TransitionProbs=P_G_cell;
                CorrTransProbs.(fn).TransitionValues_j=fvals_j_G;
                CorrTransProbs.(fn).TransitionValues_jplus1=fvals_jplus1_G;
                CorrTransProbs.(fn).TransitionMass_j=massbin_j_G;
            end
        end
        end % pagewantedG(ff,1)

        %% Grouped restricted
        if useCondlRest==1
            for rr=1:length(CondlRestnFnNames)
                if pagewantedG(ff,1+rr) % the grouped outputs of this restriction are wanted for this function
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
                    off=reshape(whichcombosG(ff,1:N_j-kk,1+rr)==0,[1,N_j-kk]); % start ages the grouped slot does not ask for
                    AutoCovRG(off)=NaN; AutoCorrRG(off)=NaN; PairMassG(off)=NaN; PairMeanxG(off)=NaN; PairMeanyG(off)=NaN; PairStdxG(off)=NaN; PairStdyG(off)=NaN;
                    CorrTransProbs.(rname).(fn).(['AutoCovariance',horizonstr{hh}])=AutoCovRG;
                    CorrTransProbs.(rname).(fn).(['AutoCorrelation',horizonstr{hh}])=AutoCorrRG;
                    CorrTransProbs.(rname).(fn).(['PairMass',horizonstr{hh}])=PairMassG;
                    CorrTransProbs.(rname).(fn).(['PairMean_j',horizonstr{hh}])=PairMeanxG;
                    CorrTransProbs.(rname).(fn).(['PairMean_jplusk',horizonstr{hh}])=PairMeanyG;
                    CorrTransProbs.(rname).(fn).(['PairStdDeviation_j',horizonstr{hh}])=PairStdxG;
                    CorrTransProbs.(rname).(fn).(['PairStdDeviation_jplusk',horizonstr{hh}])=PairStdyG;
                end
                end % pagewantedG(ff,1+rr)
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

CorrTransProbs.Notes='Per type: CorrTransProbs.(fn).(typename) is the output of EvalFnOnAgentDist_AutoCorrTransProbs_FHorz for that type (see its Notes). Grouped (CorrTransProbs.(fn).Mean etc.): Mean and StdDeviation at age j pool the types with weights ptweights*(mass of the type at age j); AutoCovariance and AutoCorrelation (and the _kK horizons) pool the pair populations of the types with weights ptweights*(pair mass), about the pooled pair means, so AutoCorrelation is the correlation of the pooled pair population. The same under each conditional restriction, where PairMass is the population mass of the pairs. Grouped TransitionProbs (when requested) pool the joint distributions of (value at j, value at j+1) of the types with weights ptweights*(mass of the type at age j) over the union of the value bins of the types, each row divided by the pooled mass of its origin bin (TransitionValues_j, TransitionValues_jplus1 label the union bins and TransitionMass_j is their within-age share of the pooled population); with simoptions.transprobquantiles the pooled joint is coarsened to the quantile bins of the pooled age-j and age-(j+1) value distributions.';

end
