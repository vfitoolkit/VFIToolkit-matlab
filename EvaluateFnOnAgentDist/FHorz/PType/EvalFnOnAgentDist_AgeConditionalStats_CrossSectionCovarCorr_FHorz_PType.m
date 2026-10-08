function AgeConditionalCrossSectionCorr=EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz_PType(StationaryDist, Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid, z_grid, simoptions)
% simoptions.whichcombos ([numFnsToEvaluate, numFnsToEvaluate, number of age groups], with a page dimension of 1+number of conditional
% restrictions when there are restrictions, optionally with a trailing type dimension of N_i+1: diagonal = a function's own
% Mean/StdDeviation, off-diagonal = a pair, third dimension = age group) selects which functions, pairs, age groups and restrictions
% are computed; see below.
% Age-conditional cross-sectional covariances/correlations between every pair of
% FnsToEvaluate, with permanent types. Calls
% EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz() for each permanent
% type, and reports the results by type and (by default) grouped over the types.
%
% Inputs follow LifeCycleProfiles_FHorz_Case1_PType: anything that depends on the
% permanent type is given as a structure with a field per type (Parameters, z_grid,
% simoptions fields, FnsToEvaluate fields) or with a trailing dimension of length N_i where
% that form is accepted. Names_i is the cell of type names, or just the number of types.
% StationaryDist is the structure output of StationaryDist_Case1_FHorz_PType (it includes
% StationaryDist.ptweights). N_j and simoptions.agegroupings must be the same for all
% types (no per-type N_j, no agejshifter).
%
% simoptions.agegroupings can be used to do conditional on 'age bins' rather than age
% e.g., simoptions.agegroupings=1:10:N_j will divide into 10 year age bins.
%
% Output (ngroups=length(simoptions.agegroupings)):
%   AgeConditionalCrossSectionCorr.(fn1).(typename).Mean, .StdDeviation, .CovarianceWith.(fn2), .CorrelationWith.(fn2)   by type, 1 x ngroups
%   AgeConditionalCrossSectionCorr.CovarianceMatrix_ptype.(typename), .CorrelationMatrix_ptype.(typename)   by type (nFn x nFn x ngroups)
%   AgeConditionalCrossSectionCorr.(fn1).Mean, .StdDeviation, .CovarianceWith.(fn2), .CorrelationWith.(fn2)   grouped, 1 x ngroups
%   AgeConditionalCrossSectionCorr.CovarianceMatrix, .CorrelationMatrix   grouped (nFn x nFn x ngroups)
% Grouped: in each age group the types are pooled with weights ptweights(ii)*(mass of type
% ii in the age group), so the grouped mean is the weighted mean, the grouped
% variance/covariance is the weighted within-type variance/covariance plus the between-type
% part (about the pooled means), and the grouped correlation is covariance/(std dev * std
% dev) of the pooled population. A pair of functions is pooled over the types for which
% both are relevant. A type with no mass in an age group is not in that pool; an empty pool
% gives NaN (the self-correlation included). simoptions.groupptypesforstats=0 skips the grouped outputs.
%
% With simoptions.conditionalrestrictions (a restriction may be a structure with a field per type), the same per type and grouped
% under each restriction, in AgeConditionalCrossSectionCorr.(restriction): the per-type output is the type's restricted output of
% the single-type command (normalised within each age group), and in each age group the types are pooled with weights
% ptweights(ii)*(restricted mass of type ii in the age group). The restricted sample masses are
%   AgeConditionalCrossSectionCorr.(restriction).RestrictedSampleMass.(typename)   1-by-N_j, the mass of the type at each age that satisfies it
%   .RestrictedSampleMass.ByAge (1-by-N_j, a share of the whole population), .ByPType (N_i-by-1), .Total   as in LifeCycleProfiles_FHorz_Case1_PType

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

if isstruct(N_j)
    error('EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz_PType: N_j must be the same for all permanent types (N_j cannot be a structure)')
end

if ~exist('simoptions','var')
    simoptions.groupptypesforstats=1;
    simoptions.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu (off by default)
    simoptions.verbose=0;
    simoptions.verboseparams=0;
    simoptions.agegroupings=1:1:N_j; % by default does each period separately
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
    if ~isfield(simoptions,'agegroupings')
        simoptions.agegroupings=1:1:N_j; % by default does each period separately
    end
end
if isstruct(simoptions.agegroupings)
    error('EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz_PType: simoptions.agegroupings must be the same for all permanent types (cannot be a structure)')
end
if isfield(simoptions,'agejshifter')
    if isstruct(simoptions.agejshifter) || any(simoptions.agejshifter~=0)
        error('EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz_PType: simoptions.agejshifter is not implemented')
    end
end
ngroups=length(simoptions.agegroupings);

useCondlRest=0;
nwhichpages=1;
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);
    nwhichpages=1+length(CondlRestnFnNames);
end

if isstruct(FnsToEvaluate)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnsToEvalNames);
else
    error('You can only use PType when FnsToEvaluate is a structure')
end

%% simoptions.whichcombos: which functions, pairs, age groups and restrictions to compute
% [numFnsToEvaluate, numFnsToEvaluate, ngroups] of zeros/ones. In each age group the diagonal (ff,ff) selects the grouped Mean and
% StdDeviation of function ff and the off-diagonal (ff1,ff2) selects the covariance and correlation of the pair (the own stats of every
% function evaluated in an age group are reported, as byproducts of the pairs, so the diagonal matters only for a function with no selected pair there); only the upper
% triangle (ff1<=ff2) is read, so a symmetric matrix or just its upper triangle can be given. A [numFnsToEvaluate, numFnsToEvaluate]
% input is applied to every age group. A function with nothing selected (its diagonal and all its pairs zero in every age group) is
% not evaluated at all; a selected pair has both its functions evaluated. Skipped entries are NaN: in the grouped output, and in the
% per-type pair fields and per-type matrices (the per-type Mean and StdDeviation of an evaluated function are reported regardless,
% the single-type command computes them anyway). Default all ones.
% With conditional restrictions a page dimension follows: [numFnsToEvaluate, numFnsToEvaluate, ngroups, 1+number of conditional
% restrictions], page 1 the unrestricted stats, pages 2:end the restrictions in the fieldnames order of
% simoptions.conditionalrestrictions; an input without it applies to every page.
% A trailing type dimension may be added (2026-10-08): [.., N_i+1] selects per permanent type, in the order of Names_i, with the
% last slot the grouped stats (so [nFns, nFns, ngroups, 1+number of restrictions, N_i+1] with restrictions, [nFns, nFns, ngroups, N_i+1]
% without). A type's slot selects that type's own computation; the grouped slot forces every type's computation of
% that combination (the grouped means, std devs and covariances are built from every type's), and a type's output is reported
% whenever it was computed. The grouped output is reported only where the grouped slot asks (the grouped own stats of a function
% wherever any grouped entry of that function's row is on, as they are byproducts of the pairs). An input without the type
% dimension applies to every ptype and to the grouped stats. With restrictions, a four-dimensional input is read as having the page
% dimension, not the type dimension. Intended for calibration/estimation.
if ~isfield(simoptions,'whichcombos')
    whichcombosAll=ones(numFnsToEvaluate,numFnsToEvaluate,ngroups,nwhichpages,N_i+1);
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ismatrix(whichcombos) && isequal(size(whichcombos),[numFnsToEvaluate,numFnsToEvaluate])
        whichcombos=repmat(whichcombos,[1,1,ngroups]); % one matrix: apply to every age group
    end
    if ndims(whichcombos)<=3 && isequal(size(whichcombos,1:3),[numFnsToEvaluate,numFnsToEvaluate,ngroups])
        whichcombos=repmat(whichcombos,[1,1,1,nwhichpages]); % no page dimension: apply to every page
    end
    if nwhichpages==1 && ndims(whichcombos)==4 && isequal(size(whichcombos),[numFnsToEvaluate,numFnsToEvaluate,ngroups,N_i+1])
        whichcombos=reshape(whichcombos,[numFnsToEvaluate,numFnsToEvaluate,ngroups,1,N_i+1]); % no restrictions: the fourth dimension is the type dimension
    elseif ndims(whichcombos)<=4 && isequal(size(whichcombos,1:4),[numFnsToEvaluate,numFnsToEvaluate,ngroups,nwhichpages])
        whichcombos=repmat(whichcombos,[1,1,1,1,N_i+1]); % no type dimension: apply to every ptype and to the grouped stats
    end
    if ~isequal(size(whichcombos,1:5),[numFnsToEvaluate,numFnsToEvaluate,ngroups,nwhichpages,N_i+1]) || ndims(whichcombos)>5
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(numFnsToEvaluate),',',num2str(ngroups),',',num2str(nwhichpages),'] (number of FnsToEvaluate, twice, number of age groups, 1+number of conditional restrictions; the fourth dimension is dropped when there are no conditional restrictions), optionally with a trailing type dimension of ',num2str(N_i+1),' (the permanent types in the order of Names_i, then the grouped stats)'])
    end
    whichcombosAll=double(whichcombos);
    for ss=1:N_i+1
        for pp=1:nwhichpages
            for kk=1:ngroups
                whichcombosAll(:,:,kk,pp,ss)=max(triu(whichcombosAll(:,:,kk,pp,ss)),triu(whichcombosAll(:,:,kk,pp,ss))'); % symmetric, from the upper triangle
            end
        end
    end
end
whichcombosG=whichcombosAll(:,:,:,:,N_i+1); % the grouped stats; whichcombosAll(:,:,:,:,ii) is ptype ii

ptweights=gather(reshape(StationaryDist.ptweights,[N_i,1]));

% Each page is filled as its own output structure; page 1 (unrestricted) becomes AgeConditionalCrossSectionCorr and pages 2:end AgeConditionalCrossSectionCorr.(restrictionname)
PageOut=cell(nwhichpages,1);
for pp=1:nwhichpages
    PageOut{pp}=struct();
end
FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i);
MeanVec=nan(numFnsToEvaluate,N_i,ngroups,nwhichpages);
StdDevVec=nan(numFnsToEvaluate,N_i,ngroups,nwhichpages);
CovarVec=nan(numFnsToEvaluate,numFnsToEvaluate,N_i,ngroups,nwhichpages);
GroupMasses=zeros(N_i,ngroups,nwhichpages); % mass of each type in each age group (from that type's age weights), and in the restriction for pages 2:end
RestrictedMassByAge=nan(N_i,N_j,nwhichpages-1); % the per-type restricted sample masses at each age

%% Loop over the permanent types
for ii=1:N_i
    iistr=Names_i{ii};

    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,iistr); % Note: already check for existence of simoptions and created it if it was not inputted
    simoptions_temp.warnzerorestrictedmass=0; % a type with no mass under a restriction is not a warning (the population-level check is below)
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
    [n_z_temp,z_grid_temp,~,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,[],simoptions_temp,3);

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
    whichcombos_ii=max(whichcombosAll(:,:,:,:,ii),whichcombosG);
    fnwanted_ii=any(any(any(whichcombos_ii,2),3),4); % numFnsToEvaluate x 1: the functions this type evaluates (their own stats or any pair, in any age group, on any page)
    % Drop the functions that whichcombos does not want from this type's evaluation
    FnsAndPTypeIndicator_ii=FnsAndPTypeIndicator_ii(:).*fnwanted_ii;
    for ff=1:numFnsToEvaluate
        if fnwanted_ii(ff)==0 && isfield(FnsToEvaluate_temp,FnsToEvalNames{ff})
            FnsToEvaluate_temp=rmfield(FnsToEvaluate_temp,FnsToEvalNames{ff});
        end
    end
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii; % (after the whichcombos reduction)

    % Mass of this type in each age group
    AgeMasses_ii=gather(sum(reshape(StationaryDist_temp,[numel(StationaryDist_temp)/N_j,N_j]),1));
    for kk=1:ngroups
        j1=simoptions.agegroupings(kk);
        if kk<ngroups
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        GroupMasses(ii,kk,1)=sum(AgeMasses_ii(j1:jend));
    end

    if sum(FnsAndPTypeIndicator_ii)==0
        continue % none of the FnsToEvaluate are relevant to this type
    end

    %% Compute for this type
    idx_ii=find(FnsAndPTypeIndicator_ii==1); % the functions this type evaluates, in their order
    simoptions_temp.whichcombos=whichcombos_ii(idx_ii,idx_ii,:,:); % the per-type command applies the selection itself (unselected entries NaN)
    AgeConditionalCrossSectionCorr_ii=EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz(StationaryDist_temp,PolicyIndexes_temp,FnsToEvaluate_temp,Parameters_temp,[],n_d_temp,n_a_temp,n_z_temp,N_j,d_grid_temp,a_grid_temp,z_grid_temp,simoptions_temp);

    % Store by type, page by page
    relevantfns=find(FnsAndPTypeIndicator_ii==1); % the matrices of the type only cover the functions relevant to it, in their order
    for pp=1:nwhichpages
        if pp==1
            Src=AgeConditionalCrossSectionCorr_ii;
        else
            Src=AgeConditionalCrossSectionCorr_ii.(CondlRestnFnNames{pp-1});
            PageOut{pp}.RestrictedSampleMass.(iistr)=Src.RestrictedSampleMass;
            RestrictedMassByAge(ii,:,pp-1)=reshape(gather(Src.RestrictedSampleMass),[1,N_j]);
            for kk=1:ngroups
                j1=simoptions.agegroupings(kk);
                if kk<ngroups
                    jend=simoptions.agegroupings(kk+1)-1;
                else
                    jend=N_j;
                end
                GroupMasses(ii,kk,pp)=sum(RestrictedMassByAge(ii,j1:jend,pp-1));
            end
        end
        for ff=1:numFnsToEvaluate
            if FnsAndPTypeIndicator_ii(ff)==1
                PageOut{pp}.(FnsToEvalNames{ff}).(iistr)=Src.(FnsToEvalNames{ff});
                MeanVec(ff,ii,:,pp)=reshape(gather(Src.(FnsToEvalNames{ff}).Mean),[1,1,ngroups]);
                StdDevVec(ff,ii,:,pp)=reshape(gather(Src.(FnsToEvalNames{ff}).StdDeviation),[1,1,ngroups]);
            end
        end
        PageOut{pp}.CovarianceMatrix_ptype.(iistr)=Src.CovarianceMatrix;
        PageOut{pp}.CorrelationMatrix_ptype.(iistr)=Src.CorrelationMatrix;
        CovarVec(relevantfns,relevantfns,ii,:,pp)=reshape(gather(Src.CovarianceMatrix),[length(relevantfns),length(relevantfns),1,ngroups]);
    end
end

%% Grouped: pool the types, page by page and age group by age group
if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        PageOut{1+rr}.RestrictedSampleMass.ByAge=sum(RestrictedMassByAge(:,:,rr).*ptweights,1); % mass at each age that satisfies the restriction, as a share of the whole population
        PageOut{1+rr}.RestrictedSampleMass.ByPType=sum(RestrictedMassByAge(:,:,rr),2); % conditional on type, what fraction satisfy the restriction
        PageOut{1+rr}.RestrictedSampleMass.Total=sum(ptweights.*sum(RestrictedMassByAge(:,:,rr),2)); % what fraction of the population satisfy the restriction
        if PageOut{1+rr}.RestrictedSampleMass.Total==0
            warning('One of the conditional restrictions evaluates to a zero mass (for every permanent type, at all j)')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
        end
    end
end
if simoptions.groupptypesforstats==1
    if simoptions.verbose==1
        fprintf('Permanent type: Grouped Stats \n')
    end
    for pp=1:nwhichpages
        PageOut{pp}.CovarianceMatrix=nan(numFnsToEvaluate,numFnsToEvaluate,ngroups);
        PageOut{pp}.CorrelationMatrix=nan(numFnsToEvaluate,numFnsToEvaluate,ngroups);
        for ff=1:numFnsToEvaluate
            PageOut{pp}.(FnsToEvalNames{ff}).Mean=nan(1,ngroups);
            PageOut{pp}.(FnsToEvalNames{ff}).StdDeviation=nan(1,ngroups);
            PageOut{pp}.(FnsToEvalNames{ff}).(FnsToEvalNames{ff})=nan(1,ngroups); % one wherever the function is evaluated and has a pool (below)
            for ff2=1:numFnsToEvaluate
                if ff2~=ff
                    PageOut{pp}.(FnsToEvalNames{ff}).CovarianceWith.(FnsToEvalNames{ff2})=nan(1,ngroups);
                    PageOut{pp}.(FnsToEvalNames{ff}).CorrelationWith.(FnsToEvalNames{ff2})=nan(1,ngroups);
                end
            end
        end
        for kk=1:ngroups
            % Grouped Mean and StdDeviation of each function
            MeanG=nan(numFnsToEvaluate,1);
            for ff=1:numFnsToEvaluate
                w=FnsAndPTypeIndicator(ff,:)'.*ptweights.*GroupMasses(:,kk,pp);
                relevant=(w>0); % (a type of zero weight, or with no (restricted) mass in the age group, is not in the pool)
                if any(whichcombosG(ff,:,kk,pp)) && any(relevant) % something grouped of the function is wanted in this age group and it has a pool (otherwise everything stays NaN)
                    p=w(relevant)/sum(w(relevant));
                    MeanG(ff)=sum(p.*MeanVec(ff,relevant,kk,pp)');
                    StdDevG=sqrt(sum(p.*(StdDevVec(ff,relevant,kk,pp)'.^2+(MeanVec(ff,relevant,kk,pp)'-MeanG(ff)).^2)));
                    PageOut{pp}.(FnsToEvalNames{ff}).Mean(kk)=MeanG(ff);
                    PageOut{pp}.(FnsToEvalNames{ff}).StdDeviation(kk)=StdDevG;
                    PageOut{pp}.(FnsToEvalNames{ff}).(FnsToEvalNames{ff})(kk)=1;
                    PageOut{pp}.CovarianceMatrix(ff,ff,kk)=StdDevG^2;
                    PageOut{pp}.CorrelationMatrix(ff,ff,kk)=1;
                end
            end
            % Grouped covariance and correlation of each pair (pooled over the types for which both functions are relevant, about the pooled means of that pool)
            for ff1=1:numFnsToEvaluate
                for ff2=ff1+1:numFnsToEvaluate
                    w=FnsAndPTypeIndicator(ff1,:)'.*FnsAndPTypeIndicator(ff2,:)'.*ptweights.*GroupMasses(:,kk,pp);
                    relevant=(w>0);
                    if whichcombosG(ff1,ff2,kk,pp)==1 && any(relevant) % the grouped pair is wanted in this age group and has a pool
                        p=w(relevant)/sum(w(relevant));
                        mu1=sum(p.*MeanVec(ff1,relevant,kk,pp)');
                        mu2=sum(p.*MeanVec(ff2,relevant,kk,pp)');
                        var1=sum(p.*(StdDevVec(ff1,relevant,kk,pp)'.^2+(MeanVec(ff1,relevant,kk,pp)'-mu1).^2));
                        var2=sum(p.*(StdDevVec(ff2,relevant,kk,pp)'.^2+(MeanVec(ff2,relevant,kk,pp)'-mu2).^2));
                        CoVar=sum(p.*(reshape(CovarVec(ff1,ff2,relevant,kk,pp),[],1)+(MeanVec(ff1,relevant,kk,pp)'-mu1).*(MeanVec(ff2,relevant,kk,pp)'-mu2)));
                        Corr=CoVar/sqrt(var1*var2);
                        PageOut{pp}.(FnsToEvalNames{ff1}).CovarianceWith.(FnsToEvalNames{ff2})(kk)=CoVar;
                        PageOut{pp}.(FnsToEvalNames{ff1}).CorrelationWith.(FnsToEvalNames{ff2})(kk)=Corr;
                        PageOut{pp}.(FnsToEvalNames{ff2}).CovarianceWith.(FnsToEvalNames{ff1})(kk)=CoVar;
                        PageOut{pp}.(FnsToEvalNames{ff2}).CorrelationWith.(FnsToEvalNames{ff1})(kk)=Corr;
                        PageOut{pp}.CovarianceMatrix(ff1,ff2,kk)=CoVar;
                        PageOut{pp}.CovarianceMatrix(ff2,ff1,kk)=CoVar;
                        PageOut{pp}.CorrelationMatrix(ff1,ff2,kk)=Corr;
                        PageOut{pp}.CorrelationMatrix(ff2,ff1,kk)=Corr;
                    end
                end
            end
        end
    end
end

%% Assemble the output
AgeConditionalCrossSectionCorr=PageOut{1};
if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        AgeConditionalCrossSectionCorr.(CondlRestnFnNames{rr})=PageOut{1+rr};
    end
end

AgeConditionalCrossSectionCorr.Notes='Per type: AgeConditionalCrossSectionCorr.(fn).(typename) is the output of EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz for that type, and CovarianceMatrix_ptype.(typename)/CorrelationMatrix_ptype.(typename) its matrices (over the functions relevant to that type; third index is the age group). Grouped (AgeConditionalCrossSectionCorr.(fn).Mean, .CovarianceWith, .CorrelationWith, CovarianceMatrix, CorrelationMatrix): in each age group the types are pooled with weights ptweights*(mass of the type in the age group), so the grouped covariance is the within-type covariance plus the between-type part about the pooled means, and the grouped correlation is that of the pooled population. Under a conditional restriction (AgeConditionalCrossSectionCorr.(restriction)) the same, with the types pooled with weights ptweights*(restricted mass of the type in the age group).';

end
