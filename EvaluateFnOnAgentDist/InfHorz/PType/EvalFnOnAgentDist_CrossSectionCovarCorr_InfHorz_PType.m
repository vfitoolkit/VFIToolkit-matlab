function CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz_PType(StationaryDist, Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, Names_i, d_grid, a_grid, z_grid, simoptions)
% Cross-sectional covariances/correlations between every pair of FnsToEvaluate, with permanent types. Calls
% EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz() for each permanent type, and reports the results by type and (by default)
% grouped over the types. (The InfHorz counterpart of EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz_PType.)
%
% Inputs follow ValueFnIter_InfHorz_PType: anything that depends on the permanent type is given as a structure with a field per
% type (Parameters, n_z, z_grid, simoptions fields, FnsToEvaluate fields) or with one value per type where that form is accepted.
% Names_i is the cell of type names, or just the number of types. StationaryDist is the structure output of
% StationaryDist_InfHorz_PType (it includes StationaryDist.ptweights).
%
% Output:
%   CrossSectionCorr.(fn1).(typename).Mean, .StdDeviation, .CovarianceWith.(fn2), .CorrelationWith.(fn2)   by type
%   CrossSectionCorr.CovarianceMatrix_ptype.(typename), .CorrelationMatrix_ptype.(typename)   by type (over the functions relevant to it)
%   CrossSectionCorr.(fn1).Mean, .StdDeviation, .CovarianceWith.(fn2), .CorrelationWith.(fn2)   grouped
%   CrossSectionCorr.CovarianceMatrix, .CorrelationMatrix   grouped (nFn x nFn)
% With simoptions.conditionalrestrictions (a restriction may be a structure with a field per type), the same per type and grouped
% under each restriction:
%   CrossSectionCorr.(restriction).(fn1).(typename), .CovarianceMatrix_ptype.(typename), .(fn1).Mean, ..., .CovarianceMatrix, ...
%   CrossSectionCorr.(restriction).RestrictedSampleMass.(typename)   the share of the type that satisfies the restriction
%   CrossSectionCorr.(restriction).RestrictedSampleMass.ByPType, .Total   the per-type shares (N_i-by-1), and the share of the population
% Grouped: the types are pooled with weights ptweights (each type's distribution has mass one), so the grouped mean is the weighted
% mean, the grouped variance/covariance is the weighted within-type variance/covariance plus the between-type part (about the pooled
% means), and the grouped correlation is covariance/(std dev * std dev) of the pooled population. A pair of functions is pooled over
% the types for which both are relevant. Under a restriction the weights are ptweights times the type's restricted sample mass, so
% the grouped stats are those of the pooled restricted population; a type with no (restricted) mass is not in the pool.
% simoptions.groupptypesforstats=0 skips the grouped outputs.
%
% simoptions.whichcombos ([numFnsToEvaluate, numFnsToEvaluate]: diagonal = a function's own Mean/StdDeviation, off-diagonal = a pair,
% with a page dimension of 1+number of conditional restrictions when there are restrictions, optionally with a trailing type dimension
% of N_i+1) selects which functions, pairs and restrictions are computed, per type and grouped; see below.

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
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
else
    if ~isfield(simoptions,'groupptypesforstats')
        simoptions.groupptypesforstats=1;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'warnzerorestrictedmass')
        simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=0;
    end
end

useCondlRest=0;
nwhichpages=1; % 'pages': the unrestricted stats, then one per restriction
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

%% simoptions.whichcombos: which functions, pairs and restrictions to compute
% [numFnsToEvaluate, numFnsToEvaluate] of zeros/ones. The diagonal (ff,ff) selects the grouped Mean and StdDeviation of function ff;
% the off-diagonal (ff1,ff2) selects the covariance and correlation of the pair. Only the upper triangle (ff1<=ff2) is read, so a
% symmetric matrix or just its upper triangle can be given. A function with nothing selected (its diagonal and all its pairs zero)
% is not evaluated at all; a selected pair has both its functions evaluated (their means and std devs are needed for the pair).
% Skipped entries are NaN: in the grouped output, and in the per-type pair fields and per-type matrices (the per-type Mean and
% StdDeviation of an evaluated function are reported regardless, the single-type command computes them anyway). Default all ones.
% With conditional restrictions a page dimension follows: [numFnsToEvaluate, numFnsToEvaluate, 1+number of conditional restrictions],
% page 1 the unrestricted stats, pages 2:end the restrictions in the fieldnames order of simoptions.conditionalrestrictions; an input
% without it applies to every page.
% A trailing type dimension may be added: [.., N_i+1] selects per permanent type, in the order of Names_i, with the
% last slot the grouped stats (so [nFns, nFns, 1+number of restrictions, N_i+1] with restrictions, [nFns, nFns, N_i+1] without). A
% type's slot selects that type's own computation; the grouped slot forces every type's computation of that combination (the grouped
% means, std devs and covariances are built from every type's), and a type's output is reported whenever it was computed. The
% grouped output is reported only where the grouped slot asks (the grouped own stats of a function wherever any grouped entry of that
% function's row is on, as they are byproducts of the pairs). An input without the type dimension applies to every ptype and to the
% grouped stats. With restrictions, a three-dimensional input is read as having the page dimension, not the type dimension.
% Intended for calibration/estimation. (As EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz_PType.)
if ~isfield(simoptions,'whichcombos')
    whichcombosAll=ones(numFnsToEvaluate,numFnsToEvaluate,nwhichpages,N_i+1); % whichcombos here is [nFns, nFns, 1+nRestr, N_i+1]
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ismatrix(whichcombos) && isequal(size(whichcombos),[numFnsToEvaluate,numFnsToEvaluate])
        whichcombos=repmat(whichcombos,[1,1,nwhichpages]); % one matrix: apply to every page
    end
    if nwhichpages==1 && ndims(whichcombos)==3 && isequal(size(whichcombos),[numFnsToEvaluate,numFnsToEvaluate,N_i+1])
        whichcombos=reshape(whichcombos,[numFnsToEvaluate,numFnsToEvaluate,1,N_i+1]); % no restrictions: the third dimension is the type dimension
    elseif ndims(whichcombos)<=3 && isequal(size(whichcombos,1:3),[numFnsToEvaluate,numFnsToEvaluate,nwhichpages])
        whichcombos=repmat(whichcombos,[1,1,1,N_i+1]); % no type dimension: apply to every ptype and to the grouped stats
    end
    if ~isequal(size(whichcombos,1:4),[numFnsToEvaluate,numFnsToEvaluate,nwhichpages,N_i+1]) || ndims(whichcombos)>4
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(numFnsToEvaluate),',',num2str(nwhichpages),'] (number of FnsToEvaluate, twice, 1+number of conditional restrictions; the third dimension is dropped when there are no conditional restrictions), optionally with a trailing type dimension of ',num2str(N_i+1),' (the permanent types in the order of Names_i, then the grouped stats)'])
    end
    whichcombosAll=double(whichcombos);
    for ss=1:N_i+1
        for pp=1:nwhichpages
            whichcombosAll(:,:,pp,ss)=max(triu(whichcombosAll(:,:,pp,ss)),triu(whichcombosAll(:,:,pp,ss))'); % symmetric, from the upper triangle
        end
    end
end
whichcombosG=whichcombosAll(:,:,:,N_i+1); % the grouped stats; whichcombosAll(:,:,:,ii) is ptype ii

ptweights=gather(reshape(StationaryDist.ptweights,[N_i,1]));

% Each page is filled as its own output structure; page 1 (unrestricted) becomes CrossSectionCorr and pages 2:end CrossSectionCorr.(restrictionname)
PageOut=cell(nwhichpages,1);
for pp=1:nwhichpages
    PageOut{pp}=struct();
end
EvalInd=zeros(numFnsToEvaluate,N_i,nwhichpages); % the type reports the function on the page
MeanVec=nan(numFnsToEvaluate,N_i,nwhichpages);
StdDevVec=nan(numFnsToEvaluate,N_i,nwhichpages);
CovarVec=nan(numFnsToEvaluate,numFnsToEvaluate,N_i,nwhichpages);
MassFactor=ones(N_i,nwhichpages); % the weight of a type in the pool of a page, relative to ptweights: one for the unrestricted page, the restricted sample mass of the type for a restriction
if useCondlRest==1
    MassFactor(:,2:end)=NaN; % (filled from the per-type outputs)
end

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
    whichcombos_ii=max(whichcombosAll(:,:,:,ii),whichcombosG);
    fnwanted_ii=any(any(whichcombos_ii,2),3); % numFnsToEvaluate x 1: the functions this type evaluates (their own stats or any pair, on any page)
    % Drop the functions that whichcombos does not want from this type's evaluation
    FnsAndPTypeIndicator_ii=FnsAndPTypeIndicator_ii(:).*fnwanted_ii;
    for ff=1:numFnsToEvaluate
        if fnwanted_ii(ff)==0 && isfield(FnsToEvaluate_temp,FnsToEvalNames{ff})
            FnsToEvaluate_temp=rmfield(FnsToEvaluate_temp,FnsToEvalNames{ff});
        end
    end
    if sum(FnsAndPTypeIndicator_ii)==0
        continue % none of the FnsToEvaluate are relevant to this type (or wanted for it)
    end

    %% Compute for this type
    idx_ii=find(FnsAndPTypeIndicator_ii==1); % the functions this type evaluates, in their order
    simoptions_temp.whichcombos=whichcombos_ii(idx_ii,idx_ii,:); % the per-type command applies the selection itself (unselected pairs NaN, a function with nothing selected on a page absent from it)
    CrossSectionCorr_ii=EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz(StationaryDist_temp,PolicyIndexes_temp,FnsToEvaluate_temp,Parameters_temp,[],n_d_temp,n_a_temp,n_z_temp,d_grid_temp,a_grid_temp,z_grid_temp,simoptions_temp);

    % Store by type, page by page
    relevantfns=find(FnsAndPTypeIndicator_ii==1); % the matrices of the type only cover the functions relevant to it, in their order
    for pp=1:nwhichpages
        if pp==1
            Src=CrossSectionCorr_ii;
        else
            Src=CrossSectionCorr_ii.(CondlRestnFnNames{pp-1});
            MassFactor(ii,pp)=gather(Src.RestrictedSampleMass);
            PageOut{pp}.RestrictedSampleMass.(iistr)=Src.RestrictedSampleMass;
        end
        for ff=1:numFnsToEvaluate
            if FnsAndPTypeIndicator_ii(ff)==1 && isfield(Src,FnsToEvalNames{ff})
                PageOut{pp}.(FnsToEvalNames{ff}).(iistr)=Src.(FnsToEvalNames{ff});
                EvalInd(ff,ii,pp)=1;
                MeanVec(ff,ii,pp)=gather(Src.(FnsToEvalNames{ff}).Mean);
                StdDevVec(ff,ii,pp)=gather(Src.(FnsToEvalNames{ff}).StdDeviation);
            end
        end
        PageOut{pp}.CovarianceMatrix_ptype.(iistr)=Src.CovarianceMatrix;
        PageOut{pp}.CorrelationMatrix_ptype.(iistr)=Src.CorrelationMatrix;
        CovarVec(relevantfns,relevantfns,ii,pp)=gather(Src.CovarianceMatrix);
    end
end

%% Grouped: pool the types, page by page
if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        PageOut{1+rr}.RestrictedSampleMass.ByPType=MassFactor(:,1+rr); % conditional on type, what fraction satisfy the restriction
        PageOut{1+rr}.RestrictedSampleMass.Total=sum(ptweights.*MassFactor(:,1+rr)); % what fraction of the population satisfy the restriction
        if PageOut{1+rr}.RestrictedSampleMass.Total==0 && simoptions.warnzerorestrictedmass>=1
            warning('One of the conditional restrictions evaluates to a zero mass (for every permanent type)')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
        end
    end
end
if simoptions.groupptypesforstats==1
    if simoptions.verbose==1
        fprintf('Permanent type: Grouped Stats \n')
    end
    for pp=1:nwhichpages
        PageOut{pp}.CovarianceMatrix=nan(numFnsToEvaluate,numFnsToEvaluate);
        PageOut{pp}.CorrelationMatrix=nan(numFnsToEvaluate,numFnsToEvaluate);

        % Grouped Mean and StdDeviation of each function
        MeanG=nan(numFnsToEvaluate,1);
        StdDevG=nan(numFnsToEvaluate,1);
        for ff=1:numFnsToEvaluate
            w=any(whichcombosG(ff,:,pp))*EvalInd(ff,:,pp)'.*ptweights.*MassFactor(:,pp); % (zero when no grouped entry of this function's row is on: nothing grouped is reported for it)
            relevant=(w>0); % (a type of zero weight, or with no restricted mass, is not in the pool; NaN*0 would otherwise poison the sums)
            if any(relevant)
                p=w(relevant)/sum(w(relevant));
                MeanG(ff)=sum(p.*MeanVec(ff,relevant,pp)');
                StdDevG(ff)=sqrt(sum(p.*(StdDevVec(ff,relevant,pp)'.^2+(MeanVec(ff,relevant,pp)'-MeanG(ff)).^2)));
            end
            PageOut{pp}.(FnsToEvalNames{ff}).Mean=MeanG(ff);
            PageOut{pp}.(FnsToEvalNames{ff}).StdDeviation=StdDevG(ff);
            PageOut{pp}.CovarianceMatrix(ff,ff)=StdDevG(ff)^2;
            if any(relevant) % evaluated and with a pool (otherwise NaN, the self-correlation included)
                PageOut{pp}.CorrelationMatrix(ff,ff)=1;
                PageOut{pp}.(FnsToEvalNames{ff}).(FnsToEvalNames{ff})=1;
            else
                PageOut{pp}.(FnsToEvalNames{ff}).(FnsToEvalNames{ff})=NaN; % nothing grouped of this function is wanted, or nobody in the pool
            end
        end

        % Grouped covariance and correlation of each pair (pooled over the types for which both functions are relevant, about the pooled means of that pool)
        for ff1=1:numFnsToEvaluate
            for ff2=ff1+1:numFnsToEvaluate
                w=EvalInd(ff1,:,pp)'.*EvalInd(ff2,:,pp)'.*ptweights.*MassFactor(:,pp);
                relevant=(w>0);
                CoVar=NaN;
                Corr=NaN;
                if whichcombosG(ff1,ff2,pp)==1 && any(relevant) % the grouped pair is wanted and has a pool
                    p=w(relevant)/sum(w(relevant));
                    mu1=sum(p.*MeanVec(ff1,relevant,pp)');
                    mu2=sum(p.*MeanVec(ff2,relevant,pp)');
                    var1=sum(p.*(StdDevVec(ff1,relevant,pp)'.^2+(MeanVec(ff1,relevant,pp)'-mu1).^2));
                    var2=sum(p.*(StdDevVec(ff2,relevant,pp)'.^2+(MeanVec(ff2,relevant,pp)'-mu2).^2));
                    CoVar=sum(p.*(reshape(CovarVec(ff1,ff2,relevant,pp),[],1)+(MeanVec(ff1,relevant,pp)'-mu1).*(MeanVec(ff2,relevant,pp)'-mu2)));
                    Corr=CoVar/sqrt(var1*var2);
                end
                PageOut{pp}.(FnsToEvalNames{ff1}).CovarianceWith.(FnsToEvalNames{ff2})=CoVar;
                PageOut{pp}.(FnsToEvalNames{ff1}).CorrelationWith.(FnsToEvalNames{ff2})=Corr;
                PageOut{pp}.(FnsToEvalNames{ff2}).CovarianceWith.(FnsToEvalNames{ff1})=CoVar;
                PageOut{pp}.(FnsToEvalNames{ff2}).CorrelationWith.(FnsToEvalNames{ff1})=Corr;
                PageOut{pp}.CovarianceMatrix(ff1,ff2)=CoVar;
                PageOut{pp}.CovarianceMatrix(ff2,ff1)=CoVar;
                PageOut{pp}.CorrelationMatrix(ff1,ff2)=Corr;
                PageOut{pp}.CorrelationMatrix(ff2,ff1)=Corr;
            end
        end
    end
end

%% Assemble the output
CrossSectionCorr=PageOut{1};
if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        CrossSectionCorr.(CondlRestnFnNames{rr})=PageOut{1+rr};
    end
end

CrossSectionCorr.Notes='Per type: CrossSectionCorr.(fn).(typename) is the output of EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz for that type, and CovarianceMatrix_ptype.(typename)/CorrelationMatrix_ptype.(typename) its matrices (over the functions relevant to that type). Grouped (CrossSectionCorr.(fn).Mean, .CovarianceWith, .CorrelationWith, CovarianceMatrix, CorrelationMatrix): the types are pooled with weights ptweights, so the grouped covariance is the within-type covariance plus the between-type part about the pooled means, and the grouped correlation is that of the pooled population. Under a conditional restriction (CrossSectionCorr.(restriction)) the same, with the types pooled with weights ptweights times their restricted sample mass.';

end
