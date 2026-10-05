function AgeConditionalCrossSectionCorr=EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz_PType(StationaryDist, Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid, z_grid, simoptions)
% simoptions.whichcombos ([numFnsToEvaluate, numFnsToEvaluate, number of age groups]: diagonal = a function's own Mean/StdDeviation,
% off-diagonal = a pair, third dimension = age group) selects which functions, pairs and age groups are computed; see below.
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
% gives NaN. simoptions.groupptypesforstats=0 skips the grouped outputs.

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

if isstruct(FnsToEvaluate)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnsToEvalNames);
else
    error('You can only use PType when FnsToEvaluate is a structure')
end

%% simoptions.whichcombos: which functions, pairs and age groups to compute
% [numFnsToEvaluate, numFnsToEvaluate, ngroups] of zeros/ones. In each age group the diagonal (ff,ff) selects the grouped Mean and
% StdDeviation of function ff and the off-diagonal (ff1,ff2) selects the covariance and correlation of the pair; only the upper
% triangle (ff1<=ff2) is read, so a symmetric matrix or just its upper triangle can be given. A [numFnsToEvaluate, numFnsToEvaluate]
% input is applied to every age group. A function with nothing selected (its diagonal and all its pairs zero in every age group) is
% not evaluated at all; a selected pair has both its functions evaluated. Skipped entries are NaN: in the grouped output, and in the
% per-type pair fields and per-type matrices (the per-type Mean and StdDeviation of an evaluated function are reported regardless,
% the single-type command computes them anyway). Default all ones.
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,numFnsToEvaluate,ngroups);
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ismatrix(whichcombos) && isequal(size(whichcombos),[numFnsToEvaluate,numFnsToEvaluate])
        whichcombos=repmat(whichcombos,[1,1,ngroups]); % one matrix: apply to every age group
    end
    if ~isequal(size(whichcombos,1:3),[numFnsToEvaluate,numFnsToEvaluate,ngroups])
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(numFnsToEvaluate),',',num2str(ngroups),'] (number of FnsToEvaluate, twice, number of age groups)'])
    end
    whichcombos=double(whichcombos);
    for kk=1:ngroups
        whichcombos(:,:,kk)=max(triu(whichcombos(:,:,kk)),triu(whichcombos(:,:,kk))'); % symmetric, from the upper triangle
    end
end
fnwanted=any(any(whichcombos,2),3); % numFnsToEvaluate x 1: the functions that get evaluated (their own stats or any pair, in any age group)

ptweights=gather(reshape(StationaryDist.ptweights,[N_i,1]));

AgeConditionalCrossSectionCorr=struct();
FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i);
MeanVec=nan(numFnsToEvaluate,N_i,ngroups);
StdDevVec=nan(numFnsToEvaluate,N_i,ngroups);
CovarVec=nan(numFnsToEvaluate,numFnsToEvaluate,N_i,ngroups);
GroupMasses=zeros(N_i,ngroups); % mass of each type in each age group (from that type's age weights)

%% Loop over the permanent types
for ii=1:N_i
    iistr=Names_i{ii};

    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,iistr); % Note: already check for existence of simoptions and created it if it was not inputted

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
    % Drop the functions that whichcombos does not want from this type's evaluation
    FnsAndPTypeIndicator_ii=FnsAndPTypeIndicator_ii(:).*fnwanted;
    for ff=1:numFnsToEvaluate
        if fnwanted(ff)==0 && isfield(FnsToEvaluate_temp,FnsToEvalNames{ff})
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
        GroupMasses(ii,kk)=sum(AgeMasses_ii(j1:jend));
    end

    if sum(FnsAndPTypeIndicator_ii)==0
        continue % none of the FnsToEvaluate are relevant to this type
    end

    %% Compute for this type
    AgeConditionalCrossSectionCorr_ii=EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz(StationaryDist_temp,PolicyIndexes_temp,FnsToEvaluate_temp,Parameters_temp,[],n_d_temp,n_a_temp,n_z_temp,N_j,d_grid_temp,a_grid_temp,z_grid_temp,simoptions_temp);

    % Store by type
    for ff=1:numFnsToEvaluate
        if FnsAndPTypeIndicator_ii(ff)==1
            AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff}).(iistr)=AgeConditionalCrossSectionCorr_ii.(FnsToEvalNames{ff});
            MeanVec(ff,ii,:)=reshape(gather(AgeConditionalCrossSectionCorr_ii.(FnsToEvalNames{ff}).Mean),[1,1,ngroups]);
            StdDevVec(ff,ii,:)=reshape(gather(AgeConditionalCrossSectionCorr_ii.(FnsToEvalNames{ff}).StdDeviation),[1,1,ngroups]);
        end
    end
    AgeConditionalCrossSectionCorr.CovarianceMatrix_ptype.(iistr)=AgeConditionalCrossSectionCorr_ii.CovarianceMatrix;
    AgeConditionalCrossSectionCorr.CorrelationMatrix_ptype.(iistr)=AgeConditionalCrossSectionCorr_ii.CorrelationMatrix;
    % The matrices of the type only cover the functions relevant to it, in their order
    relevantfns=find(FnsAndPTypeIndicator_ii==1);
    CovarVec(relevantfns,relevantfns,ii,:)=reshape(gather(AgeConditionalCrossSectionCorr_ii.CovarianceMatrix),[length(relevantfns),length(relevantfns),1,ngroups]);
    % Per-type output: the (pair, age group) entries that whichcombos does not select are NaN (the single-type command computed every pair of the functions it was given)
    for ff1=1:numFnsToEvaluate
        for ff2=1:numFnsToEvaluate
            if ff1~=ff2 && any(whichcombos(ff1,ff2,:)==0) && FnsAndPTypeIndicator_ii(ff1)==1 && FnsAndPTypeIndicator_ii(ff2)==1
                if isfield(AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff1}).(iistr),'CovarianceWith') && isfield(AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff1}).(iistr).CovarianceWith,FnsToEvalNames{ff2})
                    offkk=reshape(whichcombos(ff1,ff2,:)==0,[1,ngroups]);
                    temp=AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff1}).(iistr).CovarianceWith.(FnsToEvalNames{ff2}); temp(offkk)=NaN; AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff1}).(iistr).CovarianceWith.(FnsToEvalNames{ff2})=temp;
                    temp=AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff1}).(iistr).CorrelationWith.(FnsToEvalNames{ff2}); temp(offkk)=NaN; AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff1}).(iistr).CorrelationWith.(FnsToEvalNames{ff2})=temp;
                end
            end
        end
    end
    masksub=whichcombos(relevantfns,relevantfns,:);
    for kk=1:ngroups
        masksub(:,:,kk)=max(masksub(:,:,kk),eye(length(relevantfns))); % the per-type matrix diagonals (variances) are reported regardless
    end
    CovMat_ii=AgeConditionalCrossSectionCorr.CovarianceMatrix_ptype.(iistr); CovMat_ii(masksub==0)=NaN; AgeConditionalCrossSectionCorr.CovarianceMatrix_ptype.(iistr)=CovMat_ii;
    CorrMat_ii=AgeConditionalCrossSectionCorr.CorrelationMatrix_ptype.(iistr); CorrMat_ii(masksub==0)=NaN; AgeConditionalCrossSectionCorr.CorrelationMatrix_ptype.(iistr)=CorrMat_ii;
end

%% Grouped: pool the types, age group by age group
if simoptions.groupptypesforstats==1
    if simoptions.verbose==1
        fprintf('Permanent type: Grouped Stats \n')
    end

    AgeConditionalCrossSectionCorr.CovarianceMatrix=nan(numFnsToEvaluate,numFnsToEvaluate,ngroups);
    AgeConditionalCrossSectionCorr.CorrelationMatrix=nan(numFnsToEvaluate,numFnsToEvaluate,ngroups);
    for ff=1:numFnsToEvaluate
        AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff}).Mean=nan(1,ngroups);
        AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff}).StdDeviation=nan(1,ngroups);
        AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff}).(FnsToEvalNames{ff})=ones(1,ngroups);
        if fnwanted(ff)==0 % a function whichcombos does not evaluate: its self-correlation is NaN too (the age-group loop below never reaches it)
            AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff}).(FnsToEvalNames{ff})=nan(1,ngroups);
        end
        for ff2=1:numFnsToEvaluate
            if ff2~=ff
                AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff}).CovarianceWith.(FnsToEvalNames{ff2})=nan(1,ngroups);
                AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff}).CorrelationWith.(FnsToEvalNames{ff2})=nan(1,ngroups);
            end
        end
    end

    for kk=1:ngroups
        % Grouped Mean and StdDeviation of each function
        MeanG=nan(numFnsToEvaluate,1);
        for ff=1:numFnsToEvaluate
            w=FnsAndPTypeIndicator(ff,:)'.*ptweights.*GroupMasses(:,kk);
            if sum(w)>0
                p=w/sum(w);
                relevant=(w>0);
                MeanG(ff)=sum(p(relevant).*MeanVec(ff,relevant,kk)');
                StdDevG=sqrt(sum(p(relevant).*(StdDevVec(ff,relevant,kk)'.^2+(MeanVec(ff,relevant,kk)'-MeanG(ff)).^2)));
                if whichcombos(ff,ff,kk)==1 % the grouped Mean/StdDeviation of this function in this age group are wanted (MeanG/StdDevG are computed regardless as the pairs need them)
                AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff}).Mean(kk)=MeanG(ff);
                AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff}).StdDeviation(kk)=StdDevG;
                AgeConditionalCrossSectionCorr.CovarianceMatrix(ff,ff,kk)=StdDevG^2;
                AgeConditionalCrossSectionCorr.CorrelationMatrix(ff,ff,kk)=1;
                else
                    AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff}).(FnsToEvalNames{ff})(kk)=NaN;
                end
            end
        end

        % Grouped covariance and correlation of each pair (pooled over the types for which both functions are relevant, about the pooled means of that pool)
        for ff1=1:numFnsToEvaluate
            for ff2=ff1+1:numFnsToEvaluate
                w=FnsAndPTypeIndicator(ff1,:)'.*FnsAndPTypeIndicator(ff2,:)'.*ptweights.*GroupMasses(:,kk);
                if whichcombos(ff1,ff2,kk)==1 && sum(w)>0 % the pair is wanted in this age group and has a pool
                    p=w/sum(w);
                    relevant=(w>0);
                    mu1=sum(p(relevant).*MeanVec(ff1,relevant,kk)');
                    mu2=sum(p(relevant).*MeanVec(ff2,relevant,kk)');
                    var1=sum(p(relevant).*(StdDevVec(ff1,relevant,kk)'.^2+(MeanVec(ff1,relevant,kk)'-mu1).^2));
                    var2=sum(p(relevant).*(StdDevVec(ff2,relevant,kk)'.^2+(MeanVec(ff2,relevant,kk)'-mu2).^2));
                    CoVar=sum(p(relevant).*(reshape(CovarVec(ff1,ff2,relevant,kk),[],1)+(MeanVec(ff1,relevant,kk)'-mu1).*(MeanVec(ff2,relevant,kk)'-mu2)));
                    Corr=CoVar/sqrt(var1*var2);
                    AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff1}).CovarianceWith.(FnsToEvalNames{ff2})(kk)=CoVar;
                    AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff1}).CorrelationWith.(FnsToEvalNames{ff2})(kk)=Corr;
                    AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff2}).CovarianceWith.(FnsToEvalNames{ff1})(kk)=CoVar;
                    AgeConditionalCrossSectionCorr.(FnsToEvalNames{ff2}).CorrelationWith.(FnsToEvalNames{ff1})(kk)=Corr;
                    AgeConditionalCrossSectionCorr.CovarianceMatrix(ff1,ff2,kk)=CoVar;
                    AgeConditionalCrossSectionCorr.CovarianceMatrix(ff2,ff1,kk)=CoVar;
                    AgeConditionalCrossSectionCorr.CorrelationMatrix(ff1,ff2,kk)=Corr;
                    AgeConditionalCrossSectionCorr.CorrelationMatrix(ff2,ff1,kk)=Corr;
                end
            end
        end
    end
end

AgeConditionalCrossSectionCorr.Notes='Per type: AgeConditionalCrossSectionCorr.(fn).(typename) is the output of EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz for that type, and CovarianceMatrix_ptype.(typename)/CorrelationMatrix_ptype.(typename) its matrices (over the functions relevant to that type; third index is the age group). Grouped (AgeConditionalCrossSectionCorr.(fn).Mean, .CovarianceWith, .CorrelationWith, CovarianceMatrix, CorrelationMatrix): in each age group the types are pooled with weights ptweights*(mass of the type in the age group), so the grouped covariance is the within-type covariance plus the between-type part about the pooled means, and the grouped correlation is that of the pooled population.';

end
