function CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz_PType(StationaryDist, Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid, z_grid, simoptions)
% simoptions.whichcombos ([numFnsToEvaluate, numFnsToEvaluate]: diagonal = a function's own Mean/StdDeviation, off-diagonal = a pair)
% selects which functions and pairs are computed; see below.
% Cross-sectional covariances/correlations between every pair of FnsToEvaluate, with
% permanent types. Calls EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz() for each permanent
% type, and reports the results by type and (by default) grouped over the types.
%
% Inputs follow LifeCycleProfiles_FHorz_Case1_PType: anything that depends on the
% permanent type is given as a structure with a field per type (Parameters, z_grid,
% simoptions fields, FnsToEvaluate fields) or with a trailing dimension of length N_i where
% that form is accepted. Names_i is the cell of type names, or just the number of types.
% StationaryDist is the structure output of StationaryDist_Case1_FHorz_PType (it includes
% StationaryDist.ptweights). N_j may be a structure (per type).
%
% Output:
%   CrossSectionCorr.(fn1).(typename).Mean, .StdDeviation, .CovarianceWith.(fn2), .CorrelationWith.(fn2)   by type
%   CrossSectionCorr.CovarianceMatrix_ptype.(typename), .CorrelationMatrix_ptype.(typename)   by type (nFn x nFn)
%   CrossSectionCorr.(fn1).Mean, .StdDeviation, .CovarianceWith.(fn2), .CorrelationWith.(fn2)   grouped
%   CrossSectionCorr.CovarianceMatrix, .CorrelationMatrix   grouped (nFn x nFn)
% Grouped: the types are pooled with weights ptweights (each type's distribution has mass
% one), so the grouped mean is the weighted mean, the grouped variance/covariance is the
% weighted within-type variance/covariance plus the between-type part (about the pooled
% means), and the grouped correlation is covariance/(std dev * std dev) of the pooled
% population. A pair of functions is pooled over the types for which both are relevant.
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
end

if isstruct(FnsToEvaluate)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnsToEvalNames);
else
    error('You can only use PType when FnsToEvaluate is a structure')
end

%% simoptions.whichcombos: which functions and pairs to compute
% [numFnsToEvaluate, numFnsToEvaluate] of zeros/ones. The diagonal (ff,ff) selects the grouped Mean and StdDeviation of function ff;
% the off-diagonal (ff1,ff2) selects the covariance and correlation of the pair. Only the upper triangle (ff1<=ff2) is read, so a
% symmetric matrix or just its upper triangle can be given. A function with nothing selected (its diagonal and all its pairs zero)
% is not evaluated at all; a selected pair has both its functions evaluated (their means and std devs are needed for the pair).
% Skipped entries are NaN: in the grouped output, and in the per-type pair fields and per-type matrices (the per-type Mean and
% StdDeviation of an evaluated function are reported regardless, the single-type command computes them anyway). Default all ones.
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,numFnsToEvaluate);
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ~isequal(size(whichcombos),[numFnsToEvaluate,numFnsToEvaluate])
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(numFnsToEvaluate),'] (number of FnsToEvaluate, twice)'])
    end
    whichcombos=double(triu(whichcombos));
    whichcombos=max(whichcombos,whichcombos'); % symmetric, from the upper triangle
end
fnwanted=any(whichcombos,2); % numFnsToEvaluate x 1: the functions that get evaluated (their own stats or any pair)

ptweights=gather(reshape(StationaryDist.ptweights,[N_i,1]));

CrossSectionCorr=struct();
FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i);
MeanVec=nan(numFnsToEvaluate,N_i);
StdDevVec=nan(numFnsToEvaluate,N_i);
CovarVec=nan(numFnsToEvaluate,numFnsToEvaluate,N_i);

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
    if isstruct(N_j)
        N_j_temp=N_j.(iistr);
    else
        N_j_temp=N_j;
    end

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

    if sum(FnsAndPTypeIndicator_ii)==0
        continue % none of the FnsToEvaluate are relevant to this type
    end

    %% Compute for this type
    idx_ii=find(FnsAndPTypeIndicator_ii==1); % the functions this type evaluates, in their order
    simoptions_temp.whichcombos=whichcombos(idx_ii,idx_ii); % the per-type command applies the selection itself (the NaN-ing below then finds nothing left to do)
    CrossSectionCorr_ii=EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz(StationaryDist_temp,PolicyIndexes_temp,FnsToEvaluate_temp,Parameters_temp,[],n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,z_grid_temp,simoptions_temp);

    % Store by type
    for ff=1:numFnsToEvaluate
        if FnsAndPTypeIndicator_ii(ff)==1
            CrossSectionCorr.(FnsToEvalNames{ff}).(iistr)=CrossSectionCorr_ii.(FnsToEvalNames{ff});
            MeanVec(ff,ii)=gather(CrossSectionCorr_ii.(FnsToEvalNames{ff}).Mean);
            StdDevVec(ff,ii)=gather(CrossSectionCorr_ii.(FnsToEvalNames{ff}).StdDeviation);
        end
    end
    CrossSectionCorr.CovarianceMatrix_ptype.(iistr)=CrossSectionCorr_ii.CovarianceMatrix;
    CrossSectionCorr.CorrelationMatrix_ptype.(iistr)=CrossSectionCorr_ii.CorrelationMatrix;
    % The matrices of the type only cover the functions relevant to it, in their order
    relevantfns=find(FnsAndPTypeIndicator_ii==1);
    CovarVec(relevantfns,relevantfns,ii)=gather(CrossSectionCorr_ii.CovarianceMatrix);
    % Per-type output: the pairs that whichcombos does not select are NaN (the single-type command computed every pair of the functions it was given)
    for ff1=1:numFnsToEvaluate
        for ff2=1:numFnsToEvaluate
            if ff1~=ff2 && whichcombos(ff1,ff2)==0 && FnsAndPTypeIndicator_ii(ff1)==1 && FnsAndPTypeIndicator_ii(ff2)==1
                if isfield(CrossSectionCorr.(FnsToEvalNames{ff1}).(iistr),'CovarianceWith') && isfield(CrossSectionCorr.(FnsToEvalNames{ff1}).(iistr).CovarianceWith,FnsToEvalNames{ff2})
                    CrossSectionCorr.(FnsToEvalNames{ff1}).(iistr).CovarianceWith.(FnsToEvalNames{ff2})=NaN;
                    CrossSectionCorr.(FnsToEvalNames{ff1}).(iistr).CorrelationWith.(FnsToEvalNames{ff2})=NaN;
                end
            end
        end
    end
    masksub=whichcombos(relevantfns,relevantfns);
    masksub(logical(eye(length(relevantfns))))=1; % the per-type matrix diagonals (variances) are reported regardless
    CovMat_ii=CrossSectionCorr.CovarianceMatrix_ptype.(iistr); CovMat_ii(masksub==0)=NaN; CrossSectionCorr.CovarianceMatrix_ptype.(iistr)=CovMat_ii;
    CorrMat_ii=CrossSectionCorr.CorrelationMatrix_ptype.(iistr); CorrMat_ii(masksub==0)=NaN; CrossSectionCorr.CorrelationMatrix_ptype.(iistr)=CorrMat_ii;
end

%% Grouped: pool the types
if simoptions.groupptypesforstats==1
    if simoptions.verbose==1
        fprintf('Permanent type: Grouped Stats \n')
    end

    CrossSectionCorr.CovarianceMatrix=nan(numFnsToEvaluate,numFnsToEvaluate);
    CrossSectionCorr.CorrelationMatrix=nan(numFnsToEvaluate,numFnsToEvaluate);

    % Grouped Mean and StdDeviation of each function
    MeanG=nan(numFnsToEvaluate,1);
    StdDevG=nan(numFnsToEvaluate,1);
    for ff=1:numFnsToEvaluate
        w=FnsAndPTypeIndicator(ff,:)'.*ptweights;
        if sum(w)>0
            p=w/sum(w);
            relevant=(w>0);
            MeanG(ff)=sum(p(relevant).*MeanVec(ff,relevant)');
            StdDevG(ff)=sqrt(sum(p(relevant).*(StdDevVec(ff,relevant)'.^2+(MeanVec(ff,relevant)'-MeanG(ff)).^2)));
        end
        CrossSectionCorr.(FnsToEvalNames{ff}).Mean=MeanG(ff);
        CrossSectionCorr.(FnsToEvalNames{ff}).StdDeviation=StdDevG(ff);
        CrossSectionCorr.CovarianceMatrix(ff,ff)=StdDevG(ff)^2;
        if sum(w)>0 % evaluated (an unevaluated function keeps NaN on the diagonal)
            CrossSectionCorr.CorrelationMatrix(ff,ff)=1;
        end
        if fnwanted(ff)==1
            CrossSectionCorr.(FnsToEvalNames{ff}).(FnsToEvalNames{ff})=1;
        else
            CrossSectionCorr.(FnsToEvalNames{ff}).(FnsToEvalNames{ff})=NaN; % not evaluated
        end
    end

    % Grouped covariance and correlation of each pair (pooled over the types for which both functions are relevant, about the pooled means of that pool)
    for ff1=1:numFnsToEvaluate
        for ff2=ff1+1:numFnsToEvaluate
            w=FnsAndPTypeIndicator(ff1,:)'.*FnsAndPTypeIndicator(ff2,:)'.*ptweights;
            CoVar=NaN;
            Corr=NaN;
            if whichcombos(ff1,ff2)==1 && sum(w)>0 % the pair is wanted and has a pool
                p=w/sum(w);
                relevant=(w>0);
                mu1=sum(p(relevant).*MeanVec(ff1,relevant)');
                mu2=sum(p(relevant).*MeanVec(ff2,relevant)');
                var1=sum(p(relevant).*(StdDevVec(ff1,relevant)'.^2+(MeanVec(ff1,relevant)'-mu1).^2));
                var2=sum(p(relevant).*(StdDevVec(ff2,relevant)'.^2+(MeanVec(ff2,relevant)'-mu2).^2));
                CoVar=sum(p(relevant).*(squeeze(CovarVec(ff1,ff2,relevant))+(MeanVec(ff1,relevant)'-mu1).*(MeanVec(ff2,relevant)'-mu2)));
                Corr=CoVar/sqrt(var1*var2);
            end
            CrossSectionCorr.(FnsToEvalNames{ff1}).CovarianceWith.(FnsToEvalNames{ff2})=CoVar;
            CrossSectionCorr.(FnsToEvalNames{ff1}).CorrelationWith.(FnsToEvalNames{ff2})=Corr;
            CrossSectionCorr.(FnsToEvalNames{ff2}).CovarianceWith.(FnsToEvalNames{ff1})=CoVar;
            CrossSectionCorr.(FnsToEvalNames{ff2}).CorrelationWith.(FnsToEvalNames{ff1})=Corr;
            CrossSectionCorr.CovarianceMatrix(ff1,ff2)=CoVar;
            CrossSectionCorr.CovarianceMatrix(ff2,ff1)=CoVar;
            CrossSectionCorr.CorrelationMatrix(ff1,ff2)=Corr;
            CrossSectionCorr.CorrelationMatrix(ff2,ff1)=Corr;
        end
    end
end

CrossSectionCorr.Notes='Per type: CrossSectionCorr.(fn).(typename) is the output of EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz for that type, and CovarianceMatrix_ptype.(typename)/CorrelationMatrix_ptype.(typename) its matrices (over the functions relevant to that type). Grouped (CrossSectionCorr.(fn).Mean, .CovarianceWith, .CorrelationWith, CovarianceMatrix, CorrelationMatrix): the types are pooled with weights ptweights, so the grouped covariance is the within-type covariance plus the between-type part about the pooled means, and the grouped correlation is that of the pooled population.';

end
