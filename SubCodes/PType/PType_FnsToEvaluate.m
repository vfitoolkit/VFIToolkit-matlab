function [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate, FnsToEvaluateParamNames,Names_i,ii,l_d,l_a,l_z,FnsToEvaluate_StructToCell)
% Figure out which functions are actually relevant to the present PType. 
% Only the relevant ones need to be evaluated.
% The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are necessarily the same.

FnsToEvaluate_temp={};

if ~exist('FnsToEvaluate_StructToCell','var')
    FnsToEvaluate_StructToCell=0; % Keep structure as structure by default
end

if isstruct(FnsToEvaluate) && FnsToEvaluate_StructToCell==0 % Structure
    % Just conver from struct into the FnsToEvaluate_temp and FnsToEvaluateParamNames_temp format now.
    FnNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnNames);
    WhichFnsForCurrentPType=zeros(numFnsToEvaluate,1);
    FnsToEvaluateParamNames_temp=[]; % Ignore, is filled in by subcodes
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for ff=1:numFnsToEvaluate
        % Note: I keep it as a structure to avoid having to find the input names here (which needs to allow for things like using n_e)
        if isa(FnsToEvaluate.(FnNames{ff}),'struct')
            if isfield(FnsToEvaluate.(FnNames{ff}), Names_i{ii})
                FnsToEvaluate_temp.(FnNames{ff})=FnsToEvaluate.(FnNames{ff}).(Names_i{ii});
                %                     FnsToEvaluate_temp{jj}=FnsToEvaluate.(FnNames{kk}).(Names_i{ii});
                %                     FnsToEvaluateParamNames_temp(jj).Names=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{kk}).(Names_i{ii}));
                WhichFnsForCurrentPType(ff)=jj; jj=jj+1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                %  % Implicitly, WhichFnsForCurrentPType(kk)=0
                FnsAndPTypeIndicator_ii(ff)=1;
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            FnsToEvaluate_temp.(FnNames{ff})=FnsToEvaluate.(FnNames{ff});
            %                 FnsToEvaluate_temp{jj}=FnsToEvaluate.(FnNames{kk});
            %                 FnsToEvaluateParamNames_temp(jj).Names=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{kk}));
            WhichFnsForCurrentPType(ff)=jj; jj=jj+1;
            FnsAndPTypeIndicator_ii(ff)=1;
        end
    end

elseif isstruct(FnsToEvaluate) && FnsToEvaluate_StructToCell==1 % Structure, but output as cell
    FnsToEvaluateParamNames_temp=struct(); %(1).Names={}; % This is just an initialization value and will be overwritten.
    FnNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnNames);
    WhichFnsForCurrentPType=zeros(numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for ff=1:numFnsToEvaluate
        if isstruct(FnsToEvaluate.(FnNames{ff}))
            if isfield(FnsToEvaluate{ff}, Names_i{ii})
                FnsToEvaluate_temp{jj}=FnsToEvaluate.(FnNames{ff}).(Names_i{ii});
                temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{ff}).(Names_i{ii}));
                if length(temp)>(l_d+l_a+l_a+l_z)
                    FnsToEvaluateParamNames_temp(jj).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
                else
                    FnsToEvaluateParamNames_temp(jj).Names={};
                end
                WhichFnsForCurrentPType(ff)=jj; jj=jj+1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                % % Implicitly, WhichFnsForCurrentPType(kk)=0
                FnsAndPTypeIndicator_ii(ff)=1;
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            FnsToEvaluate_temp{jj}=FnsToEvaluate.(FnNames{ff});
            temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{ff}));
            if length(temp)>(l_d+l_a+l_a+l_z)
                FnsToEvaluateParamNames_temp(jj).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
            else
                FnsToEvaluateParamNames_temp(jj).Names={};
            end
            WhichFnsForCurrentPType(ff)=jj; jj=jj+1;
            FnsAndPTypeIndicator_ii(ff)=1;
        end
    end
    
else
    FnsToEvaluateParamNames_temp=struct(); %(1).Names={}; % This is just an initialization value and will be overwritten.
    numFnsToEvaluate=length(FnsToEvaluate);
    WhichFnsForCurrentPType=zeros(numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for ff=1:numFnsToEvaluate
        if isa(FnsToEvaluate{ff},'struct')
            if isfield(FnsToEvaluate{ff}, Names_i{ii})
                FnsToEvaluate_temp{jj}=FnsToEvaluate{ff}.(Names_i{ii});
                if isa(FnsToEvaluateParamNames(ff).Names,'struct')
                    FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(ff).Names.(Names_i{ii});
                else
                    FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(ff).Names;
                end
                WhichFnsForCurrentPType(ff)=jj; jj=jj+1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                % % Implicitly, WhichFnsForCurrentPType(kk)=0
                FnsAndPTypeIndicator_ii(ff)=1;
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            FnsToEvaluate_temp{jj}=FnsToEvaluate{ff};
            FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(ff).Names;
            WhichFnsForCurrentPType(ff)=jj; jj=jj+1;
            FnsAndPTypeIndicator_ii(ff)=1;
        end
    end
    varargout={FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType};
end



end
    