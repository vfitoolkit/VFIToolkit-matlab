function [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType]=PType_FnsToEvaluate(FnsToEvaluate, FnsToEvaluateParamNames,Names_i,ii,l_d,l_a,l_z)

FnsToEvaluate_temp={};
FnsToEvaluateParamNames_temp=struct(); %(1).Names={}; % This is just an initialization value and will be overwritten

if isstruct(FnsToEvaluate)
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(AggVarNames);
    WhichFnsForCurrentPType=zeros(numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for ff=1:numFnsToEvaluate
        if isstruct(FnsToEvaluate.(AggVarNames{ff}))
            if isfield(FnsToEvaluate{ff}, Names_i{ii})
                FnsToEvaluate_temp{jj}=FnsToEvaluate.(AggVarNames{ff}).(Names_i{ii});
                temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}).(Names_i{ii}));
                if length(temp)>(l_d+l_a+l_a+l_z)
                    FnsToEvaluateParamNames_temp(jj).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
                else
                    FnsToEvaluateParamNames_temp(jj).Names={};
                end
                WhichFnsForCurrentPType(ff)=jj; jj=jj+1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                % % Implicitly, WhichFnsForCurrentPType(kk)=0
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            FnsToEvaluate_temp{jj}=FnsToEvaluate.(AggVarNames{ff});
            temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
            if length(temp)>(l_d+l_a+l_a+l_z)
                FnsToEvaluateParamNames_temp(jj).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
            else
                FnsToEvaluateParamNames_temp(jj).Names={};
            end
            WhichFnsForCurrentPType(ff)=jj; jj=jj+1;
        end
    end
else
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
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            FnsToEvaluate_temp{jj}=FnsToEvaluate{ff};
            FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(ff).Names;
            WhichFnsForCurrentPType(ff)=jj; jj=jj+1;
        end
    end 
end

end
    