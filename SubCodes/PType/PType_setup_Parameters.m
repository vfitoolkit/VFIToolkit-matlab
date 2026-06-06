function Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,structordim)
% Parameters are allowed to be given as structure, or as vector/matrix
% (in terms of their dependence on fixed type).
%
% structordim controls which forms are resolved:
%   1 = struct only (lenient: pass through if struct lacks the field)
%   2 = vector/matrix only (trailing dim of length N_i)
%   3 = both (matches single-level PType convention)

Parameters_temp=Parameters;
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);
for kField=1:nFields
    val=Parameters.(FullParamNames{kField});
    if structordim==1
        if isa(val, 'struct') % Check for permanent type in structure form
            names=fieldnames(val);
            for jj=1:length(names)
                if strcmp(names{jj},iistr)
                    Parameters_temp.(FullParamNames{kField})=val.(names{jj});
                end
            end
        end
    elseif structordim==2
        if ~isa(val, 'struct') && any(size(val)==N_i) % Check for permanent type in vector/matrix form.
            [~,ptypedim]=max(size(val)==N_i);
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=val(ii,:);
            elseif ptypedim==2
                Parameters_temp.(FullParamNames{kField})=val(:,ii);
            end
        end
    elseif structordim==3
        if isa(val, 'struct') % Check for permanent type in structure form
            names=fieldnames(val);
            for jj=1:length(names)
                if strcmp(names{jj},iistr)
                    Parameters_temp.(FullParamNames{kField})=val.(names{jj});
                end
            end
        elseif any(size(val)==N_i) % Check for permanent type in vector/matrix form.
            [~,ptypedim]=max(size(val)==N_i);
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=val(ii,:);
            elseif ptypedim==2
                Parameters_temp.(FullParamNames{kField})=val(:,ii);
            end
        end
    end
end

end
