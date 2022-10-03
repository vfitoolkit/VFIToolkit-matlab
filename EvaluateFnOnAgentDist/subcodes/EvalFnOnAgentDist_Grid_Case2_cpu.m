function Values=EvalFnOnAgentDist_Grid_Case2_cpu(FnToEvaluate,FnToEvaluateParamNames,PolicyIndexes,Parameters,n_d,n_a,n_z,d_grid,a_grid,z_grid)

N_a=prod(n_a);
N_z=prod(n_z);

% ParamCell=cell(length(FnToEvaluateParamNames),1);
% for ii=1:length(FnToEvaluateParamNames)
%     ParamCell(ii,1)={FnToEvaluateParamNames(ii)};
% end

[d_gridvals, ~]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,2, 2);
a_gridvals=CreateGridvals(n_a,a_grid,2);
if all(size(z_grid)==[sum(n_z),1])
    z_gridvals=CreateGridvals(n_z,z_grid,2);
elseif all(size(z_grid)==[prod(n_z),length(n_z)])
    z_gridvals=z_grid;
end

% Includes check for cases in which no parameters are actually required
if isempty(FnToEvaluateParamNames.Names) % check for 'FnsToEvaluateParamNames={}'
    Values=zeros(N_a*N_z,1);
    for ii=1:N_a*N_z
        %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
        j1=rem(ii-1,N_a)+1;
        j2=ceil(ii/N_a);
        Values(ii)=FnToEvaluate(d_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
    end
else
    Values=zeros(N_a*N_z,1);
    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnToEvaluateParamNames.Names));
    for ii=1:N_a*N_z
        %                 j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
        j1=rem(ii-1,N_a)+1;
        j2=ceil(ii/N_a);
        Values(ii)=FnToEvaluate(d_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
    end
end

Values=reshape(Values,[N_a,N_z]);


end