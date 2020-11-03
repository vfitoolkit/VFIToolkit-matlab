function Values=EvalFnOnAgentDist_Grid_Case1_cpu(FnToEvaluate,FnToEvaluateParamNames,PolicyIndexes,Parameters,n_d,n_a,n_z,d_grid,a_grid,z_grid)

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

% ParamCell=cell(length(FnToEvaluateParams),1);
% for ii=1:length(FnToEvaluateParams)
%     ParamCell(ii,1)={FnToEvaluateParams(ii)};
% end

[d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
a_gridvals=CreateGridvals(n_a,a_grid,2);
z_gridvals=CreateGridvals(n_z,z_grid,2);

% Includes check for cases in which no parameters are actually required
if isempty(FnToEvaluateParamNames.Names) % check for 'FnToEvaluateParamNames={}'
    Values=zeros(N_a*N_z,1);
    if l_d==0
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            Values(ii)=FnToEvaluate(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
        end
    else % l_d>0
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            Values(ii)=FnToEvaluate(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
        end
    end
else
    Values=zeros(N_a*N_z,1);
    if l_d==0
        FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnToEvaluateParamNames.Names));
        Values=zeros(N_a*N_z,1);
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            Values(ii)=FnToEvaluate(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
        end
    else % l_d>0
        FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnToEvaluateParamNames.Names));
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            Values(ii)=FnToEvaluate(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
        end
    end
end

Values=reshape(Values,[N_a,N_z]);


end