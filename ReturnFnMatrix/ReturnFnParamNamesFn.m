function ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters)

if isempty(N_j) % N_j is optional input
    N_j=0; % Infinite horizon
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
if n_a(1)==0
    l_a=0;
else
    l_a=length(n_a);
end
l_aprime=l_a;
l_z=length(n_z);
if n_z(1)==0
    l_z=0;
end
if isfield(vfoptions,'n_semiz')
    l_z=l_z+length(vfoptions.n_semiz);
end
l_e=0;
if isfield(vfoptions,'n_e')
    l_e=length(vfoptions.n_e);
end
if isfield(vfoptions,'experienceasset')
    % One of the endogenous states should only be counted once.
    l_aprime=l_aprime-vfoptions.experienceasset;
end
if isfield(vfoptions,'experienceassetu')
    % One of the endogenous states should only be counted once.
    l_aprime=l_aprime-vfoptions.experienceassetu;
end
if isfield(vfoptions,'riskyasset')
    % One of the endogenous states should only be counted once.
    l_aprime=l_aprime-vfoptions.riskyasset;
end
if isfield(vfoptions,'residualasset')
    % One of the endogenous states should only be counted once.
    l_aprime=l_aprime-vfoptions.residualasset;
end
if isfield(vfoptions,'refine_d')
    % Remove d2
    l_d=l_d-vfoptions.refine_d(2);
end
if isfield(vfoptions,'endotype')
    if max(vfoptions.endotype)==1
        l_aprime=l_aprime-sum(vfoptions.endotype); % Some of the endogenous states is an endogenous type, so it won't appear at this
        l_a=l_a-sum(vfoptions.endotype); % Some of the endogenous states is an endogenous type, so it won't appear at this
        l_z=l_z+sum(vfoptions.endotype); % The variables after z is the endogenous types
    end
end


% Figure out ReturnFnParamNames from ReturnFn
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_aprime+l_a+l_z+l_e) % This is largely pointless, the ReturnFn is always going to have some parameters
    ReturnFnParamNames={temp{l_d+l_aprime+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z,e)
else
    ReturnFnParamNames={};
end
% [l_d,l_aprime,l_a,l_z,l_e]
% ReturnFnParamNames
% clear l_d l_a l_z l_e % These are all messed up so make sure they are not reused later

% Decided to do the check of the parameters to here.
% Inputs to ReturnFn should all be in Parameters, and should either be scalar or age-dependent
for pp=1:length(ReturnFnParamNames)
    if ~isfield(Parameters,ReturnFnParamNames{pp})
        error(['Cannot find the parameter ',ReturnFnParamNames{pp}, ' in the Parameters structure (it is needed as an input to the ReturnFn)'])
    else
        if isscalar(Parameters.(ReturnFnParamNames{pp}))
            %  scalar is fine
        elseif all(size(Parameters.(ReturnFnParamNames{pp}))==[1,N_j]) || all(size(Parameters.(ReturnFnParamNames{pp}))==[N_j,1])
            % age-dependent vector is fine
        else
            error(['The parameter ',ReturnFnParamNames{pp}, ' must be scalar or age-dependent (check the size of this parameter; it is needed as an input to the ReturnFn)'])
        end
    end
end


end
