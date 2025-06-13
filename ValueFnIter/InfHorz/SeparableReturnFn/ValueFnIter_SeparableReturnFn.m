function [V,Policy]=ValueFnIter_SeparableReturnFn(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

ReturnFnParamNames.R1=ReturnFnParamNamesFn(ReturnFn.R1,n_d+n_a,0,n_z,0,vfoptions,Parameters);
% (d,a,z,..)
ReturnFnParamNames.R2=ReturnFnParamNamesFn(ReturnFn.R2,[n_a,1],0,0,0,vfoptions,Parameters);
% (aprime,R1result,..)

ReturnFnParamNames.R1
ReturnFnParamNames.R2

% If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
V0=gpuArray(V0);
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);

if vfoptions.verbose==1
    vfoptions
end

%% Switch to z_gridvals
if vfoptions.alreadygridvals==0
    if vfoptions.parallel<2
        % only basics allowed with cpu
        z_gridvals=z_grid;
    else
        [z_gridvals, pi_z, vfoptions]=ExogShockSetup(n_z,z_grid,pi_z,Parameters,vfoptions,3);
    end
elseif vfoptions.alreadygridvals==1
    z_gridvals=z_grid;
end

%% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec.R1=CreateVectorFromParams(Parameters, ReturnFnParamNames.R1);
ReturnFnParamsVec.R2=CreateVectorFromParams(Parameters, ReturnFnParamNames.R2);

ParamCell.R1=cell(length(ReturnFnParamsVec.R1),1);
for ii=1:length(ReturnFnParamsVec.R1)
    ParamCell.R1(ii,1)={ReturnFnParamsVec.R1(ii)};
end

ParamCell.R2=cell(length(ReturnFnParamsVec.R2),1);
for ii=1:length(ReturnFnParamsVec.R2)
    ParamCell.R2(ii,1)={ReturnFnParamsVec.R2(ii)};
end

if isfield(vfoptions,'exoticpreferences')
    if vfoptions.exoticpreferences~=3
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
        if vfoptions.exoticpreferences==0
            DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.
end

%% Solve 
% % (I still don't know if I want to use refine, for now there is no d variable)
% if strcmp(vfoptions.solnmethod,'purediscretization')
%     % TO BE IMPLEMENTED
% end
% 
% if strcmp(vfoptions.solnmethod,'purediscretization_refinement') 
%     % TO BE IMPLEMENTED
% end


aprime1vals = a_grid;              %(a',1,1)
a1vals      = shiftdim(a_grid,-1); %(1,a,1)

ReturnFn.R1
ReturnFn.R2

% Cash is (1,a,z)
if l_z==1
    cash_on_hand=arrayfun(ReturnFn.R1, a1vals,shiftdim(z_gridvals(:,1),-2),ParamCell.R1{:});
elseif l_z==2
    cash_on_hand=arrayfun(ReturnFn.R1, a1vals,shiftdim(z_gridvals(:,1),-2),shiftdim(z_gridvals(:,2),-2),ParamCell.R1{:});
end
% ReturnMatrix is (a',a,z)
ReturnMatrix=arrayfun(ReturnFn.R2, aprime1vals,cash_on_hand,ParamCell.R2{:});

N_z = n_z;
N_a = n_a;
Tolerance = vfoptions.tolerance;
Howards   = vfoptions.howards;
Howards2  = vfoptions.maxhowards;

VKron = V0;

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

addindexforaz=N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1);

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance

    VKronold=VKron;

    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*shiftdim(pi_z',-1); %kron(ones(N_a,1),pi_z(z_c,:));
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %aprime by a by z

    %Calc the max and it's index
    [VKron,PolicyIndexes]=max(entireRHS,[],1);

    tempmaxindex=shiftdim(PolicyIndexes,1)+addindexforaz; % aprime index, add the index for a and z

    Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards

    PolicyIndexes=PolicyIndexes(:); % a by z (this shape is just convenient for Howards)
    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:); 
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 
        for Howards_counter=1:Howards
            EVKrontemp=VKron(PolicyIndexes,:);
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter=tempcounter+1;

end %end while
  
Policy=reshape(PolicyIndexes,[N_a,N_z]);
Policy = shiftdim(Policy,-1);
V = VKron;

end %end function