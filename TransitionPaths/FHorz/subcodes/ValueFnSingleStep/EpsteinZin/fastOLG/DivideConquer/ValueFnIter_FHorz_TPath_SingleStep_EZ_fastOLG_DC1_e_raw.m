function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin, one period of the FHorz transition path, fastOLG (every age at once), with divide-and-conquer.
% The V input is next period value fn, the V output is this period.
% The Epstein-Zin graft of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_e_raw. It does, vectorised over the
% age dimension, exactly what the slowOLG raw ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_e_raw does age by age.
% See ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_raw for the graft. With e, the transform
% (ezc4*V')^ezc5 comes before BOTH expectations (over e', then over z'), as in the slowOLG raw: one
% certainty equivalent over (z',e') jointly. Next period's V has (a,j) stacked in its rows, so the age
% exponent is laid along the rows there.
% The age dimension is dim 4 of the first level's return matrix, [d,aprime,a,j,z,e], and dim 3 of the
% second level's, [d & aprime,a,j,z,e], so each has its own shaping of the age-dependent constants.
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z,e), rather than standard (a,z,e,j)
% V is (a,j)-by-z-by-e

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% fastOLG, so a-j-z
z_gridvals_J=shiftdim(z_gridvals_J,-3);
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]);

Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray'); % first dim indexes the optimal choice for d and aprime
% THE THIRD OUTPUT is for vfoptions.localsearch, on a SEED sweep (vfoptions.localsearchseed, set by
% Step1): the aprime argmax FOR EACH d, [N_d,N_a,N_j,N_z,N_e] (the order of Policy). See
% ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_e_raw. Only built when asked for.
if nargout>2
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,N_z,N_e,'gpuArray');
end

%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);

jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
zBind=shiftdim(gpuArray(0:1:N_z-1),-2);
eBind=shiftdim(gpuArray(0:1:N_e-1),-3);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z*N_e (note: N_aprime is just equal to N_a)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

%% The age-dependent Epstein-Zin constants, along the age dimension: dim 3 (the expectation, the second
% level), dim 4 (the first level)
ezc2_J=reshape(ezc2,[1,1,N_j]);
ezc5_J=reshape(ezc5,[1,1,N_j]);
ezc6_J=reshape(ezc6,[1,1,N_j]);
ezc7_J=reshape(ezc7,[1,1,N_j]);
ezc8_J=reshape(ezc8,[1,1,N_j]);
sj_J=reshape(sj,[1,1,N_j]);
if vfoptions.EZoneminusbeta==1
    ezc1_J=reshape(1-DiscountFactor_J,[1,1,N_j]); % (1-beta)* on this period's return, age by age
elseif vfoptions.EZoneminusbeta==2
    ezc1_J=reshape(1-sj(:).*DiscountFactor_J,[1,1,N_j]); % (1-sj*beta)* on this period's return, age by age
else
    ezc1_J=ezc1;
end
ezc1_J4=shiftdim(ezc1_J,-1); % [1,1,1,N_j] (a scalar is unchanged)
ezc2_J4=shiftdim(ezc2_J,-1);
ezc7_J4=shiftdim(ezc7_J,-1);

%% The warm-glow of bequests, (aprime,1,j)
if warmglow==1
    WGmatrixraw=zeros(N_a,1,N_j,'gpuArray');
    for jj=1:N_j
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw(:,1,jj)=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
    end
    WGfinite=isfinite(WGmatrixraw);
    WGmatrix=ezc4*WGmatrixraw;
    WGmatrix(~WGfinite)=1; % set aside, the power is for the finite entries
    WGmatrix=WGmatrix.^ezc5_J;
    WGmatrix(~WGfinite)=WGmatrixraw(~WGfinite);
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
    WGmatrix(WGmatrixraw==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
end

pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
Vnext=V(N_a+1:end,:,:); % next period's value fn at ages 2..N_j, (a,j)-by-z-by-e: the continuation of ages 1..N_j-1
% Part of Epstein-Zin is before taking expectation
ezc5_rows=repelem(ezc5(1:N_j-1),N_a,1); % the age exponent, laid along the (a,j) rows
Vnextfinite=isfinite(Vnext);
temp=ezc4*Vnext;
temp(~Vnextfinite)=1; % set aside, the power is for the finite entries
temp=temp.^ezc5_rows;
temp(~Vnextfinite)=Vnext(~Vnextfinite);
temp(Vnext==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
% Expectation over e
EVpre=[sum(temp.*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
clear Vnext Vnextfinite temp EVpre
V=zeros(N_a,N_j,N_z,N_e,'gpuArray'); % V is over (a,j)

%% The certainty equivalent, age by age
temp4=EV;
if warmglow==1
    becareful=(~(temp4==-Inf)) & isfinite(WGmatrix); % both are finite
    temp4b=temp4;
    temp4b(~becareful)=1; % set aside
    WGb=WGmatrix;
    WGb(~isfinite(WGb))=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J+(1-sj_J).*WGb.^ezc8_J).^ezc6_J;
    temp4(becareful)=temp4b(becareful);
    temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
    clear temp4b WGb
else % not using warmglow
    becareful=~(temp4==-Inf);
    temp4b=temp4;
    temp4b(~becareful)=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J).^ezc6_J;
    temp4(becareful)=temp4b(becareful);
    temp4(EV==0)=0;
    clear temp4b
end
% The terminal age has no continuation: its certainty equivalent is the warm-glow term alone, exactly the
% terminal formula of the slowOLG raw (the formula above, with its zero expectation, can differ from it:
% 0^ezc8 is Inf when ezc8<0, and sj(N_j)*Inf is NaN when sj(N_j)=0)
if warmglow==1
    WGterminal=WGmatrix(:,1,N_j);
    becareful=(WGterminal==0);
    WGterminal(isfinite(WGterminal))=((1-sj(N_j))*WGterminal(isfinite(WGterminal)).^ezc8(N_j)).^ezc6(N_j);
    WGterminal(becareful)=0;
    temp4(:,1,N_j,:)=WGterminal.*ones(1,1,1,N_z);
else
    temp4(:,1,N_j,:)=0;
end
DiscountedEV=repelem(shiftdim((ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4,-1),N_d,1,1,1); % [N_d,N_aprime,1,N_j,N_z]
clear EV temp4 becareful

if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, N_j, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % [d,aprime,a,j,z,e]

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
    ReturnMatrix_ii(~becareful)=1; % set aside
    ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J4;
    ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;

    entireRHS_ii=ezc1_J4.*ReturnMatrix_ii+DiscountedEV; % (d,aprime,a and j,z,e), autofills a and e for expectation term
    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS_ii(~temp5)=1; % set aside
    entireRHS_ii=entireRHS_ii.^ezc7_J4;
    entireRHS_ii(~temp5)=-Inf;

    % First, we want aprime conditional on (d,1,a,j,z,e)
    [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'+N_d*vfoptions.level1n*(0:1:N_j-1)+N_d*vfoptions.level1n*N_j*shiftdim((0:1:N_z-1),-1)+N_d*vfoptions.level1n*N_j*N_z*shiftdim((0:1:N_e-1),-2)); % aprime

    % Store
    V(level1ii,:,:,:)=shiftdim(Vtempii,2);
    Policy(level1ii,:,:,:)=maxindex2+N_d*(maxindex1d-1); % d,aprime
    if nargout>2
        aprimeReferencePolicyNew(:,level1ii,:,:,:)=reshape(maxindex1,[N_d,vfoptions.level1n,N_j,N_z,N_e]); % seed reference, see top
    end

    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j-by-N_z-by-N_e
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, N_j, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            % [d & aprime,a,j,z,e]
            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
            ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
            ReturnMatrix_ii(~becareful)=1; % set aside
            ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
            ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
            daprimejz=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1,1)+N_d*N_a*jind+N_d*N_a*N_j*zind; % all the d, with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEV(daprimejz(:)),[N_d*(maxgap(ii)+1),level1iidiff(ii),N_j,N_z,N_e]); % autofill e for the expectations
            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
            entireRHS_ii(~temp5)=1; % set aside
            entireRHS_ii=entireRHS_ii.^ezc7_J;
            entireRHS_ii(~temp5)=-Inf;
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(Vtempii,1);
            d_ind=rem(maxindex-1,N_d)+1;
            allind=d_ind+N_d*jBind+N_d*N_j*zBind+N_d*N_j*N_z*eBind;
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(maxindex+N_d*(reshape(loweredge(allind),size(maxindex))-1),1); % loweredge(given the d)
            if nargout>2
                [~,aprimeoffset]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,level1iidiff(ii),N_j,N_z,N_e]),[],2); % per-d max, see top
                aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=reshape(loweredge+aprimeoffset-1,[N_d,level1iidiff(ii),N_j,N_z,N_e]);
            end
        else
            loweredge=maxindex1(:,1,ii,:,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, N_j, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            % [d,a,j,z,e]
            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
            ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
            ReturnMatrix_ii(~becareful)=1; % set aside
            ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
            ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
            daprimejz=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d*N_a*jind+N_d*N_a*N_j*zind; % all the d, with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEV(daprimejz(:)),[N_d,level1iidiff(ii),N_j,N_z,N_e]); % autofill e for the expectations
            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
            entireRHS_ii(~temp5)=1; % set aside
            entireRHS_ii=entireRHS_ii.^ezc7_J;
            entireRHS_ii(~temp5)=-Inf;
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(Vtempii,1);
            d_ind=rem(maxindex-1,N_d)+1;
            allind=d_ind+N_d*jBind+N_d*N_j*zBind+N_d*N_j*N_z*eBind;
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(maxindex+N_d*(reshape(loweredge(allind),size(maxindex))-1),1); % loweredge(given the d)
            if nargout>2
                aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=repmat(reshape(loweredge,[N_d,1,N_j,N_z,N_e]),1,level1iidiff(ii),1,1,1); % one aprime per d, see top
            end
        end
    end

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e)

        % n-Monotonicity
        ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,1);
        % [d,aprime,a,j,z]
        becareful=isfinite(ReturnMatrix_ii_e); % a zero return is a value, not infeasible
        ReturnMatrixnonfinite=ReturnMatrix_ii_e(~becareful);
        ReturnMatrix_ii_e(~becareful)=1; % set aside
        ReturnMatrix_ii_e=ReturnMatrix_ii_e.^ezc2_J4;
        ReturnMatrix_ii_e(~becareful)=ReturnMatrixnonfinite;

        entireRHS_ii=ezc1_J4.*ReturnMatrix_ii_e+DiscountedEV; % (d,aprime,a and j,z), autofills a for expectation term
        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
        entireRHS_ii(~temp5)=1; % set aside
        entireRHS_ii=entireRHS_ii.^ezc7_J4;
        entireRHS_ii(~temp5)=-Inf;

        % First, we want aprime conditional on (d,1,a,j)
        [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
        % Now, we get the d and we store the (d,aprime) and the

        %Calc the max and it's index
        [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
        maxindex2=shiftdim(maxindex2,2); % d
        maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'+N_d*vfoptions.level1n*(0:1:N_j-1)+N_d*vfoptions.level1n*N_j*shiftdim((0:1:N_z-1),-1)); % aprime

        % Store
        V(level1ii,:,:,e_c)=shiftdim(Vtempii,2);
        Policy(level1ii,:,:,e_c)=maxindex2+N_d*(maxindex1d-1); % d,aprime
        if nargout>2
            aprimeReferencePolicyNew(:,level1ii,:,:,e_c)=reshape(maxindex1,[N_d,vfoptions.level1n,N_j,N_z]); % seed reference, see top
        end

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j-by-N_z
                ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, N_j, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
                % [d & aprime,a,j,z]
                becareful=isfinite(ReturnMatrix_ii_e); % a zero return is a value, not infeasible
                ReturnMatrixnonfinite=ReturnMatrix_ii_e(~becareful);
                ReturnMatrix_ii_e(~becareful)=1; % set aside
                ReturnMatrix_ii_e=ReturnMatrix_ii_e.^ezc2_J;
                ReturnMatrix_ii_e(~becareful)=ReturnMatrixnonfinite;
                daprimejz=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1,1)+N_d*N_a*jind+N_d*N_a*N_j*zind; % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ezc1_J.*ReturnMatrix_ii_e+reshape(DiscountedEV(daprimejz(:)),[N_d*(maxgap(ii)+1),level1iidiff(ii),N_j,N_z]);
                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
                entireRHS_ii(~temp5)=1; % set aside
                entireRHS_ii=entireRHS_ii.^ezc7_J;
                entireRHS_ii(~temp5)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindex-1,N_d)+1;
                allind=d_ind+N_d*jBind+N_d*N_j*zBind;
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(maxindex+N_d*(reshape(loweredge(allind),size(maxindex))-1),1); % loweredge(given the d)
                if nargout>2
                    [~,aprimeoffset]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,level1iidiff(ii),N_j,N_z]),[],2); % per-d max, see top
                    aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=reshape(loweredge+aprimeoffset-1,[N_d,level1iidiff(ii),N_j,N_z]);
                end
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, n_z, special_n_e, N_j, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
                % [d,a,j,z]
                becareful=isfinite(ReturnMatrix_ii_e); % a zero return is a value, not infeasible
                ReturnMatrixnonfinite=ReturnMatrix_ii_e(~becareful);
                ReturnMatrix_ii_e(~becareful)=1; % set aside
                ReturnMatrix_ii_e=ReturnMatrix_ii_e.^ezc2_J;
                ReturnMatrix_ii_e(~becareful)=ReturnMatrixnonfinite;
                daprimejz=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1,1)+N_d*N_a*jind+N_d*N_a*N_j*zind; % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ezc1_J.*ReturnMatrix_ii_e+reshape(DiscountedEV(daprimejz(:)),[N_d,level1iidiff(ii),N_j,N_z]);
                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
                entireRHS_ii(~temp5)=1; % set aside
                entireRHS_ii=entireRHS_ii.^ezc7_J;
                entireRHS_ii(~temp5)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindex-1,N_d)+1;
                allind=d_ind+N_d*jBind+N_d*N_j*zBind;
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(maxindex+N_d*(reshape(loweredge(allind),size(maxindex))-1),1); % loweredge(given the d)
                if nargout>2
                    aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=repmat(reshape(loweredge,[N_d,1,N_j,N_z]),1,level1iidiff(ii),1,1); % one aprime per d, see top
                end
            end
        end
    end
elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,j,prod(n_z),l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e)

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid(level1ii), z_vals, e_vals, ReturnFnParamsAgeMatrix,1);
            % [d,aprime,a,j]
            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
            ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
            ReturnMatrix_ii(~becareful)=1; % set aside
            ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J4;
            ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;

            entireRHS_ii=ezc1_J4.*ReturnMatrix_ii+DiscountedEV_z; % (d,aprime,a and j,z), autofills j for expectation term
            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
            entireRHS_ii(~temp5)=1; % set aside
            entireRHS_ii=entireRHS_ii.^ezc7_J4;
            entireRHS_ii(~temp5)=-Inf;

            % First, we want aprime conditional on (d,1,a,j)
            [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
            % Now, we get the d and we store the (d,aprime) and the

            %Calc the max and it's index
            [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
            maxindex2=shiftdim(maxindex2,2); % d
            maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'+N_d*vfoptions.level1n*(0:1:N_j-1)); % aprime

            % Store
            V(level1ii,:,z_c,e_c)=shiftdim(Vtempii,2);
            Policy(level1ii,:,z_c,e_c)=maxindex2+N_d*(maxindex1d-1); % d,aprime
            if nargout>2
                aprimeReferencePolicyNew(:,level1ii,:,z_c,e_c)=reshape(maxindex1,[N_d,vfoptions.level1n,N_j]); % seed reference, see top
            end

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j
                    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, N_j, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, e_vals, ReturnFnParamsAgeMatrix,2);
                    % [d & aprime,a,j]
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
                    ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
                    ReturnMatrix_ii(~becareful)=1; % set aside
                    ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
                    ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
                    daprimej=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d*N_a*jind; % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEV_z(daprimej(:)),[N_d*(maxgap(ii)+1),level1iidiff(ii),N_j]);
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
                    entireRHS_ii(~temp5)=1; % set aside
                    entireRHS_ii=entireRHS_ii.^ezc7_J;
                    entireRHS_ii(~temp5)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindex-1,N_d)+1;
                    allind=d_ind+N_d*jBind;
                    Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(maxindex+N_d*(reshape(loweredge(allind),size(maxindex))-1),1); % loweredge(given the d)
                    if nargout>2
                        [~,aprimeoffset]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,level1iidiff(ii),N_j]),[],2); % per-d max, see top
                        aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=reshape(loweredge+aprimeoffset-1,[N_d,level1iidiff(ii),N_j]);
                    end
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_e(ReturnFn, n_d, special_n_z, special_n_e, N_j, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, e_vals, ReturnFnParamsAgeMatrix,2);
                    % [d,a,j]
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
                    ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
                    ReturnMatrix_ii(~becareful)=1; % set aside
                    ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
                    ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
                    daprimej=(1:1:N_d)'+N_d*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d*N_a*jind; % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEV_z(daprimej(:)),[N_d,level1iidiff(ii),N_j]);
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
                    entireRHS_ii(~temp5)=1; % set aside
                    entireRHS_ii=entireRHS_ii.^ezc7_J;
                    entireRHS_ii(~temp5)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindex-1,N_d)+1;
                    allind=d_ind+N_d*jBind;
                    Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(maxindex+N_d*(reshape(loweredge(allind),size(maxindex))-1),1); % loweredge(given the d)
                    if nargout>2
                        aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=repmat(reshape(loweredge,[N_d,1,N_j]),1,level1iidiff(ii),1); % one aprime per d, see top
                    end
                end
            end
        end
     end
end

%% fastOLG with z & e, so need output to take certain shapes
V=reshape(V,[N_a*N_j,N_z,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_z,N_e]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
