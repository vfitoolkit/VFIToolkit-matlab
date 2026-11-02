function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin, one period of the FHorz transition path, fastOLG (every age at once), with divide-and-conquer.
% The V input is next period value fn, the V output is this period.
% The Epstein-Zin graft of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_raw. It does, vectorised over the
% age dimension, exactly what the slowOLG raw ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_raw does age by age.
% See ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_raw for the graft: the certainty equivalent of next
% period's value is formed once, pointwise in aprime, and stands where the vNM raw has the discounted EV;
% every return matrix gets ^ezc2 and ezc1*, and every right-hand side gets ^ezc7 before its max.
% The age dimension is dim 4 of the first level's return matrix, [d,aprime,a,j,z], and dim 3 of the
% second level's, [d & aprime,a,j,z], so each has its own shaping of the age-dependent constants.
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z
% pi_z_J is (j,z',z) for fastOLG; it arrives with N_j-1 slices (the transitions from periods 1..N_j-1) and a j=N_j slot of zeros is appended below
% z_gridvals_J is (j,N_z,l_z) for fastOLG

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% fastOLG, so a-j-z
Policy=zeros(N_a,N_j,N_z,'gpuArray'); % first dim indexes the optimal choice for d and aprime
% THE THIRD OUTPUT is for vfoptions.localsearch, on a SEED sweep (vfoptions.localsearchseed, set by
% Step1): the aprime argmax FOR EACH d, [N_d,N_a,N_j,N_z]. See ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_raw.
% Only built when asked for.
if nargout>2
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,N_z,'gpuArray');
end

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
zBind=shiftdim(gpuArray(0:1:N_z-1),-2);


%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

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
Vnext=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % next period's value fn at ages 2..N_j: the continuation of ages 1..N_j-1
% Part of Epstein-Zin is before taking expectation
Vnextfinite=isfinite(Vnext);
temp=ezc4*Vnext;
temp(~Vnextfinite)=1; % set aside, the power is for the finite entries
temp=temp.^ezc5_J(1,1,1:N_j-1);
temp(~Vnextfinite)=Vnext(~Vnextfinite);
temp(Vnext==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
EVpre=zeros(N_a,1,N_j,N_z,'gpuArray');
EVpre(:,1,1:N_j-1,:)=temp; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
clear Vnext Vnextfinite temp EVpre
V=zeros(N_a,N_j,N_z,'gpuArray'); % V is over (a,j)

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
DiscountedEV=shiftdim((ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4,-1); % [1,N_a,1,N_j,N_z] 1st dim will autofill d, 3rd dim will autofill a
clear EV temp4 becareful


if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_z, N_j, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % [d,aprime,a,j,z]

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
    ReturnMatrix_ii(~becareful)=1; % set aside
    ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J4;
    ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;

    entireRHS_ii=ezc1_J4.*ReturnMatrix_ii+DiscountedEV; % (d,aprime,a and j,z), autofills a for expectation term
    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
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
    V(level1ii,:,:)=shiftdim(Vtempii,2);
    Policy(level1ii,:,:)=maxindex2+N_d*(maxindex1d-1); % d,aprime
    if nargout>2
        aprimeReferencePolicyNew(:,level1ii,:,:)=reshape(maxindex1,[N_d,vfoptions.level1n,N_j,N_z]); % seed reference, see top
    end

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j-by-N_z
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_z, N_j, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            % [d & aprime,a,j,z]
            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
            ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
            ReturnMatrix_ii(~becareful)=1; % set aside
            ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
            ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
            aprimez=aprimeindexes+N_a*jind+N_a*N_j*zind;
            entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEV(aprimez(:)),[N_d*(maxgap(ii)+1),1,N_j,N_z]); % broadcast level1iidiff at dim 2
            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
            entireRHS_ii(~temp5)=1; % set aside
            entireRHS_ii=entireRHS_ii.^ezc7_J;
            entireRHS_ii(~temp5)=-Inf;
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*jBind+N_d*N_j*zBind)-1),1); % loweredge(given the d)
            if nargout>2
                [~,aprimeoffset]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,level1iidiff(ii),N_j,N_z]),[],2); % per-d max, see top
                aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:,:)=reshape(loweredge+aprimeoffset-1,[N_d,level1iidiff(ii),N_j,N_z]);
            end
        else
            loweredge=maxindex1(:,1,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_z, N_j, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            % [d,a,j,z]
            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
            ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
            ReturnMatrix_ii(~becareful)=1; % set aside
            ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
            ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
            aprimez=loweredge+N_a*jind+N_a*N_j*zind;
            entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEV(aprimez(:)),[N_d,1,N_j,N_z]); % broadcast level1iidiff at dim 2
            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
            entireRHS_ii(~temp5)=1; % set aside
            entireRHS_ii=entireRHS_ii.^ezc7_J;
            entireRHS_ii(~temp5)=-Inf;
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*jBind+N_d*N_j*zBind)-1),1); % loweredge(given the d)
            if nargout>2
                aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:,:)=repmat(reshape(loweredge,[N_d,1,N_j,N_z]),1,level1iidiff(ii)); % one aprime per d, see top
            end
        end
    end

elseif vfoptions.lowmemory==1

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,N_j,N_z,l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a_grid, a_grid(level1ii), z_vals, ReturnFnParamsAgeMatrix,1);
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
        V(level1ii,:,z_c)=shiftdim(Vtempii,2);
        Policy(level1ii,:,z_c)=maxindex2+N_d*(maxindex1d-1); % d,aprime
        if nargout>2
            aprimeReferencePolicyNew(:,level1ii,:,z_c)=reshape(maxindex1,[N_d,vfoptions.level1n,N_j]); % seed reference, see top
        end

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-N_j
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_z, N_j, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, ReturnFnParamsAgeMatrix,2);
                % [d & aprime,a,j]
                becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
                ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
                ReturnMatrix_ii(~becareful)=1; % set aside
                ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
                ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
                aprime=aprimeindexes+N_a*jind;
                entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEV_z(aprime(:)),[N_d*(maxgap(ii)+1),1,N_j]); % broadcast level1iidiff at dim 2
                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
                entireRHS_ii(~temp5)=1; % set aside
                entireRHS_ii=entireRHS_ii.^ezc7_J;
                entireRHS_ii(~temp5)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(Vtempii,1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*jBind)-1),1); % loweredge(given the d)
                if nargout>2
                    [~,aprimeoffset]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,level1iidiff(ii),N_j]),[],2); % per-d max, see top
                    aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=reshape(loweredge+aprimeoffset-1,[N_d,level1iidiff(ii),N_j]);
                end
            else
                loweredge=maxindex1(:,1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_z, N_j, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, ReturnFnParamsAgeMatrix,2);
                % [d,a,j]
                becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
                ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
                ReturnMatrix_ii(~becareful)=1; % set aside
                ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
                ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
                aprime=loweredge+N_a*jind;
                entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEV_z(aprime(:)),[N_d,1,N_j]); % broadcast level1iidiff at dim 2
                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
                entireRHS_ii(~temp5)=1; % set aside
                entireRHS_ii=entireRHS_ii.^ezc7_J;
                entireRHS_ii(~temp5)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(Vtempii,1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*jBind)-1),1); % loweredge(given the d)
                if nargout>2
                    aprimeReferencePolicyNew(:,level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=repmat(reshape(loweredge,[N_d,1,N_j]),1,level1iidiff(ii)); % one aprime per d, see top
                end
            end
        end
    end
end

%% fastOLG with z, so need to output to take certain shapes
V=reshape(V,[N_a*N_j,N_z]);
% Policy=reshape(Policy,[1,N_a,N_j,N_z]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point



end
