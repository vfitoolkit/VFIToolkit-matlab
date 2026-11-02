function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_GI1_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin, one period of the FHorz transition path, fastOLG (every age at once), with divide-and-conquer
% and the grid interpolation layer. The V input is next period value fn, the V output is this period.
% The Epstein-Zin graft of ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI1_nod_raw. It does, vectorised
% over the age dimension, exactly what the slowOLG raw ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_GI1_nod_raw
% does age by age. See ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_nod_raw for the graft, and
% ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_GI1_raw for the (d,z) version: the interpolation acts on
% EV=E[(ezc4*V')^ezc5] BEFORE the certainty-equivalent power, which is then applied on the coarse and the
% fine grid; the warm-glow is evaluated exactly on the fine grid, the coarse pass using its strided
% subset. Every right-hand side gets ^ezc7 before its max.
% The age dimension is dim 3 of every return matrix here, [aprime,a,j,z] (CreateReturnFnMatrix_fastOLG_Disc_DC1_nod,
% both Level 1 and Level 2).
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z
% pi_z_J is (j,z',z) for fastOLG; it arrives with N_j-1 slices (the transitions from periods 1..N_j-1) and a j=N_j slot of zeros is appended below
% z_gridvals_J is (j,N_z,l_z) for fastOLG

N_a=prod(n_a);
N_z=prod(n_z);

% fastOLG, so a-j-z
Policy=zeros(3,N_a,N_j,N_z,'gpuArray'); % first dim indexes the optimal choice for aprime (midpoint, L2, L2 flag)
% THE THIRD OUTPUT is for vfoptions.localsearch, on a SEED sweep (vfoptions.localsearchseed, set by
% Step1): the COARSE aprime argmax per (a,j,z), [1,N_a,N_j,N_z], which is this raw's own clamped
% midpoints_jj, the same thing the GI raw's seed reference is (see fastOLG GI1_nod_raw). Only built
% when asked for.
if nargout>2
    aprimeReferencePolicyNew=zeros(1,N_a,N_j,N_z,'gpuArray');
end

z_gridvals_J=shiftdim(z_gridvals_J,-2); % [1,1,N_j,N_z,l_z]

%%
% Preallocate
if vfoptions.lowmemory==0
    midpoints_jj=zeros(1,N_a,N_j,N_z,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(1,N_a,N_j,'gpuArray');
end

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

jind=shiftdim(gpuArray(0:1:N_j-1),-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-2);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

%% The age-dependent Epstein-Zin constants, along the age dimension (dim 3: the expectation, and every
% return matrix)
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

%% The warm-glow of bequests, exactly on the fine grid, (aprime,1,j); the coarse pass uses the strided subset
if warmglow==1
    WGmatrixfineraw=zeros(n2aprime,1,N_j,'gpuArray');
    for jj=1:N_j
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixfineraw(:,1,jj)=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n2aprime, aprime_grid, WGParamsVec);
    end
    WGfinite=isfinite(WGmatrixfineraw);
    WGmatrixfine=ezc4*WGmatrixfineraw;
    WGmatrixfine(~WGfinite)=1; % set aside, the power is for the finite entries
    WGmatrixfine=WGmatrixfine.^ezc5_J;
    WGmatrixfine(~WGfinite)=WGmatrixfineraw(~WGfinite);
    WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
    WGmatrixfine(WGmatrixfineraw==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
    WGmatrix=WGmatrixfine(1:(n2short+1):end,:,:); % coarse-grid subset, (aprime,1,j)
    clear WGmatrixfineraw WGfinite
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

% Interpolate EV over aprime_grid (BEFORE the certainty-equivalent power ^ezc6)
EVinterp=interp1(a_grid,EV,aprime_grid);

%% The certainty equivalent, age by age, on the coarse and on the fine grid
temp4=EV;
temp4interp=EVinterp;
if warmglow==1
    becareful=(~(temp4==-Inf)) & isfinite(WGmatrix); % both are finite
    temp4b=temp4;
    temp4b(~becareful)=1; % set aside
    WGb=WGmatrix;
    WGb(~isfinite(WGb))=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J+(1-sj_J).*WGb.^ezc8_J).^ezc6_J;
    temp4(becareful)=temp4b(becareful);
    temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
    becareful=(~(temp4interp==-Inf)) & isfinite(WGmatrixfine); % both are finite
    temp4b=temp4interp;
    temp4b(~becareful)=1; % set aside
    WGb=WGmatrixfine;
    WGb(~isfinite(WGb))=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J+(1-sj_J).*WGb.^ezc8_J).^ezc6_J;
    temp4interp(becareful)=temp4b(becareful);
    temp4interp((EVinterp==0)&(WGmatrixfine==0))=0; % Is actually zero
    clear temp4b WGb
else % not using warmglow
    becareful=~(temp4==-Inf);
    temp4b=temp4;
    temp4b(~becareful)=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J).^ezc6_J;
    temp4(becareful)=temp4b(becareful);
    temp4(EV==0)=0;
    becareful=~(temp4interp==-Inf);
    temp4b=temp4interp;
    temp4b(~becareful)=1; % set aside
    temp4b=(sj_J.*temp4b.^ezc8_J).^ezc6_J;
    temp4interp(becareful)=temp4b(becareful);
    temp4interp(EVinterp==0)=0;
    clear temp4b
end
% The terminal age has no continuation: its certainty equivalent is the warm-glow term alone, exactly the
% terminal formula of the slowOLG raw (the formula above, with its zero expectation, can differ from it:
% 0^ezc8 is Inf when ezc8<0, and sj(N_j)*Inf is NaN when sj(N_j)=0)
if warmglow==1
    WGterminal=WGmatrixfine(:,1,N_j);
    becareful=(WGterminal==0);
    WGterminal(isfinite(WGterminal))=((1-sj(N_j))*WGterminal(isfinite(WGterminal)).^ezc8(N_j)).^ezc6(N_j);
    WGterminal(becareful)=0;
    temp4interp(:,1,N_j,:)=WGterminal.*ones(1,1,1,N_z);
    temp4(:,1,N_j,:)=WGterminal(1:(n2short+1):end).*ones(1,1,1,N_z); % coarse-grid subset
else
    temp4interp(:,1,N_j,:)=0;
    temp4(:,1,N_j,:)=0;
end
clear EV EVinterp becareful

DiscountedEV=(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4;
DiscountedEVinterp=(ezc3*reshape(DiscountFactor_J,[1,1,N_j])).*temp4interp;
clear temp4 temp4interp

if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_z, N_j, a_grid, a_grid(level1ii), z_gridvals_J, ReturnFnParamsAgeMatrix,1);
    % [aprime,a,j,z]
    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
    ReturnMatrix_ii(~becareful)=1; % set aside
    ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
    ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;

    entireRHS_ii=ezc1_J.*ReturnMatrix_ii+DiscountedEV; % (aprime,a and j,z), autofills a for expectation term
    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS_ii(~temp5)=1; % set aside
    entireRHS_ii=entireRHS_ii.^ezc7_J;
    entireRHS_ii(~temp5)=-Inf;

    % Calc the max and it's index
    [~,maxindex1]=max(entireRHS_ii,[],1);

    % Just keep the 'midpoint' version of maxindex1 [as GI]
    midpoints_jj(1,level1ii,:,:)=maxindex1;

    % Attempt for improved version
    maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-N_j-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii))'; % ' due to no d
            % aprimeindexes is 1-by-maxgap(ii)+1-by-N_j-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,n_z,N_j,a_grid(aprimeindexes),a_grid(level1ii(ii)+1:level1ii(ii+1)-1),z_gridvals_J,ReturnFnParamsAgeMatrix,2);
            % [aprime,a,j,z]
            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
            ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
            ReturnMatrix_ii(~becareful)=1; % set aside
            ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
            ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
            aprimejz=repelem(aprimeindexes,1,level1iidiff(ii),1)+N_a*jind+N_a*N_j*zind;
            entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEV(aprimejz(:)),[maxgap(ii)+1,level1iidiff(ii),N_j,N_z]);
            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
            entireRHS_ii(~temp5)=1; % set aside
            entireRHS_ii=entireRHS_ii.^ezc7_J;
            entireRHS_ii(~temp5)=-Inf;
            [~,maxindex]=max(entireRHS_ii,[],1);
            midpoints_jj(1,curraindex,:,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(1,ii,:,:);
            midpoints_jj(1,curraindex,:,:)=repelem(loweredge,1,length(curraindex),1);
        end
    end

    % Turn this into the 'midpoint'
    midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
    if nargout>2
        aprimeReferencePolicyNew(1,:,:,:)=midpoints_jj; % seed reference, see top
    end
    % midpoint is 1-by-n_a-by-N_j-by-n_z
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
    % aprime possibilities are n2long-by-n_a-by-N_j-by-n_z
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,n_z,N_j,aprime_grid(aprimeindexes),a_grid,z_gridvals_J,ReturnFnParamsAgeMatrix,2);
    % [aprime,a,j,z]
    % Modify the Return Function appropriately for Epstein-Zin Preferences (in place: -Inf stays -Inf, so
    % the L2 flag below still reads the raw -Inf pattern)
    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
    ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
    ReturnMatrix_ii(~becareful)=1; % set aside
    ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
    ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
    clear ReturnMatrixnonfinite
    aprimejz=aprimeindexes+n2aprime*jind+n2aprime*N_j*zind;
    entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEVinterp(aprimejz(:)),[n2long,N_a,N_j,N_z]);
    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
    entireRHS_ii(~temp5)=1; % set aside
    entireRHS_ii=entireRHS_ii.^ezc7_J;
    entireRHS_ii(~temp5)=-Inf;
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    V=reshape(Vtempii,[N_a*N_j,N_z]);
    Policy(1,:,:,:)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
    Policy(2,:,:,:)=shiftdim(maxindexL2,-1); % aprimeL2ind

    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
    isInfLower    = (ReturnMatrix_ii(1,     :,:,:) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:,:,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,:,:) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

elseif vfoptions.lowmemory==1

    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,'gpuArray'); % preallocate: V is over (a,j,z)

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,:,z_c,:); % z_gridvals_J has shape (1,1,N_j,N_z,l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);
        DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,z_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, special_n_z, N_j, a_grid, a_grid(level1ii), z_vals, ReturnFnParamsAgeMatrix,1);
        % [aprime,a,j]
        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
        ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
        ReturnMatrix_ii(~becareful)=1; % set aside
        ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
        ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;

        entireRHS_ii=ezc1_J.*ReturnMatrix_ii+DiscountedEV_z; % (aprime,a and j,z), autofills j for expectation term
        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
        entireRHS_ii(~temp5)=1; % set aside
        entireRHS_ii=entireRHS_ii.^ezc7_J;
        entireRHS_ii(~temp5)=-Inf;

        % Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(1,level1ii,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1-by-N_j
                aprimeindexes=loweredge+(0:1:maxgap(ii))'; % ' due to no d
                % aprimeindexes is 1-by-maxgap(ii)+1-by-N_j
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,special_n_z,N_j,a_grid(aprimeindexes),a_grid(level1ii(ii)+1:level1ii(ii+1)-1),z_vals,ReturnFnParamsAgeMatrix,2);
                % [aprime,a,j]
                becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
                ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
                ReturnMatrix_ii(~becareful)=1; % set aside
                ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
                ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
                aprimej=repelem(aprimeindexes,1,level1iidiff(ii),1)+N_a*jind; % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEV_z(aprimej(:)),[maxgap(ii)+1,level1iidiff(ii),N_j]);
                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
                entireRHS_ii(~temp5)=1; % set aside
                entireRHS_ii=entireRHS_ii.^ezc7_J;
                entireRHS_ii(~temp5)=-Inf;
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,ii,:);
                midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        if nargout>2
            aprimeReferencePolicyNew(1,:,:,z_c)=midpoints_jj; % seed reference, see top
        end
        % midpoint is 1-by-n_a-by-N_j
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a-by-N_j
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn,special_n_z,N_j,aprime_grid(aprimeindexes),a_grid,z_vals,ReturnFnParamsAgeMatrix,2);
        % [aprime,a,j]
        % Modify the Return Function appropriately for Epstein-Zin Preferences (in place: -Inf stays -Inf, so
        % the L2 flag below still reads the raw -Inf pattern)
        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible
        ReturnMatrixnonfinite=ReturnMatrix_ii(~becareful);
        ReturnMatrix_ii(~becareful)=1; % set aside
        ReturnMatrix_ii=ReturnMatrix_ii.^ezc2_J;
        ReturnMatrix_ii(~becareful)=ReturnMatrixnonfinite;
        aprimej=aprimeindexes+n2aprime*jind;
        entireRHS_ii=ezc1_J.*ReturnMatrix_ii+reshape(DiscountedEVinterp_z(aprimej(:)),[n2long,N_a,N_j]);
        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included
        entireRHS_ii(~temp5)=1; % set aside
        entireRHS_ii=entireRHS_ii.^ezc7_J;
        entireRHS_ii(~temp5)=-Inf;
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,z_c)=reshape(Vtempii,[N_a*N_j,1]);
        Policy(1,:,:,z_c)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
        Policy(2,:,:,z_c)=shiftdim(maxindexL2,-1); % aprimeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        Policy(3,:,:,z_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
    end
end


%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust; % lower grid point
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

%% fastOLG with z, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z]);
% Policy=reshape(Policy,[size(Policy,1),N_a,N_j,N_z]);


end
