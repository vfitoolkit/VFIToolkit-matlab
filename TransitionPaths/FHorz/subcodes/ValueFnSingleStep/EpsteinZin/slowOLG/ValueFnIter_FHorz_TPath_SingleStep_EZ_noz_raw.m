function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_EZ_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Epstein-Zin, one period of the FHorz transition path (slowOLG). The V input is next period value fn
% (across all ages), the V output is this period.
% The transition-path twin of ValueFnIter_FHorz_EpsteinZin_noz_raw (whose description follows): identical but
% for (i) V comes in as next period's value fn, and the continuation of age jj is next period's age jj+1
% (presaved in Vtemp_j before V(...,jj) is overwritten), and (ii) there is no V_Jplus1: the terminal age
% has no continuation, only the warm-glow.
% No shocks, so the certainty-equivalent is just the identity (Epstein-Zin
% without shocks does not make much sense, but is allowed so that models can
% be compared with/without shocks without also having to change preferences).

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a

%% j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
elseif vfoptions.EZoneminusbeta==2
    ezc1=1-sj(N_j)*DiscountFactorParamsVec;
end

% If there is a warm-glow at end of the final period, evaluate the warmglowfn
if warmglow==1
    WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,N_j);
    WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
    WGmatrix=WGmatrixraw;
    WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(N_j);
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
    WGmatrix(WGmatrixraw==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
    % The warm-glow enters INSIDE the ^ezc7 root at the terminal age (the bequest is the
    % terminal condition of the recursion, composed like any continuation value; see Kraft,
    % Munk & Weiss (2022) and the Epstein-Zin appendix of the Intro to Life-Cycle Models).
    % So build the temp4-analogue of the main loop (with no EV term):
    becareful=(WGmatrix==0);
    WGmatrix(isfinite(WGmatrix))=((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j);
    WGmatrix(becareful)=0;
    % Now just make it the right shape (currently has aprime, needs the d dimension; autoexpands over a)
    WGmatrix=kron(WGmatrix,ones(N_d,1));
else
    WGmatrix=0;
end


ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);
% (d,aprime,a)

% Modify the Return Function appropriately for Epstein-Zin Preferences
becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
ReturnMatrix(becareful)=ReturnMatrix(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix now holds what was temp2 (avoids a full-size copy)

% Compose the warm-glow INSIDE the ^ezc7 root (the main-loop composition with the EV term absent)
ReturnMatrix=ezc1*ReturnMatrix+ezc3*DiscountFactorParamsVec*WGmatrix; % autoexpands over the a dimension % in-place: ReturnMatrix now holds what was entireRHS

ReturnMatrix(isnan(ReturnMatrix))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
temp5=~(ReturnMatrix==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
ReturnMatrix(temp5)=ReturnMatrix(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

% Calc the max and it's index
[Vtemp,maxindex]=max(ReturnMatrix,[],1);
V(:,N_j)=Vtemp;
Policy(:,N_j)=maxindex;

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % If there is a warm-glow, evaluate the warmglowfn
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        WGmatrix(WGmatrixraw==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
        % Now just make it the right shape (currently has aprime, needs the d dimension; autoexpands over a)
        WGmatrix=kron(WGmatrix,ones(N_d,1));
    end

    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,jj); % Grab this before it is replaced/updated
    EVpre=VKronNext_j;
    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);
    % (d,aprime,a)

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    temp2=ReturnMatrix;
    temp2(becareful)=ReturnMatrix(becareful).^ezc2(jj);

    % No shocks, so no expectation to take (the certainty-equivalent is just the identity)
    EV=temp;
    entireEV=kron(EV,ones(N_d,1));

    temp4=entireEV;
    if warmglow==1
        becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
        temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
        temp4((entireEV==0)&(WGmatrix==0))=0; % Is actually zero
    else % not using warmglow
        temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
        temp4(entireEV==0)=0;
    end

    entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4; % autoexpands over the a dimension

    entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,jj)=Vtemp;
    Policy(:,jj)=maxindex;
end


%%
Policy=shiftdim(Policy,-1);

end
