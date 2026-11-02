function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_nod_noz_raw(V,n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Epstein-Zin, one period of the FHorz transition path (slowOLG). The V input is next period value fn
% (across all ages), the V output is this period.
% The transition-path twin of ValueFnIter_FHorz_EpsteinZin_DC1_nod_noz_raw (whose description follows): identical but
% for (i) V comes in as next period's value fn, and the continuation of age jj is next period's age jj+1
% (presaved in Vtemp_j before V(...,jj) is overwritten), and (ii) there is no V_Jplus1: the terminal age
% has no continuation, only the warm-glow.
% Divide-and-conquer version of ValueFnIter_FHorz_EpsteinZin_nod_noz_raw.
% Grafts the Epstein-Zin transforms onto ValueFnIter_FHorz_DC1_nod_noz_raw: temp4
% (the post-certainty-equivalent continuation) is pointwise in aprime, so it is
% computed once per age over the full aprime grid and indexed exactly where the
% vNM code indexes EV; the return transform and the final ^ezc7 wrap each
% level's entireRHS before its max (a monotone transform, so the
% divide-and-conquer monotonicity logic is unaffected).
% No shocks, so the certainty-equivalent is just the identity (Epstein-Zin
% without shocks does not make much sense, but is allowed so that models can
% be compared with/without shocks without also having to change preferences).

N_a=prod(n_a);

Policy=zeros(1,N_a,N_j,'gpuArray'); % indexes the optimal choice for aprime rest of dimensions a,z

%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

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
    % The warm-glow enters INSIDE the ^ezc7 root at the terminal age: build the
    % temp4-analogue of the main loop (with no EV term)
    becareful=(WGmatrix==0);
    WGmatrix(isfinite(WGmatrix))=((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j);
    WGmatrix(becareful)=0;
    % WGmatrix is a column over the full aprime grid; it is indexed by aprime below
else
    WGmatrix=zeros(N_a,1,'gpuArray');
end


% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

% Modify the Return Function appropriately for Epstein-Zin Preferences
becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)

% Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1 composition with the EV term absent)
ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix; % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii

ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

%Calc the max and it's index
[Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);

V(level1ii,N_j)=shiftdim(Vtempii,1);
Policy(1,level1ii,N_j)=maxindex1;

for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(maxindex1(ii):maxindex1(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
    ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix(maxindex1(ii):maxindex1(ii+1)); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
    ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);
    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
    V(curraindex,N_j)=shiftdim(Vtempii,1);
    Policy(1,curraindex,N_j)=maxindex+maxindex1(ii)-1;
end

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
        % WGmatrix is a column over the full aprime grid; combined into temp4 below
    end

    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,jj); % Grab this before it is replaced/updated
    EVpre=VKronNext_j;
    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

    % No shocks, so no expectation to take (the certainty-equivalent is just the identity)
    EV=temp;

    % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
    temp4=EV;
    if warmglow==1
        becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
        temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
        temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
    else % not using warmglow
        temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
        temp4(EV==0)=0;
    end

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    temp2_ii=ReturnMatrix_ii;
    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);

    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

    %Calc the max and it's index
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

    V(level1ii,jj)=shiftdim(Vtempii,1);
    Policy(1,level1ii,jj)=maxindex1;

    % Note: Did a runtime test, this simple version is faster than actually checking if maxgap(ii)=0 like in all the other DC1 codes.
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(maxindex1(ii):maxindex1(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4(maxindex1(ii):maxindex1(ii+1));
        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(curraindex,jj)=shiftdim(Vtempii,1);
        Policy(1,curraindex,jj)=maxindex+maxindex1(ii)-1;
    end

end

end
