function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Epstein-Zin, one period of the FHorz transition path (slowOLG). The V input is next period value fn
% (across all ages), the V output is this period.
% The transition-path twin of ValueFnIter_FHorz_EpsteinZin_DC1_noz_raw (whose description follows): identical but
% for (i) V comes in as next period's value fn, and the continuation of age jj is next period's age jj+1
% (presaved in Vtemp_j before V(...,jj) is overwritten), and (ii) there is no V_Jplus1: the terminal age
% has no continuation, only the warm-glow.
% Divide-and-conquer version of ValueFnIter_FHorz_EpsteinZin_noz_raw.
% Grafts the Epstein-Zin transforms onto ValueFnIter_FHorz_DC1_noz_raw: temp4
% (the post-certainty-equivalent continuation) is pointwise in aprime, so it is
% computed once per age over the full aprime grid and indexed exactly where the
% vNM code indexes EV; the return transform and the final ^ezc7 wrap each
% level's entireRHS before its max (a monotone transform, so the
% divide-and-conquer monotonicity logic is unaffected).
% No shocks, so the certainty-equivalent is just the identity (Epstein-Zin
% without shocks does not make much sense, but is allowed so that models can
% be compared with/without shocks without also having to change preferences).

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
% THE THIRD OUTPUT is for vfoptions.localsearch, on a SEED sweep (vfoptions.localsearchseed, set by
% Step1): the aprime argmax FOR EACH d, [N_d,N_a,N_j], as LocalSearch/..._LSseed_noz_raw returns it.
% At the level-1 points that is maxindex1. Between them it is an EXTRA per-d max over the level-2 RHS
% (fold is d fastest, so reshape to [N_d,maxgap+1,...]); when maxgap==0 it is loweredge. Under DC's own
% monotonicity this is the unrestricted per-d argmax, as the interval is built over all d. The max that
% sets V/Policy is NOT changed (its tie-breaking is seen by every bank). Only built when asked for.
if nargout>2
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,'gpuArray');
end


%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
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
    becareful=(WGmatrix==0);
    % The warm-glow enters INSIDE the ^ezc7 root at the terminal age: build the
    % temp4-analogue of the main loop (with no EV term)
    WGmatrix(isfinite(WGmatrix))=((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j);
    WGmatrix(becareful)=0;
    % WGmatrix is a column over the full aprime grid; it is indexed by aprime below
else
    WGmatrix=zeros(N_a,1,'gpuArray');
end

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

% Modify the Return Function appropriately for Epstein-Zin Preferences
becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*shiftdim(WGmatrix,-1); % warm-glow (zero if not using); (d,aprime,a) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);

% First, we want aprime conditional on (d,1,a,z)
[~,maxindex1]=max(ReturnMatrix_ii,[],2);

% Now, get and store the full (d,aprime)
[Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);

% Store
V(level1ii,N_j)=shiftdim(Vtempii,1);
Policy(level1ii,N_j)=shiftdim(maxindex2,1); % d,aprime
if nargout>2
    aprimeReferencePolicyNew(:,level1ii,N_j)=reshape(maxindex1,[N_d,vfoptions.level1n]); % seed reference, see top
end

% Attempt for improved version
maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
        % loweredge is n_d-by-1
        aprimeindexes=loweredge+(0:1:maxgap(ii));
        % aprime possibilities are n_d-by-maxgap(ii)+1
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
        ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(aprimeindexes),[N_d*(maxgap(ii)+1),1]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
        ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        V(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(reshape(loweredge(rem(maxindex-1,N_d)+1),[],1)-1); % loweredge(given the d)
        if nargout>2
            [~,aprimeoffset]=max(reshape(ReturnMatrix_ii,[N_d,maxgap(ii)+1,length(curraindex)]),[],2); % per-d max, see top
            aprimeReferencePolicyNew(:,curraindex,N_j)=reshape(loweredge+aprimeoffset-1,[N_d,length(curraindex)]);
        end
    else
        loweredge=maxindex1(:,1,ii);
        % Just use aprime(ii) for everything
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
        ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(loweredge),[N_d*1,1]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
        ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        V(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(reshape(loweredge(rem(maxindex-1,N_d)+1),[],1)-1); % loweredge(given the d)
        if nargout>2
            aprimeReferencePolicyNew(:,curraindex,N_j)=repmat(loweredge,1,length(curraindex)); % one aprime per d, see top
        end
    end
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
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    temp2_ii=ReturnMatrix_ii;
    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);

    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1); % (d,aprime,a)

    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

    % First, we want aprime conditional on (d,1,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

    % Store
    V(level1ii,jj)=shiftdim(Vtempii,1);
    Policy(level1ii,jj)=shiftdim(maxindex2,1); % d,aprime
    if nargout>2
        aprimeReferencePolicyNew(:,level1ii,jj)=reshape(maxindex1,[N_d,vfoptions.level1n]); % seed reference, see top
    end

    % Attempt for improved version
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, reshape(a_grid(aprimeindexes),size(aprimeindexes)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimeindexes),[N_d*(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,jj)=shiftdim(maxindex,1)+N_d*(reshape(loweredge(rem(maxindex-1,N_d)+1),[],1)-1); % loweredge(given the d)
            if nargout>2
                [~,aprimeoffset]=max(reshape(entireRHS_ii,[N_d,maxgap(ii)+1,length(curraindex)]),[],2); % per-d max, see top
                aprimeReferencePolicyNew(:,curraindex,jj)=reshape(loweredge+aprimeoffset-1,[N_d,length(curraindex)]);
            end
        else
            loweredge=maxindex1(:,1,ii);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, reshape(a_grid(loweredge),size(loweredge)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(loweredge),size(loweredge)); % autoexpand level1iidiff(ii) in 2nd-dim
            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,jj)=shiftdim(maxindex,1)+N_d*(reshape(loweredge(rem(maxindex-1,N_d)+1),[],1)-1); % loweredge(given the d)
            if nargout>2
                aprimeReferencePolicyNew(:,curraindex,jj)=repmat(loweredge,1,length(curraindex)); % one aprime per d, see top
            end
        end
    end

end

%%
Policy=shiftdim(Policy,-1);

end
