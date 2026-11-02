function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Epstein-Zin, one period of the FHorz transition path (slowOLG). The V input is next period value fn
% (across all ages), the V output is this period.
% The transition-path twin of ValueFnIter_FHorz_EpsteinZin_DC1_nod_raw (whose description follows): identical but
% for (i) V comes in as next period's value fn, and the continuation of age jj is next period's age jj+1
% (presaved in Vtemp_j before V(...,jj) is overwritten), and (ii) there is no V_Jplus1: the terminal age
% has no continuation, only the warm-glow.
% Divide-and-conquer version of ValueFnIter_FHorz_EpsteinZin_nod_raw.
% Grafts the Epstein-Zin transforms onto ValueFnIter_FHorz_DC1_nod_raw: temp4
% (the post-certainty-equivalent continuation) is pointwise in aprime, so it is
% computed once per age over the full aprime grid and indexed exactly where the
% vNM code indexes EV; the return transform and the final ^ezc7 wrap each
% level's entireRHS before its max (a monotone transform, so the
% divide-and-conquer monotonicity logic is unaffected).

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(1,N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
if vfoptions.lowmemory==0
    loweredgesize=[1,1,N_z];
elseif vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

zind=shiftdim(gpuArray((0:1:N_z-1)),-1);

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,N_j);
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
    becareful=(WGmatrix==0);
    % The warm-glow enters INSIDE the ^ezc7 root at the terminal age: build the
    % temp4-analogue of the main loop (with no EV term)
    WGmatrix(isfinite(WGmatrix))=((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j);
    WGmatrix(becareful)=0;
    % WGmatrix is a column over the full aprime grid; it is indexed by aprime below
else
    WGmatrix=zeros(N_a,1,'gpuArray');
end

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    % Modify the Return Function appropriately for Epstein-Zin Preferences
    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
    ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
    ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix; % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
    ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);

    %Calc the max and it's index
    [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);

    V(level1ii,:,N_j)=shiftdim(Vtempii,1);
    Policy(1,level1ii,:,N_j)=maxindex1;

    % Attempt for improved version
    maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are maxgap(ii)+1-by-1-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
            ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(aprimeindexes),[maxgap(ii)+1,1,N_z]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
            ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,N_j)=shiftdim(Vtempii,1);
            Policy(1,curraindex,:,N_j)=maxindex+loweredge-1;
        else
            loweredge=maxindex1(1,ii,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
            ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(loweredge),[1,1,N_z]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
            ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);
            V(curraindex,:,N_j)=shiftdim(ReturnMatrix_ii,1);
            Policy(1,curraindex,:,N_j)=repelem(loweredge,1,level1iidiff(ii),1);
        end
    end
elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
        ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix; % warm-glow (zero if not using) % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
        ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);

        %Calc the max and it's index
        [Vtempii,maxindex1]=max(ReturnMatrix_ii,[],1);

        V(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
        Policy(1,level1ii,z_c,N_j)=maxindex1;

        % Attempt for improved version
        maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are maxgap(ii)+1-by-1
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix(aprimeindexes); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                Policy(1,curraindex,z_c,N_j)=maxindex+loweredge-1;
            else
                loweredge=maxindex1(1,ii);
                % Just use aprime(ii) for everything
                % No need for reshape(a_grid(loweredge),loweredgesize) as is just a single point
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_z, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix(loweredge); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);
                V(curraindex,z_c,N_j)=shiftdim(ReturnMatrix_ii,1);
                Policy(1,curraindex,z_c,N_j)=loweredge;
            end
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
    Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated
    EVpre=VKronNext_j;
    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

    %Calc the expectation term (except beta)
    EV=temp.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,z)
    temp4=EV;
    if warmglow==1
        WGmatrixbig=WGmatrix.*ones(1,1,N_z);
        becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrixbig)); % both are finite
        temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
        temp4((EV==0)&(WGmatrixbig==0))=0; % Is actually zero
    else % not using warmglow
        temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
        temp4(EV==0)=0;
    end

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);

        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

        % Calc the max and it's index
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

        V(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policy(1,level1ii,:,jj)=maxindex1;

        % Attempt for improved version
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are maxgap(ii)+1-by-1-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[maxgap(ii)+1,1,N_z]); % autofill level1iidiff(ii) in 2nd-dim
                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii,1);
                Policy(1,curraindex,:,jj)=maxindex+loweredge-1;
            else
                loweredge=maxindex1(1,ii,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                aprimez=loweredge+N_a*zind;
                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1,N_z]);
                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                V(curraindex,:,jj)=shiftdim(entireRHS_ii,1);
                Policy(1,curraindex,:,jj)=repelem(loweredge,1,level1iidiff(ii),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            temp4_z=temp4(:,:,z_c);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);

            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4_z;

            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);

            %Calc the max and it's index
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

            V(level1ii,z_c,jj)=shiftdim(Vtempii,1);
            Policy(1,level1ii,z_c,jj)=maxindex1;

            % Attempt for improved version
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4_z(aprimeindexes); % autofill level1iidiff(ii) in 2nd-dim
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    Policy(1,curraindex,z_c,jj)=maxindex+loweredge-1;
                else
                    loweredge=maxindex1(1,ii);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_z, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4_z(loweredge);
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    V(curraindex,z_c,jj)=shiftdim(entireRHS_ii,1);
                    Policy(1,curraindex,z_c,jj)=loweredge;
                end
            end
        end
    end

end


end
