function [V,Policy]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_nod1_noz_raw(n_d2,n_a,n_semiz,N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Divide-and-conquer version of ValueFnIter_FHorz_EpsteinZin_SemiExo_nod1_noz_raw.
% Grafts the Epstein-Zin transforms onto ValueFnIter_FHorz_SemiExo_DC1_nod1_noz_raw:
% V' is transformed by ^ezc5 once per age (d2-independent, pointwise), the
% d2-dependent expectation is taken, then temp4 (the post-certainty-equivalent
% continuation) is pointwise in (aprime,semiz), so it is computed once per d2
% over the full aprime grid and indexed exactly where the vNM code indexes
% EV_d2; the return transform and the final ^ezc7 wrap each level's entireRHS
% before its max (a monotone transform, so the divide-and-conquer monotonicity
% logic is unaffected). The final max over d2 then compares fully-transformed
% values, so it is unaffected.

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy=zeros(2,N_a,N_semiz,N_j,'gpuArray');

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1);

loweredgesize=[1,1,N_semiz];

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

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
    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        % The warm-glow enters INSIDE the ^ezc7 root at the terminal age, so build the
        % temp4-analogue of the main loop (with no EV term):
        WGmatrix(isfinite(WGmatrix))=((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j);
        WGmatrix(becareful)=0;
    end
    % WGmatrix is a column over the full aprime grid; it is indexed by aprime below
else
    WGmatrix=zeros(N_a,1,'gpuArray');
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_semiz, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=isfinite(ReturnMatrix_d2ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            ReturnMatrix_d2ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_d2ii now holds what was temp2 (avoids a full-size copy)
            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
            ReturnMatrix_d2ii=ezc1*ReturnMatrix_d2ii+ezc3*DiscountFactorParamsVec*WGmatrix; % warm-glow (zero if not using) % in-place: ReturnMatrix_d2ii now holds what was entireRHS_ii
            ReturnMatrix_d2ii(isnan(ReturnMatrix_d2ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(ReturnMatrix_d2ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            ReturnMatrix_d2ii(temp5)=ReturnMatrix_d2ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

            % First, we want aprime conditional on (1,a,z)
            [Vtempii,maxindex1]=max(ReturnMatrix_d2ii,[],1);

            % Store
            V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex1,1); % d,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_semiz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                    % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                    ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(aprimeindexes),[maxgap(ii)+1,1,N_semiz]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                    ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(curraindex,:,d2_c)=maxindex+(loweredge-1); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_semiz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                    % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                    ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(loweredge),[1,1,N_semiz]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                    ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(ReturnMatrix_ii,1);
                    Policy_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                end
            end
        end

    elseif vfoptions.lowmemory>=1 % loop semiz

        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);

            for semiz_c=1:N_semiz
                semiz_val=semiz_gridvals_J(semiz_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_semiz, d2_val, a_grid, a_grid(level1ii), semiz_val, ReturnFnParamsVec,4);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=isfinite(ReturnMatrix_d2ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                ReturnMatrix_d2ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_d2ii now holds what was temp2 (avoids a full-size copy)
                % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                ReturnMatrix_d2ii=ezc1*ReturnMatrix_d2ii+ezc3*DiscountFactorParamsVec*WGmatrix; % warm-glow (zero if not using) % in-place: ReturnMatrix_d2ii now holds what was entireRHS_ii
                ReturnMatrix_d2ii(isnan(ReturnMatrix_d2ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(ReturnMatrix_d2ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                ReturnMatrix_d2ii(temp5)=ReturnMatrix_d2ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

                % First, we want aprime conditional on (1,a,z)
                [Vtempii,maxindex1]=max(ReturnMatrix_d2ii,[],1);

                % Store
                V_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                % Second level based on monotonicity
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-1-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are maxgap(ii)+1-by-1-by-1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_semiz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                        % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                        ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix(aprimeindexes); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                        ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=maxindex+(loweredge-1); % loweredge(given the d and z)
                    else
                        loweredge=maxindex1(1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_semiz, d2_val, reshape(a_grid(loweredge),[1,1,1]), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                        % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                        ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*WGmatrix(loweredge); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                        ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(ReturnMatrix_ii,1);
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                    end
                end
            end
        end

    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policy(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);    % First, switch V_Jplus1 into Kron form

    % Part of Epstein-Zin is before taking expectation (d2-independent, so done once)
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

    if vfoptions.lowmemory==0

        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j); % reverse order

            EV_d2=temp.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,semiz)
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_semiz);
                becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(~(temp4==-Inf))=(sj(N_j)*temp4(~(temp4==-Inf)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV_d2==0)=0;
            end

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_semiz, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);
            becareful=isfinite(ReturnMatrix_d2ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            temp2_ii=ReturnMatrix_d2ii;
            temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);

            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

            % First, we want aprime conditional on (1,a,z)
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

            % Store
            V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex1,1); % d,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_semiz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    aprimez=aprimeindexes+N_a*semizind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(curraindex,:,d2_c)=maxindex+(loweredge-1); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_semiz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    aprimez=loweredge+N_a*semizind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(entireRHS_ii,1);
                    Policy_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                end
            end
        end

    elseif vfoptions.lowmemory>=1 % loop semiz

        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j); % reverse order

            for semiz_c=1:N_semiz
                semiz_val=semiz_gridvals_J(semiz_c,:,N_j);

                EV_d2z=temp.*pi_semiz(semiz_c,:);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % sum over z', leaving a singular second dimension

                % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
                temp4=EV_d2z;
                if warmglow==1
                    becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_d2z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(~(temp4==-Inf))=(sj(N_j)*temp4(~(temp4==-Inf)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_d2z==0)=0;
                end

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_semiz, d2_val, a_grid, a_grid(level1ii), semiz_val, ReturnFnParamsVec,4);
                becareful=isfinite(ReturnMatrix_d2ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                temp2_ii=ReturnMatrix_d2ii;
                temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);

                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);

                % First, we want aprime conditional on (1,a,z)
                [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

                % Store
                V_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                % Second level based on monotonicity
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-1-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are maxgap(ii)+1-by-1-by-1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_semiz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimeindexes),[(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=maxindex+(loweredge-1); % loweredge(given the d and z)
                    else
                        loweredge=maxindex1(1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_semiz, d2_val, reshape(a_grid(loweredge),[1,1,1]), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(loweredge),[1,1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(entireRHS_ii,1);
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                    end
                end
            end
        end

    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj;
    Policy(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policy(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);

end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end


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

    EVpre=V(:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation (d2-independent, so done once)
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

    if vfoptions.lowmemory==0

        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,jj); % reverse order

            EV_d2=temp.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,semiz)
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_semiz);
                becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
                temp4(EV_d2==0)=0;
            end

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_semiz, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,4);
            becareful=isfinite(ReturnMatrix_d2ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            temp2_ii=ReturnMatrix_d2ii;
            temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);

            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

            % First, we want aprime conditional on (1,a,semiz)
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

            % Store
            V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex1,1); % aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_semiz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    aprimez=aprimeindexes+N_a*semizind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(curraindex,:,d2_c)=maxindex+(loweredge-1); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, n_semiz, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    aprimez=loweredge+N_a*semizind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(entireRHS_ii,1);
                    Policy_ford2_jj(curraindex,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                end
            end
        end

    elseif vfoptions.lowmemory>=1 % loop semiz

        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,jj); % reverse order

            for semiz_c=1:N_semiz
                semiz_val=semiz_gridvals_J(semiz_c,:,jj);

                EV_d2z=temp.*pi_semiz(semiz_c,:);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % sum over z', leaving a singular second dimension

                % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
                temp4=EV_d2z;
                if warmglow==1
                    becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_d2z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_d2z==0)=0;
                end

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_semiz, d2_val, a_grid, a_grid(level1ii), semiz_val, ReturnFnParamsVec,4);
                becareful=isfinite(ReturnMatrix_d2ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                temp2_ii=ReturnMatrix_d2ii;
                temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);

                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);

                % First, we want aprime conditional on (1,a,semiz)
                [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

                % Store
                V_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(maxindex1,1); % aprime

                % Second level based on monotonicity
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-1-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are maxgap(ii)+1-by-1-by-1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_semiz, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimeindexes),[(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=maxindex+(loweredge-1); % loweredge(given the d and z)
                    else
                        loweredge=maxindex1(1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d2, special_n_semiz, d2_val, reshape(a_grid(loweredge),[1,1,1]), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(loweredge),[1,1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(entireRHS_ii,1);
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                    end
                end
            end
        end

    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,jj)=V_jj;
    Policy(1,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    Policy(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);

end


end
