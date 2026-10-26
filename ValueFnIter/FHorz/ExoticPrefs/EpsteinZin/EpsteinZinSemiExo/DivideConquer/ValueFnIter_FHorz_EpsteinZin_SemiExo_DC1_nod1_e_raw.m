function [V,Policy3]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_nod1_e_raw(n_d2,n_a,n_z,n_semiz, n_e,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Divide-and-conquer version of ValueFnIter_FHorz_EpsteinZin_SemiExo_nod1_e_raw.
% Grafts the Epstein-Zin transforms onto ValueFnIter_FHorz_SemiExo_DC1_nod1_e_raw:
% the certainty-equivalent is taken over the JOINT distribution of
% (semizprime,zprime,eprime), where the (semizprime,zprime) part depends on
% the chosen d2: V' is transformed by ^ezc5 once per age (d2-independent,
% pointwise), the e'-expectation (d2-independent, so hoisted outside the d2
% loop, exactly where the vNM code takes it) and then the d2-dependent
% bothz'-expectation are taken, then temp4 (the post-certainty-equivalent
% continuation) is pointwise in (aprime,bothz), so it is computed once per d2
% over the full aprime grid and indexed exactly where the vNM code indexes
% EV_d2; the return transform and the final ^ezc7 wrap each level's entireRHS
% before its max (a monotone transform, so the divide-and-conquer monotonicity
% logic is unaffected). The final max over d2 then compares fully-transformed
% values, so it is unaffected.

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy3=zeros(2,N_a,N_semiz*N_z,N_e,N_j,'gpuArray'); % first dim indexes the optimal choice for d2, aprime

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory==0
    loweredgesize=[1,1,N_semiz*N_z,N_e];
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    loweredgesize=[1,1,N_semiz*N_z];
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
    loweredgesize=[1,1,N_semiz];
elseif vfoptions.lowmemory>=3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_e=ones(1,length(n_e));
    loweredgesize=[1,1,1];
end

bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray'); % [the vNM source preallocates the lowmemory==1 case without the N_e dimension, but then indexes (:,:,e_c,d2_c) everywhere, which errors if N_d2~=N_e]
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

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
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz,n_e, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=isfinite(ReturnMatrix_d2ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            ReturnMatrix_d2ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_d2ii now holds what was temp2 (avoids a full-size copy)
            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
            ReturnMatrix_d2ii=ezc1*ReturnMatrix_d2ii+ezc3*DiscountFactorParamsVec*WGmatrix; % warm-glow (zero if not using) % in-place: ReturnMatrix_d2ii now holds what was entireRHS_ii
            ReturnMatrix_d2ii(isnan(ReturnMatrix_d2ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(ReturnMatrix_d2ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            ReturnMatrix_d2ii(temp5)=ReturnMatrix_d2ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

            % First, we want aprime conditional on (1,a,z,e)
            [Vtempii,maxindex1]=max(ReturnMatrix_d2ii,[],1);

            % Store
            V_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex1,1); % d,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_bothz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                    % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                    ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(aprimeindexes),[maxgap(ii)+1,1,N_bothz,N_e]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                    ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                else
                    loweredge=maxindex1(1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    ReturnMatrix_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_ii now holds what was temp2 (avoids a full-size copy)
                    % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                    ReturnMatrix_ii=ezc1*ReturnMatrix_ii+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(loweredge),[1,1,N_bothz,N_e]); % in-place: ReturnMatrix_ii now holds what was entireRHS_ii
                    ReturnMatrix_ii(isnan(ReturnMatrix_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(ReturnMatrix_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    ReturnMatrix_ii(temp5)=ReturnMatrix_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(ReturnMatrix_ii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1); % no d1
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory==1

        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_d2iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,4);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=isfinite(ReturnMatrix_d2iie); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                ReturnMatrix_d2iie(becareful)=ReturnMatrix_d2iie(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_d2iie now holds what was temp2 (avoids a full-size copy)
                % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                ReturnMatrix_d2iie=ezc1*ReturnMatrix_d2iie+ezc3*DiscountFactorParamsVec*WGmatrix; % warm-glow (zero if not using) % in-place: ReturnMatrix_d2iie now holds what was entireRHS_ii
                ReturnMatrix_d2iie(isnan(ReturnMatrix_d2iie))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(ReturnMatrix_d2iie==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                ReturnMatrix_d2iie(temp5)=ReturnMatrix_d2iie(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

                % First, we want aprime conditional on (d,1,a,z,e)
                [Vtempii,maxindex1]=max(ReturnMatrix_d2iie,[],1);

                % Store
                V_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                % Second level based on monotonicity
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-1-by-n_bothz
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are maxgap(ii)+1-by-1-by-n_bothz
                        ReturnMatrix_iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_iie); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        ReturnMatrix_iie(becareful)=ReturnMatrix_iie(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_iie now holds what was temp2 (avoids a full-size copy)
                        % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                        ReturnMatrix_iie=ezc1*ReturnMatrix_iie+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(aprimeindexes),[maxgap(ii)+1,1,N_bothz]); % in-place: ReturnMatrix_iie now holds what was entireRHS_ii
                        ReturnMatrix_iie(isnan(ReturnMatrix_iie))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(ReturnMatrix_iie==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        ReturnMatrix_iie(temp5)=ReturnMatrix_iie(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                        [Vtempii,maxindex]=max(ReturnMatrix_iie,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                    else
                        loweredge=maxindex1(1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_iie); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        ReturnMatrix_iie(becareful)=ReturnMatrix_iie(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_iie now holds what was temp2 (avoids a full-size copy)
                        % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                        ReturnMatrix_iie=ezc1*ReturnMatrix_iie+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(loweredge),[1,1,N_bothz]); % in-place: ReturnMatrix_iie now holds what was entireRHS_ii
                        ReturnMatrix_iie(isnan(ReturnMatrix_iie))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(ReturnMatrix_iie==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        ReturnMatrix_iie(temp5)=ReturnMatrix_iie(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(ReturnMatrix_iie,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz

        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    % n-Monotonicity
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a_grid, a_grid(level1ii), z_valblock, e_val, ReturnFnParamsVec,4);
                    % Modify the Return Function appropriately for Epstein-Zin Preferences
                    becareful=isfinite(ReturnMatrix_d2iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    ReturnMatrix_d2iize(becareful)=ReturnMatrix_d2iize(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_d2iize now holds what was temp2 (avoids a full-size copy)
                    % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                    ReturnMatrix_d2iize=ezc1*ReturnMatrix_d2iize+ezc3*DiscountFactorParamsVec*WGmatrix; % warm-glow (zero if not using) % in-place: ReturnMatrix_d2iize now holds what was entireRHS_ii
                    ReturnMatrix_d2iize(isnan(ReturnMatrix_d2iize))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(ReturnMatrix_d2iize==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    ReturnMatrix_d2iize(temp5)=ReturnMatrix_d2iize(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

                    % First, we want aprime conditional on (d,1,a,z,e)
                    [Vtempii,maxindex1]=max(ReturnMatrix_d2iize,[],1);

                    % Store
                    V_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                    % Second level based on monotonicity
                    maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            % loweredge is 1-by-1-by-n_semiz
                            aprimeindexes=loweredge+(0:1:maxgap(ii))';
                            % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            ReturnMatrix_iize(becareful)=ReturnMatrix_iize(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_iize now holds what was temp2 (avoids a full-size copy)
                            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                            ReturnMatrix_iize=ezc1*ReturnMatrix_iize+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(aprimeindexes),[maxgap(ii)+1,1,N_semiz]); % in-place: ReturnMatrix_iize now holds what was entireRHS_ii
                            ReturnMatrix_iize(isnan(ReturnMatrix_iize))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(ReturnMatrix_iize==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            ReturnMatrix_iize(temp5)=ReturnMatrix_iize(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                            [Vtempii,maxindex]=max(ReturnMatrix_iize,[],1);
                            V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                        else
                            loweredge=maxindex1(1,ii,:);
                            % Just use aprime(ii) for everything
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            ReturnMatrix_iize(becareful)=ReturnMatrix_iize(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_iize now holds what was temp2 (avoids a full-size copy)
                            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                            ReturnMatrix_iize=ezc1*ReturnMatrix_iize+ezc3*DiscountFactorParamsVec*reshape(WGmatrix(loweredge),[1,1,N_semiz]); % in-place: ReturnMatrix_iize now holds what was entireRHS_ii
                            ReturnMatrix_iize(isnan(ReturnMatrix_iize))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(ReturnMatrix_iize==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            ReturnMatrix_iize(temp5)=ReturnMatrix_iize(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                            V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(ReturnMatrix_iize,1);
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                        end
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory>=3 % joint bothz, inner e

        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    % n-Monotonicity
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,4);
                    % Modify the Return Function appropriately for Epstein-Zin Preferences
                    becareful=isfinite(ReturnMatrix_d2iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    ReturnMatrix_d2iize(becareful)=ReturnMatrix_d2iize(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_d2iize now holds what was temp2 (avoids a full-size copy)
                    % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                    ReturnMatrix_d2iize=ezc1*ReturnMatrix_d2iize+ezc3*DiscountFactorParamsVec*WGmatrix; % warm-glow (zero if not using) % in-place: ReturnMatrix_d2iize now holds what was entireRHS_ii
                    ReturnMatrix_d2iize(isnan(ReturnMatrix_d2iize))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(ReturnMatrix_d2iize==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    ReturnMatrix_d2iize(temp5)=ReturnMatrix_d2iize(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

                    % First, we want aprime conditional on (d,1,a,z,e)
                    [Vtempii,maxindex1]=max(ReturnMatrix_d2iize,[],1);

                    % Store
                    V_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                    % Second level based on monotonicity
                    maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            % loweredge is 1-by-1-by-1
                            aprimeindexes=loweredge+(0:1:maxgap(ii))';
                            % aprime possibilities are maxgap(ii)+1-by-1
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            ReturnMatrix_iize(becareful)=ReturnMatrix_iize(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_iize now holds what was temp2 (avoids a full-size copy)
                            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                            ReturnMatrix_iize=ezc1*ReturnMatrix_iize+ezc3*DiscountFactorParamsVec*WGmatrix(aprimeindexes); % in-place: ReturnMatrix_iize now holds what was entireRHS_ii
                            ReturnMatrix_iize(isnan(ReturnMatrix_iize))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(ReturnMatrix_iize==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            ReturnMatrix_iize(temp5)=ReturnMatrix_iize(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                            [Vtempii,maxindex]=max(ReturnMatrix_iize,[],1);
                            V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                        else
                            loweredge=maxindex1(1,ii,:,:);
                            % Just use aprime(ii) for everything
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            ReturnMatrix_iize(becareful)=ReturnMatrix_iize(becareful).^ezc2(N_j); % in-place transform: ReturnMatrix_iize now holds what was temp2 (avoids a full-size copy)
                            % Compose the warm-glow INSIDE the ^ezc7 root (the V_Jplus1-branch composition with the EV term absent)
                            ReturnMatrix_iize=ezc1*ReturnMatrix_iize+ezc3*DiscountFactorParamsVec*WGmatrix(loweredge); % in-place: ReturnMatrix_iize now holds what was entireRHS_ii
                            ReturnMatrix_iize(isnan(ReturnMatrix_iize))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(ReturnMatrix_iize==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            ReturnMatrix_iize(temp5)=ReturnMatrix_iize(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                            V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(ReturnMatrix_iize,1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                        end
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    % Part of Epstein-Zin is before taking expectation (d2-independent, so done once)
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

    EV=sum(temp.*pi_e_J(1,1,:,N_j+1),3); % expectation over e' of the TRANSFORMED V' (d2-independent, part of the joint certainty-equivalent)

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,bothz)
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(~(temp4==-Inf))=(sj(N_j)*temp4(~(temp4==-Inf)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV_d2==0)=0;
            end

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz,n_e, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);
            becareful=isfinite(ReturnMatrix_d2ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            temp2_ii=ReturnMatrix_d2ii;
            temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);

            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity

            % First, we want aprime conditional on (1,a,z,e)
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

            % Store
            V_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex1,1); % d,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_bothz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    aprimez=aprimeindexes+N_a*bothzind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[(maxgap(ii)+1),1,N_bothz,N_e]);  % autoexpand level1iidiff(ii) in 2nd-dim
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                else
                    loweredge=maxindex1(1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    aprimez=loweredge+N_a*bothzind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1,N_bothz,N_e]);  % autoexpand level1iidiff(ii) in 2nd-dim
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(entireRHS_ii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1); % no d1
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform (e-independent, so done before the e loop)
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(~(temp4==-Inf))=(sj(N_j)*temp4(~(temp4==-Inf)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV_d2==0)=0;
            end

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_d2iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,4);
                becareful=isfinite(ReturnMatrix_d2iie); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                temp2_ii=ReturnMatrix_d2iie;
                temp2_ii(becareful)=ReturnMatrix_d2iie(becareful).^ezc2(N_j);

                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);

                % First, we want aprime conditional on (d,1,a,z,e)
                [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

                % Store
                V_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                % Second level based on monotonicity
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-1-by-n_bothz
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are maxgap(ii)+1-by-1-by-n_bothz
                        ReturnMatrix_iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_iie); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        temp2_ii=ReturnMatrix_iie;
                        temp2_ii(becareful)=ReturnMatrix_iie(becareful).^ezc2(N_j);
                        aprimez=aprimeindexes+N_a*bothzind;
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[(maxgap(ii)+1),1,N_bothz]); % autoexpand level1iidiff(ii) in 2nd-dim
                        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                    else
                        loweredge=maxindex1(1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_iie); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        temp2_ii=ReturnMatrix_iie;
                        temp2_ii(becareful)=ReturnMatrix_iie(becareful).^ezc2(N_j);
                        aprimez=loweredge+N_a*bothzind;
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1,N_bothz]); % autoexpand level1iidiff(ii) in 2nd-dim
                        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(entireRHS_ii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % Certainty-equivalent (and mortality-risk/warm-glow) transform (e-independent, so done before the e loop)
                temp4=EV_d2z;
                if warmglow==1
                    WGmatrixbig=WGmatrix.*ones(1,1,N_semiz);
                    becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrixbig)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_d2z==0)&(WGmatrixbig==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(~(temp4==-Inf))=(sj(N_j)*temp4(~(temp4==-Inf)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_d2z==0)=0;
                end

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    % n-Monotonicity
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a_grid, a_grid(level1ii), z_valblock, e_val, ReturnFnParamsVec,4);
                    becareful=isfinite(ReturnMatrix_d2iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_d2iize;
                    temp2_ii(becareful)=ReturnMatrix_d2iize(becareful).^ezc2(N_j);

                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);

                    % First, we want aprime conditional on (d,1,a,z,e)
                    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

                    % Store
                    V_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                    % Second level based on monotonicity
                    maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            % loweredge is 1-by-1-by-n_semiz
                            aprimeindexes=loweredge+(0:1:maxgap(ii))';
                            % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            temp2_ii=ReturnMatrix_iize;
                            temp2_ii(becareful)=ReturnMatrix_iize(becareful).^ezc2(N_j);
                            aprimez=aprimeindexes+N_a*semizind;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[(maxgap(ii)+1),1,N_semiz]);  % autoexpand level1iidiff(ii) in 2nd-dim
                            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                        else
                            loweredge=maxindex1(1,ii,:);
                            % Just use aprime(ii) for everything
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            temp2_ii=ReturnMatrix_iize;
                            temp2_ii(becareful)=ReturnMatrix_iize(becareful).^ezc2(N_j);
                            aprimez=loweredge+N_a*semizind;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1,N_semiz]);  % autoexpand level1iidiff(ii) in 2nd-dim
                            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                            V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(entireRHS_ii,1);
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                        end
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory>=3 % joint bothz, inner e
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1]

                % Certainty-equivalent (and mortality-risk/warm-glow) transform (e-independent, so done before the e loop)
                temp4=EV_d2z;
                if warmglow==1
                    becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_d2z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(~(temp4==-Inf))=(sj(N_j)*temp4(~(temp4==-Inf)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_d2z==0)=0;
                end

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    % n-Monotonicity
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,4);
                    becareful=isfinite(ReturnMatrix_d2iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_d2iize;
                    temp2_ii(becareful)=ReturnMatrix_d2iize(becareful).^ezc2(N_j);

                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);

                    % First, we want aprime conditional on (d,1,a,z,e)
                    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

                    % Store
                    V_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                    % Second level based on monotonicity
                    maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            % loweredge is 1-by-1-by-1
                            aprimeindexes=loweredge+(0:1:maxgap(ii))';
                            % aprime possibilities are maxgap(ii)+1-by-1
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            temp2_ii=ReturnMatrix_iize;
                            temp2_ii(becareful)=ReturnMatrix_iize(becareful).^ezc2(N_j);
                            aprimez=aprimeindexes;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[(maxgap(ii)+1),1]);  % autoexpand level1iidiff(ii) in 2nd-dim
                            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                        else
                            loweredge=maxindex1(1,ii,:,:);
                            % Just use aprime(ii) for everything
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            temp2_ii=ReturnMatrix_iize;
                            temp2_ii(becareful)=ReturnMatrix_iize(becareful).^ezc2(N_j);
                            aprimez=loweredge;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1]);  % autoexpand level1iidiff(ii) in 2nd-dim
                            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                            V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(entireRHS_ii,1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                        end
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,N_j)=shiftdim(aprime_ind,-1);

    end
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

    EVpre=V(:,:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation (d2-independent, so done once)
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

    EV=sum(temp.*pi_e_J(1,1,:,jj+1),3); % expectation over e' of the TRANSFORMED V' (d2-independent, part of the joint certainty-equivalent)

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj)); % reverse order

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,bothz)
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
                temp4(EV_d2==0)=0;
            end

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,4);
            becareful=isfinite(ReturnMatrix_d2ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            temp2_ii=ReturnMatrix_d2ii;
            temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);

            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

            % First, we want aprime conditional on (1,a,z,e)
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
            % Store
            V_ford2_jj(level1ii,:,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,:,d2_c)=shiftdim(maxindex1,1); % d,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_bothz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    aprimez=aprimeindexes+N_a*bothzind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[(maxgap(ii)+1),1,N_bothz,N_e]);  % autoexpand level1iidiff(ii) in 2nd-dim
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                else
                    loweredge=maxindex1(1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                    becareful=isfinite(ReturnMatrix_ii); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    aprimez=loweredge+N_a*bothzind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1,N_bothz,N_e]);  % autoexpand level1iidiff(ii) in 2nd-dim
                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    V_ford2_jj(curraindex,:,:,d2_c)=shiftdim(entireRHS_ii,1);
                    Policy_ford2_jj(curraindex,:,:,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1); % no d1
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,jj)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform (e-independent, so done before the e loop)
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
                temp4(EV_d2==0)=0;
            end

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                % n-Monotonicity
                ReturnMatrix_d2iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,4);
                becareful=isfinite(ReturnMatrix_d2iie); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                temp2_ii=ReturnMatrix_d2iie;
                temp2_ii(becareful)=ReturnMatrix_d2iie(becareful).^ezc2(jj);

                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

                entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);

                % First, we want aprime conditional on (1,a,z,e)
                [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

                % Store
                V_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                % Second level based on monotonicity
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-1-by-n_bothz
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are maxgap(ii)+1-by-1-by-n_bothz
                        ReturnMatrix_iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_iie); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        temp2_ii=ReturnMatrix_iie;
                        temp2_ii(becareful)=ReturnMatrix_iie(becareful).^ezc2(jj);
                        aprimez=aprimeindexes+N_a*bothzind;
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[(maxgap(ii)+1),1,N_bothz]); % autoexpand level1iidiff(ii) in 2nd-dim
                        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                    else
                        loweredge=maxindex1(1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_iie=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5);
                        becareful=isfinite(ReturnMatrix_iie); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                        temp2_ii=ReturnMatrix_iie;
                        temp2_ii(becareful)=ReturnMatrix_iie(becareful).^ezc2(jj);
                        aprimez=loweredge+N_a*bothzind;
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1,N_bothz]); % autoexpand level1iidiff(ii) in 2nd-dim
                        entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                        temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                        V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(entireRHS_ii,1);
                        Policy_ford2_jj(curraindex,:,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,jj)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % Certainty-equivalent (and mortality-risk/warm-glow) transform (e-independent, so done before the e loop)
                temp4=EV_d2z;
                if warmglow==1
                    WGmatrixbig=WGmatrix.*ones(1,1,N_semiz);
                    becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrixbig)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_d2z==0)&(WGmatrixbig==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_d2z==0)=0;
                end

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    % n-Monotonicity
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a_grid, a_grid(level1ii), z_valblock, e_val, ReturnFnParamsVec,4);
                    becareful=isfinite(ReturnMatrix_d2iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_d2iize;
                    temp2_ii(becareful)=ReturnMatrix_d2iize(becareful).^ezc2(jj);

                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);

                    % First, we want aprime conditional on (d,1,a,z,e)
                    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

                    % Store
                    V_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,semizblock,e_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                    % Second level based on monotonicity
                    maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            % loweredge is 1-by-1-by-n_semiz
                            aprimeindexes=loweredge+(0:1:maxgap(ii))';
                            % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            temp2_ii=ReturnMatrix_iize;
                            temp2_ii(becareful)=ReturnMatrix_iize(becareful).^ezc2(jj);
                            aprimez=aprimeindexes+N_a*semizind;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[(maxgap(ii)+1),1,N_semiz]);  % autoexpand level1iidiff(ii) in 2nd-dim
                            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(Vtempii,1);
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                        else
                            loweredge=maxindex1(1,ii,:);
                            % Just use aprime(ii) for everything
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, [n_semiz,special_n_z], special_n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            temp2_ii=ReturnMatrix_iize;
                            temp2_ii(becareful)=ReturnMatrix_iize(becareful).^ezc2(jj);
                            aprimez=loweredge+N_a*semizind;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1,N_semiz]);  % autoexpand level1iidiff(ii) in 2nd-dim
                            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                            V_ford2_jj(curraindex,semizblock,e_c,d2_c)=shiftdim(entireRHS_ii,1);
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                        end
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,jj)=shiftdim(aprime_ind,-1);

    elseif vfoptions.lowmemory>=3 % joint bothz, inner e
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1]

                % Certainty-equivalent (and mortality-risk/warm-glow) transform (e-independent, so done before the e loop)
                temp4=EV_d2z;
                if warmglow==1
                    becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_d2z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_d2z==0)=0;
                end

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    % n-Monotonicity
                    ReturnMatrix_d2iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,4);
                    becareful=isfinite(ReturnMatrix_d2iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                    temp2_ii=ReturnMatrix_d2iize;
                    temp2_ii(becareful)=ReturnMatrix_d2iize(becareful).^ezc2(jj);

                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*temp4;

                    entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                    temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);

                    % First, we want aprime conditional on (d,1,a,z,e)
                    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

                    % Store
                    V_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                    Policy_ford2_jj(level1ii,z_c,e_c,d2_c)=shiftdim(maxindex1,1); % d,aprime

                    % Second level based on monotonicity
                    maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                            % loweredge is 1-by-1-by-1
                            aprimeindexes=loweredge+(0:1:maxgap(ii))';
                            % aprime possibilities are maxgap(ii)+1-by-1
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            temp2_ii=ReturnMatrix_iize;
                            temp2_ii(becareful)=ReturnMatrix_iize(becareful).^ezc2(jj);
                            aprimez=aprimeindexes;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[(maxgap(ii)+1),1]);  % autoexpand level1iidiff(ii) in 2nd-dim
                            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(Vtempii,1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(maxindex+(loweredge-1)); % no d1
                        else
                            loweredge=maxindex1(1,ii,:,:);
                            % Just use aprime(ii) for everything
                            ReturnMatrix_iize=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_val, reshape(a_grid(loweredge),loweredgesize), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, e_val, ReturnFnParamsVec,5);
                            becareful=isfinite(ReturnMatrix_iize); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                            temp2_ii=ReturnMatrix_iize;
                            temp2_ii(becareful)=ReturnMatrix_iize(becareful).^ezc2(jj);
                            aprimez=loweredge;
                            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[1,1]);  % autoexpand level1iidiff(ii) in 2nd-dim
                            entireRHS_ii(isnan(entireRHS_ii))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                            temp5=~(entireRHS_ii==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                            V_ford2_jj(curraindex,z_c,e_c,d2_c)=shiftdim(entireRHS_ii,1);
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c)=repelem(shiftdim(loweredge,1),level1iidiff(ii),1);
                        end
                    end
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy3(2,:,:,:,jj)=shiftdim(aprime_ind,-1);

    end
end


end
