function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_EZ_LSseed_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin SEED SWEEP, finite horizon, slowOLG, with z and e, with a d and without the grid
% interpolation layer. An UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d,
% [N_d,N_a,N_z,N_e,N_j], which is what vfoptions.localsearch needs as its reference.
% The search and the outputs are EXACTLY those of ValueFnIter_FHorz_TPath_SingleStep_LSseed_e_raw (see
% it, and ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw for why this exists), and the right-hand side is
% EXACTLY that of ValueFnIter_FHorz_TPath_SingleStep_EZ_e_raw, branch by branch: the continuation is
% transformed (ezc4,ezc5) and THEN the expectation over e is taken (not the vNM Vnext up front), then the
% expectation over z (in the form the EZ raw uses at that lowmemory), and it is aggregated with survival
% and the warm-glow (ezc6,ezc8) into temp4, which depends on (aprime,z) only, so it is computed once over
% the aprime grid and broadcast over d (the EZ raw repeats it N_d times with repelem, which copies
% values). So V is bit-identical to the EZ raw. TIE-BREAKING IS THE ONE DIFFERENCE in Policy, as in the
% vNM LSseed raw.
% The terminal age is, as in the EZ raw, the return composed with the warm-glow alone (no continuation),
% or with zero when there is no warm-glow.
%
% Every lowmemory branch of the EZ e raw is honoured (a seed is a full sweep, lowmemory is a memory option).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
end
if vfoptions.lowmemory>=3
    error('vfoptions.lowmemory=K not supported for ValueFnIter_FHorz_TPath_SingleStep_EZ_LSseed_e_raw')
end

Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_z,N_e,N_j,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

%% Iterate backwards through j, starting from j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,:,N_j);

for reverse_j=0:N_j-1
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

    % If there is a warm-glow, evaluate the warmglowfn, (aprime,1)
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        WGmatrix(WGmatrixraw==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf
    end

    if jj==N_j
        % The terminal age has no continuation: the warm-glow enters INSIDE the ^ezc7 root, as the terminal
        % condition of the recursion (the main-loop composition with the EV term absent), as in the EZ raw
        if warmglow==1
            becareful=(WGmatrix==0);
            WGmatrix(isfinite(WGmatrix))=((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j);
            WGmatrix(becareful)=0;
            temp4=WGmatrix; % [N_a,1], broadcast over (d,a,z,e) below
        else
            temp4=0; % the EZ raw adds ezc3*beta*0 at the terminal age
        end
    else
        VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,:,:,jj); % Grab this before it is replaced/updated
        EVpre=VKronNext_j;
        % Part of Epstein-Zin is before taking expectation
        temp=EVpre;
        temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
        temp(EVpre==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

        % Expectation over e
        temp=sum(temp.*pi_e_J(1,1,:,jj+1),3);

        if vfoptions.lowmemory==0 || vfoptions.lowmemory==1
            % Calc the expectation term (except beta)
            EV=temp.*shiftdim(pi_z_J(:,:,jj)',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', [N_a,1,N_z]

            temp4=EV;
            if warmglow==1
                WGmatrix=WGmatrix.*ones(1,1,N_z); % (aprime,1,z), matching temp4
                becareful=logical((~(temp4==-Inf)).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(~(temp4==-Inf))=(sj(jj)*temp4(~(temp4==-Inf)).^ezc8(jj)).^ezc6(jj);
                temp4(EV==0)=0;
            end
        end
        % lowmemory==2 takes the expectation over z one z at a time, below, as the EZ raw does
    end

    if vfoptions.lowmemory==0

        % Refine=1 keeps d and aprime in separate dimensions, (d,aprime,a,z,e). temp4 depends on
        % (aprime,z), so it broadcasts as shiftdim(temp4,-1).
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        temp2=ReturnMatrix;
        temp2(becareful)=ReturnMatrix(becareful).^ezc2(jj);

        entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);

        entireRHS(isnan(entireRHS))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

        % Two maxima. The first is over aprime FOR EACH d.
        [Vd,offset1]=max(entireRHS,[],2);                        % [N_d,1,N_a,N_z,N_e]
        % No window here, so the argmax is already a grid index in [1,N_a]
        aprimeRefNew_j=reshape(offset1,[N_d,N_a,N_z,N_e]);

        [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_z,N_e]),[],1);   % [1,N_a,N_z,N_e]
        V(:,:,:,jj)=reshape(Vtemp,[N_a,N_z,N_e]);
        dstar=reshape(dstar,[N_a,N_z,N_e]);

        % aprime at the chosen d. The lookup must use dstar and not the state index. Reshaped because with
        % N_d=1 the source can be a vector, and a vector indexed by a vector takes the SOURCE's orientation.
        aprimestar=reshape(aprimeRefNew_j(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)+N_d*N_a*N_z*shiftdim((0:1:N_e-1),-1)),[N_a,N_z,N_e]); % [N_a,N_z,N_e]

        % The folded (d,aprime) index, d fastest, the same form as the folded raw's Policy
        Policy(:,:,:,jj)=dstar+N_d*(aprimestar-1);
        aprimeReferencePolicyNew(:,:,:,:,jj)=aprimeRefNew_j;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, d_gridvals, a_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);
            % (d,aprime,a,z)

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=isfinite(ReturnMatrix_e); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
            temp2=ReturnMatrix_e;
            temp2(becareful)=ReturnMatrix_e(becareful).^ezc2(jj);

            entireRHS_e=ezc1*temp2+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1);

            entireRHS_e(isnan(entireRHS_e))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(entireRHS_e==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            entireRHS_e(temp5)=entireRHS_e(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

            [Vd,offset1]=max(entireRHS_e,[],2);                      % [N_d,1,N_a,N_z]
            aprimeRefNew_e=reshape(offset1,[N_d,N_a,N_z]);

            [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_z]),[],1);       % [1,N_a,N_z]
            V(:,:,e_c,jj)=reshape(Vtemp,[N_a,N_z]);
            dstar=reshape(dstar,[N_a,N_z]);

            aprimestar=reshape(aprimeRefNew_e(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)),[N_a,N_z]); % [N_a,N_z]

            Policy(:,:,e_c,jj)=dstar+N_d*(aprimestar-1);
            aprimeReferencePolicyNew(:,:,:,e_c,jj)=aprimeRefNew_e;
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);

            if jj==N_j
                temp4_z=temp4; % the terminal composition, the same for every z
            else
                % Calc the expectation term (except beta)
                EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2); % sum over z', [N_a,1]

                temp4_z=EV_z;
                if warmglow==1
                    becareful=logical((~(temp4_z==-Inf)).*isfinite(WGmatrix)); % both are finite
                    temp4_z(becareful)=(sj(jj)*temp4_z(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4_z((EV_z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4_z(~(temp4_z==-Inf))=(sj(jj)*temp4_z(~(temp4_z==-Inf)).^ezc8(jj)).^ezc6(jj);
                    temp4_z(EV_z==0)=0;
                end
            end

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,1);
                % (d,aprime,a)

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=isfinite(ReturnMatrix_ze); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
                temp2=ReturnMatrix_ze;
                temp2(becareful)=ReturnMatrix_ze(becareful).^ezc2(jj);

                entireRHS_ze=ezc1*temp2+ezc3*DiscountFactorParamsVec*shiftdim(temp4_z,-1);

                entireRHS_ze(isnan(entireRHS_ze))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
                temp5=~(entireRHS_ze==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
                entireRHS_ze(temp5)=entireRHS_ze(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity

                [Vd,offset1]=max(entireRHS_ze,[],2);                 % [N_d,1,N_a]
                aprimeRefNew_ze=reshape(offset1,[N_d,N_a]);

                [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a]),[],1);       % [1,N_a]
                V(:,z_c,e_c,jj)=Vtemp';
                dstar=dstar';

                aprimestar=reshape(aprimeRefNew_ze(dstar+N_d*(0:1:N_a-1)'),[N_a,1]);  % [N_a,1]

                Policy(:,z_c,e_c,jj)=dstar+N_d*(aprimestar-1);
                aprimeReferencePolicyNew(:,:,z_c,e_c,jj)=aprimeRefNew_ze;
            end
        end

    end
end

%% Output shape for policy, the same as the folded raw
Policy=shiftdim(Policy,-1);

end
