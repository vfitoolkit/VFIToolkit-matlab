function [V,Policy3]=ValueFnIter_FHorz_EpsteinZin_SemiExo_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, n_e,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Epstein-Zin version of ValueFnIter_FHorz_SemiExo_e_raw.
% The certainty-equivalent is taken over the JOINT distribution of
% (semizprime,zprime,eprime), which depends on the chosen d2: V' is
% transformed by ^ezc5 once per age (d2-independent, pointwise), the
% eprime-expectation (also d2-independent) is taken on the transformed object,
% then the d2-dependent (semizprime,zprime)-expectation, then the
% certainty-equivalent power ^ezc6 and the rest of the transform chain are
% applied before each conditional-on-d2 max (the d1-expansion of the
% expectation term happens after the transform chain, which is pointwise). The
% final max over d2 then compares fully-transformed values, so it is
% unaffected.

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod([n_d1,n_d2]); % Needed at end to reshape output
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy3=zeros(3,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

special_n_d=[n_d1,ones(1,length(n_d2))];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]); % version to use when looping over d2

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_e=ones(1,length(n_e));
end
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
if vfoptions.lowmemory==0
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e (stored per-e, so the e dimension is needed)
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
elseif vfoptions.lowmemory==3 % joint bothz, inner e
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
end

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

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
    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrix(becareful)=0;
    end
    % WGmatrix is a column over aprime; in the joint-(d1,d2,aprime) terminal branch it needs the d dimension
else
    WGmatrix=zeros(N_a,1,'gpuArray');
end

if ~isfield(vfoptions,'V_Jplus1')
    WGmatrixkron=kron(WGmatrix,ones(N_d,1)); % rows of the terminal ReturnMatrix are (d1,d2,aprime) with d fastest

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_bothz, n_e, d_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite but not zero
        ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2(N_j)).^ezc7(N_j);
        ReturnMatrix(ReturnMatrix==0)=-Inf;
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix+WGmatrixkron,[],1);
        V(:,:,:,N_j)=Vtemp;
        d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        Policy3(1,:,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy3(2,:,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy3(3,:,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_bothz, special_n_e, d_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite but not zero
            ReturnMatrix_e(becareful)=(ezc1*ReturnMatrix_e(becareful).^ezc2(N_j)).^ezc7(N_j);
            ReturnMatrix_e(ReturnMatrix_e==0)=-Inf;
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e+WGmatrixkron,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
            Policy3(1,:,:,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy3(2,:,:,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy3(3,:,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        end

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz

        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, [n_semiz,special_n_z], special_n_e, d_gridvals, a_grid, z_valblock, e_val, ReturnFnParamsVec,0);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ze).*(ReturnMatrix_ze~=0)); % finite but not zero
                ReturnMatrix_ze(becareful)=(ezc1*ReturnMatrix_ze(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_ze(ReturnMatrix_ze==0)=-Inf;
                [Vtemp,maxindex]=max(ReturnMatrix_ze+WGmatrixkron,[],1);
                V(:,semizblock,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d)+1;
                Policy3(1,:,semizblock,e_c,N_j)=rem(d_ind-1,N_d1)+1;
                Policy3(2,:,semizblock,e_c,N_j)=ceil(d_ind/N_d1);
                Policy3(3,:,semizblock,e_c,N_j)=ceil(maxindex/N_d);
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e

        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_bothz, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ze).*(ReturnMatrix_ze~=0)); % finite but not zero
                ReturnMatrix_ze(becareful)=(ezc1*ReturnMatrix_ze(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_ze(ReturnMatrix_ze==0)=-Inf;
                [Vtemp,maxindex]=max(ReturnMatrix_ze+WGmatrixkron,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
                Policy3(1,:,z_c,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
                Policy3(2,:,z_c,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
                Policy3(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
            end
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    % Part of Epstein-Zin is before taking expectation (d2- and e-independent, so done once)
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0;

    % Expectation over e (on the transformed object; e-transition is d2-independent, so hoisted out of the d2 loop)
    temp=sum(temp.*pi_e_J(1,1,:,N_j+1),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
            % (d,aprime,a,z,e)
            becareful=logical(isfinite(ReturnMatrix_d2).*(ReturnMatrix_d2~=0)); % finite but not zero
            temp2=ReturnMatrix_d2;
            temp2(becareful)=ReturnMatrix_d2(becareful).^ezc2(N_j);
            temp2(ReturnMatrix_d2==0)=-Inf;

            EV_d2=temp.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,bothz)
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV_d2==0)=0;
            end

            entireEV=repelem(temp4,N_d1,1,1); % expand over d1 (after the transform chain, which is pointwise)
            entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*entireEV;

            temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
            entireRHS(temp5)=entireRHS(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS(entireRHS==0)=-Inf;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,N_j)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,N_j)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            EV_d2=temp.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,bothz); e-independent so done before the e loop
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV_d2==0)=0;
            end

            entireEV=repelem(temp4,N_d1,1,1); % expand over d1 (after the transform chain, which is pointwise)

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, special_n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
                % (d,aprime,a,z)
                becareful=logical(isfinite(ReturnMatrix_d2e).*(ReturnMatrix_d2e~=0)); % finite but not zero
                temp2=ReturnMatrix_d2e;
                temp2(becareful)=ReturnMatrix_d2e(becareful).^ezc2(N_j);
                temp2(ReturnMatrix_d2e==0)=-Inf;

                entireRHS_d2e=ezc1*temp2+ezc3*DiscountFactorParamsVec*entireEV;

                temp5=logical(isfinite(entireRHS_d2e).*(entireRHS_d2e~=0));
                entireRHS_d2e(temp5)=entireRHS_d2e(temp5).^ezc7(N_j);
                entireRHS_d2e(entireRHS_d2e==0)=-Inf;

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d2e,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,N_j)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,N_j)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                EV_d2z=temp.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % Certainty-equivalent (and mortality-risk/warm-glow) transform; e-independent so done before the e loop
                temp4=EV_d2z;
                if warmglow==1
                    WGmatrixbig=WGmatrix.*ones(1,1,N_semiz);
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_d2z==0)&(WGmatrixbig==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_d2z==0)=0;
                end

                entireEV=repelem(temp4,N_d1,1,1); % expand over d1 (after the transform chain, which is pointwise)

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, z_valblock, e_val, ReturnFnParamsVec,0);
                    becareful=logical(isfinite(ReturnMatrix_d2ze).*(ReturnMatrix_d2ze~=0)); % finite but not zero
                    temp2=ReturnMatrix_d2ze;
                    temp2(becareful)=ReturnMatrix_d2ze(becareful).^ezc2(N_j);
                    temp2(ReturnMatrix_d2ze==0)=-Inf;

                    entireRHS_ze=ezc1*temp2+ezc3*DiscountFactorParamsVec*entireEV;

                    temp5=logical(isfinite(entireRHS_ze).*(entireRHS_ze~=0));
                    entireRHS_ze(temp5)=entireRHS_ze(temp5).^ezc7(N_j);
                    entireRHS_ze(entireRHS_ze==0)=-Inf;

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(maxindex,1);
                end
            end
        end
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,N_j)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,N_j)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                EV_d2z=temp.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2);

                % Certainty-equivalent (and mortality-risk/warm-glow) transform; e-independent so done before the e loop
                temp4=EV_d2z;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_d2z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_d2z==0)=0;
                end

                entireEV=repelem(temp4,N_d1,1,1); % expand over d1 (after the transform chain, which is pointwise)

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, special_n_bothz, special_n_e, d12c_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                    becareful=logical(isfinite(ReturnMatrix_d2ze).*(ReturnMatrix_d2ze~=0)); % finite but not zero
                    temp2=ReturnMatrix_d2ze;
                    temp2(becareful)=ReturnMatrix_d2ze(becareful).^ezc2(N_j);
                    temp2(ReturnMatrix_d2ze==0)=-Inf;

                    entireRHS_ze=ezc1*temp2+ezc3*DiscountFactorParamsVec*entireEV;

                    temp5=logical(isfinite(entireRHS_ze).*(entireRHS_ze~=0));
                    entireRHS_ze(temp5)=entireRHS_ze(temp5).^ezc7(N_j);
                    entireRHS_ze(entireRHS_ze==0)=-Inf;

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford2_jj(:,z_c,e_c,d2_c)=Vtemp;
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=maxindex;
                end
            end
        end
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,N_j)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,N_j)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);
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
        % WGmatrix is a column over aprime; combined into temp4 below
    end

    EVpre=V(:,:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation (d2- and e-independent, so done once)
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    % Expectation over e (on the transformed object; e-transition is d2-independent, so hoisted out of the d2 loop)
    temp=sum(temp.*pi_e_J(1,1,:,jj+1),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);
            % (d,aprime,a,z,e)
            becareful=logical(isfinite(ReturnMatrix_d2).*(ReturnMatrix_d2~=0)); % finite but not zero
            temp2=ReturnMatrix_d2;
            temp2(becareful)=ReturnMatrix_d2(becareful).^ezc2(jj);
            temp2(ReturnMatrix_d2==0)=-Inf;

            EV_d2=temp.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,bothz)
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV_d2==0)=0;
            end

            entireEV=repelem(temp4,N_d1,1,1); % expand over d1 (after the transform chain, which is pointwise)
            entireRHS=ezc1*temp2+ezc3*DiscountFactorParamsVec*entireEV;

            temp5=logical(isfinite(entireRHS).*(entireRHS~=0));
            entireRHS(temp5)=entireRHS(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS(entireRHS==0)=-Inf;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);
        end

        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,jj)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,jj)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            EV_d2=temp.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,bothz); e-independent so done before the e loop
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_bothz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV_d2==0)=0;
            end

            entireEV=repelem(temp4,N_d1,1,1); % expand over d1 (after the transform chain, which is pointwise)

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, special_n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0);
                % (d,aprime,a,z)
                becareful=logical(isfinite(ReturnMatrix_e).*(ReturnMatrix_e~=0)); % finite but not zero
                temp2=ReturnMatrix_e;
                temp2(becareful)=ReturnMatrix_e(becareful).^ezc2(jj);
                temp2(ReturnMatrix_e==0)=-Inf;

                entireRHS_e=ezc1*temp2+ezc3*DiscountFactorParamsVec*entireEV;

                temp5=logical(isfinite(entireRHS_e).*(entireRHS_e~=0));
                entireRHS_e(temp5)=entireRHS_e(temp5).^ezc7(jj);
                entireRHS_e(entireRHS_e==0)=-Inf;

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_e,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,jj)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,jj)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                EV_d2z=temp.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % Certainty-equivalent (and mortality-risk/warm-glow) transform; e-independent so done before the e loop
                temp4=EV_d2z;
                if warmglow==1
                    WGmatrixbig=WGmatrix.*ones(1,1,N_semiz);
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_d2z==0)&(WGmatrixbig==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_d2z==0)=0;
                end

                entireEV=repelem(temp4,N_d1,1,1); % expand over d1 (after the transform chain, which is pointwise)

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, z_valblock, e_val, ReturnFnParamsVec,0);
                    becareful=logical(isfinite(ReturnMatrix_d2ze).*(ReturnMatrix_d2ze~=0)); % finite but not zero
                    temp2=ReturnMatrix_d2ze;
                    temp2(becareful)=ReturnMatrix_d2ze(becareful).^ezc2(jj);
                    temp2(ReturnMatrix_d2ze==0)=-Inf;

                    entireRHS_ze=ezc1*temp2+ezc3*DiscountFactorParamsVec*entireEV;

                    temp5=logical(isfinite(entireRHS_ze).*(entireRHS_ze~=0));
                    entireRHS_ze(temp5)=entireRHS_ze(temp5).^ezc7(jj);
                    entireRHS_ze(entireRHS_ze==0)=-Inf;

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(maxindex,1);
                end
            end
        end
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,jj)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,jj)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                EV_d2z=temp.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2);

                % Certainty-equivalent (and mortality-risk/warm-glow) transform; e-independent so done before the e loop
                temp4=EV_d2z;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_d2z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_d2z==0)=0;
                end

                entireEV=repelem(temp4,N_d1,1,1); % expand over d1 (after the transform chain, which is pointwise)

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, special_n_bothz, special_n_e, d12c_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                    becareful=logical(isfinite(ReturnMatrix_d2ze).*(ReturnMatrix_d2ze~=0)); % finite but not zero
                    temp2=ReturnMatrix_d2ze;
                    temp2(becareful)=ReturnMatrix_d2ze(becareful).^ezc2(jj);
                    temp2(ReturnMatrix_d2ze==0)=-Inf;

                    entireRHS_ze=ezc1*temp2+ezc3*DiscountFactorParamsVec*entireEV;

                    temp5=logical(isfinite(entireRHS_ze).*(entireRHS_ze~=0));
                    entireRHS_ze(temp5)=entireRHS_ze(temp5).^ezc7(jj);
                    entireRHS_ze(entireRHS_ze==0)=-Inf;

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford2_jj(:,z_c,e_c,d2_c)=Vtemp;
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=maxindex;
                end
            end
        end
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,jj)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,jj)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);
    end
end



end
