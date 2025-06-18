function [V,Policy3]=ValueFnIter_FHorz_SemiExo_GI_nod1_e_raw(n_d2,n_a,n_z,n_semiz, n_e,N_j, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy=zeros(3,N_a,N_semiz*N_z,N_j,'gpuArray'); % first dim indexes the optimal choice for d2, aprime and aprime2 (in GI layer)

%%
d2_grid=gpuArray(d2_grid);
a_grid=gpuArray(a_grid);

special_n_d2=ones(1,length(n_d2));
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_z)+length(n_semiz));
end

aind=0:1:N_a-1; % already includes -1
zind=shiftdim((0:1:N_z-1),-1); % already includes -1
eind=shiftdim((0:1:N_e-1),-2); % already includes -1

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d2, n_a, n_bothz, n_e, d2_grid, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % Treat standard problem as just being the first layer
        [~,maxindex]=max(ReturnMatrix,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_z-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d2,n_bothz,n_e,d2_gridvals,aprime_grid(aprimeindexes),a_grid,bothz_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*zind+N_d2*N_a*N_bothz*eind; % midpoint is n_d-by-1-by-n_a-by-n_z-by-n_e
        Policy(1,:,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d2, n_a, n_bothz, special_n_e, d2_grid, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(ReturnMatrix_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a-by-n_z
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d2,n_bothz,special_n_e,d2_gridvals,aprime_grid(aprimeindexes),a_grid,bothz_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*zind; % midpoint is n_d-by-1-by-n_a-by-n_z
            Policy(1,:,:,e_c,N_j)=d_ind; % d2
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,:,e_c,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz*N_z
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d2, n_a, special_n_bothz, special_n_e, d2_grid, a_grid, z_val, e_val, ReturnFnParamsVec,1);
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(ReturnMatrix_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d2,special_n_bothz,special_n_e,d2_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind; % midpoint is n_d-by-1-by-n_a
                Policy(1,:,z_c,e_c,N_j)=d_ind; % d2
                Policy(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
                Policy(3,:,z_c,e_c,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
            end
        end

    end
else
    % Using V_Jplus1
    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=sum(EV.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            EV=EV.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV,aprime_grid);

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, special_n_d2, n_a, n_bothz, n_e, d2_gridvals(d2_c,:)', a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*EV;
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS,[],1); % no d1, loop over d2

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-n_bothz-by-n_e
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are
            % n2long-by-n_a-by-n_bothz-by-n_e
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, n_e, d2_gridvals(d2_c,:), aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_bothz-1),-2); % the current aprime
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);
            
            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind+N_d1*N_a*N_bothz*eind; % loweredge is n_d-by-1-by-n_a-by-n_bothz-by-n_e
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoint(allind));

        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy(2,:,:,:,N_j)=aprimeL2_ind; % aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
            d2_val=d2_gridvals(d2_c,:)';

            EV=EV.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, special_n_d2, n_a, n_bothz, special_n_e, d2_val, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
                entireRHS_d2e=ReturnMatrix_d2e+DiscountFactorParamsVec*EV;
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(entireRHS_d2e,[],1); % no d1, loop over d2

                % Turn maxindex into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is 1-by-n_a-by-n_bothz
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
                % aprime possibilities are n2long-by-n_a-by-n_bothz
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_gridvals(d2_c,:), aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_bothz-1),-2); % the current aprime
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_bothz]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind; % loweredge is n_d-by-1-by-n_a-by-n_bothz
                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoint(allind));
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy(2,:,:,:,N_j)=aprimeL2_ind; % aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        
    elseif vfoptions.lowmemory==2
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));
            d2_val=d2_gridvals(d2_c,:)';

            for z_c=1:N_semiz*N_z
                z_val=bothz_gridvals_J(z_c,:,N_j);

                %Calc the condl expectation term (except beta) which depends on z but not control variables
                EV_z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                % Interpolate EV over aprime_grid
                EVinterp_z=interp1(a_grid,EV_z,aprime_grid);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, special_n_d2, n_a, special_n_bothz, special_n_e, d2_val, a_grid, z_val, e_val, ReturnFnParamsVec);
                    entireRHS_d2ze=ReturnMatrix_d2ze+DiscountFactorParamsVec*EV_z;
                    % Treat standard problem as just being the first layer
                    [~,maxindex]=max(entireRHS_d2ze,[],1); % no d1, loop over d2

                    % Turn maxindex into the 'midpoint'
                    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                    % midpoint is 1-by-n_a
                    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
                    % aprime possibilities are n2long-by-n_a
                    ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_gridvals(d2_c,:), aprime_grid(aprimeindexes), a_grid, z_val, e_val, ReturnFnParamsVec,2);
                    % aprimez=aprimeindexes; % the current aprime
                    entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
                    [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                    V_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(maxindex,1);

                    d1_ind=rem(maxindex-1,N_d1)+1;
                    allind=d1_ind+N_d1*aind; % loweredge is n_d-by-1-by-n_a
                    midpoint_ford2_jj(:,z_c,e_c,d2_c)=squeeze(midpoint(allind));
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy(2,:,:,:,N_j)=aprimeL2_ind; % aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
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
    
    EV=V(:,:,:,jj+1);
        
    EV=sum(EV.*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            EV=EV.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV,aprime_grid);

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, special_n_d2, n_a, n_bothz, n_e, d2_gridvals(d2_c,:)', a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);
            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*EV;
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS,[],1); % no d1, loop over d2

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-n_bothz-by-n_e
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are
            % n2long-by-n_a-by-n_bothz-by-n_e
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, n_e, d2_gridvals(d2_c,:), aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_bothz-1),-2); % the current aprime
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);
            
            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind+N_d1*N_a*N_bothz*eind; % loweredge is n_d-by-1-by-n_a-by-n_bothz-by-n_e
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoint(allind));

        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy(2,:,:,:,jj)=aprimeL2_ind; % aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
            d2_val=d2_gridvals(d2_c,:)';

            EV=EV.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d2e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, special_n_d2, n_a, n_bothz, special_n_e, d2_val, a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
                entireRHS_d2e=ReturnMatrix_d2e+DiscountFactorParamsVec*EV;
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(entireRHS_d2e,[],1); % no d1, loop over d2

                % Turn maxindex into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is 1-by-n_a-by-n_bothz
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
                % aprime possibilities are n2long-by-n_a-by-n_bothz
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_bothz, special_n_e, d2_gridvals(d2_c,:), aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_bothz-1),-2); % the current aprime
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_bothz]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind; % loweredge is n_d-by-1-by-n_a-by-n_bothz
                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoint(allind));
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy(2,:,:,:,jj)=aprimeL2_ind; % aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        
    elseif vfoptions.lowmemory==2
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));
            d2_val=d2_gridvals(d2_c,:)';

            for z_c=1:N_semiz*N_z
                z_val=bothz_gridvals_J(z_c,:,jj);

                %Calc the condl expectation term (except beta) which depends on z but not control variables
                EV_z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                % Interpolate EV over aprime_grid
                EVinterp_z=interp1(a_grid,EV_z,aprime_grid);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, special_n_d2, n_a, special_n_bothz, special_n_e, d2_val, a_grid, z_val, e_val, ReturnFnParamsVec);
                    entireRHS_d2ze=ReturnMatrix_d2ze+DiscountFactorParamsVec*EV_z;
                    % Treat standard problem as just being the first layer
                    [~,maxindex]=max(entireRHS_d2ze,[],1); % no d1, loop over d2

                    % Turn maxindex into the 'midpoint'
                    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                    % midpoint is 1-by-n_a
                    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
                    % aprime possibilities are n2long-by-n_a
                    ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, special_n_bothz, special_n_e, d2_gridvals(d2_c,:), aprime_grid(aprimeindexes), a_grid, z_val, e_val, ReturnFnParamsVec,2);
                    % aprimez=aprimeindexes; % the current aprime
                    entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
                    [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                    V_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(maxindex,1);

                    d1_ind=rem(maxindex-1,N_d1)+1;
                    allind=d1_ind+N_d1*aind; % loweredge is n_d-by-1-by-n_a
                    midpoint_ford2_jj(:,z_c,e_c,d2_c)=squeeze(midpoint(allind));
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
        Policy(2,:,:,:,jj)=aprimeL2_ind; % aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
    end
end


%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=squeeze(Policy(1,:,:,:,:)+N_d2*(Policy(2,:,:,:,:)-1)+N_d2*N_a*(Policy(3,:,:,:,:)-1));



end
