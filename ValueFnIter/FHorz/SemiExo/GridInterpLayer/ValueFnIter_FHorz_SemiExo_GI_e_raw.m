function [V,Policy]=ValueFnIter_FHorz_SemiExo_GI_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, n_e,N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

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
Policy4=zeros(4,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% First dimension: d1, d2, aprime, aprime2

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a_grid=gpuArray(a_grid);

special_n_d2=ones(1,length(n_d2));
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
d_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
else
    eind=shiftdim((0:1:N_e-1),-2); % already includes -1
end
% if vfoptions.lowmemory>1
%     special_n_bothz=ones(1,length(n_z)+length(n_semiz));
% end
aind=0:1:N_a-1; % already includes -1
bothzind=shiftdim((0:1:N_bothz-1),-1); % already includes -1

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
if vfoptions.lowmemory==0
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
    midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
elseif vfoptions.lowmemory==2 % loops over e and z
    V_ford2_jj=zeros(N_a,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_d2,'gpuArray');
    midpoint_ford2_jj=zeros(N_a,N_d2,'gpuArray');
end

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
if length(n_a)>1
    error('can only do gridinterplayer with one endo state (you have length(n_a)>1)')
end
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
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,n_d2], n_a, n_bothz, n_e, [d1_grid; d2_grid], a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        ReturnMatrix=reshape(ReturnMatrix,[N_d,N_a,N_a,N_bothz,N_e]);
        % Treat standard problem as just being the first layer
        [~,maxindex]=max(ReturnMatrix,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, [n_d1,n_d2], n_bothz, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*bothzind+N_d*N_a*N_bothz*eind; % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        Policy4(1,:,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy4(2,:,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy4(3,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy4(4,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        % for e_c=1:N_e
        %     e_val=e_gridvals_J(e_c,:,N_j);
        %     ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,n_d2], n_a, n_bothz, special_n_e, [d1_grid; d2_grid], a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
        %     % Calc the max and it's index
        %     [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
        %     V(:,:,e_c,N_j)=Vtemp;
        %     d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        %     Policy4(1,:,:,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        %     Policy4(2,:,:,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        %     Policy4(3,:,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        % end

    elseif vfoptions.lowmemory==2

        % for e_c=1:N_e
        %     e_val=e_gridvals_J(e_c,:,N_j);
        %     for z_c=1:N_semiz*N_z
        %         z_val=bothz_gridvals_J(z_c,:,N_j);
        %         ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,n_d2], n_a, special_n_bothz, special_n_e, [d1_grid; d2_grid], a_grid, z_val, e_val, ReturnFnParamsVec);
        %         % Calc the max and it's index
        %         [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
        %         V(:,z_c,e_c,N_j)=Vtemp;
        %         d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        %         Policy4(1,:,z_c,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        %         Policy4(2,:,z_c,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        %         Policy4(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        %     end
        % end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    V_Jplus1=sum(V_Jplus1.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        % for d2_c=1:N_d2
        %     % Note: By definition V_Jplus1 does not depend on d (only aprime)
        %     pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
        % 
        %     ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, n_bothz, n_e, [d1_grid; d2_gridvals(d2_c,:)'], a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %     % (d,aprime,a,z,e)
        % 
        %     EV=V_Jplus1.*shiftdim(pi_bothz',-1);
        %     EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        %     EV=sum(EV,2); % sum over z', leaving a singular second dimension
        % 
        %     entireEV=repelem(EV,N_d1,1,1);
        %     entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*entireEV; %*repmat(entireEV,1,N_a,1,N_e);
        % 
        %     % Calc the max and it's index
        %     [Vtemp,maxindex]=max(entireRHS,[],1);
        % 
        %     V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
        %     Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);
        % end
        % % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        % [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        % V(:,:,:,N_j)=V_jj;
        % Policy4(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        % maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        % d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[N_a,N_semiz*N_z*N_e]);
        % Policy4(1,:,:,:,N_j)=rem(d1aprime_ind-1,N_d1)+1;
        % Policy4(3,:,:,:,N_j)=ceil(d1aprime_ind/N_d1);

    elseif vfoptions.lowmemory==1
        % for d2_c=1:N_d2
        %     % Note: By definition V_Jplus1 does not depend on d (only aprime)
        %     pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
        %     d2_val=d2_gridvals(d2_c,:)';
        % 
        %     EV=V_Jplus1.*shiftdim(pi_bothz',-1);
        %     EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        %     EV=sum(EV,2); % sum over z', leaving a singular second dimension
        % 
        %     entireEV=repelem(EV,N_d1,1,1);
        % 
        %     for e_c=1:N_e
        %         e_val=e_gridvals_J(e_c,:,N_j);
        %         ReturnMatrix_d2e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, n_bothz, special_n_e, [d1_grid;d2_val], a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
        %         % (d,aprime,a,z)
        % 
        %         entireRHS_d2e=ReturnMatrix_d2e+DiscountFactorParamsVec*entireEV; %.*ones(1,N_a,1);
        % 
        %         % Calc the max and it's index
        %         [Vtemp,maxindex]=max(entireRHS_d2e,[],1);
        % 
        %         V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
        %         Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);
        %     end
        % end
        % % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        % [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        % V(:,:,:,N_j)=V_jj;
        % Policy4(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        % maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        % d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        % Policy4(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        % Policy4(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

    elseif vfoptions.lowmemory==2
        % for d2_c=1:N_d2
        %     % Note: By definition V_Jplus1 does not depend on d (only aprime)
        %     pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));
        %     d2_val=d2_gridvals(d2_c,:)';
        % 
        %     for z_c=1:N_semiz*N_z
        %         z_val=bothz_gridvals_J(z_c,:,N_j);
        % 
        %         %Calc the condl expectation term (except beta) which depends on z but not control variables
        %         EV_z=V_Jplus1.*shiftdim(pi_bothz(z_c,:)',-1);
        %         EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        %         EV_z=sum(EV_z,2);
        %         entireEV_z=kron(EV_z,ones(N_d1,1));
        % 
        %         for e_c=1:N_e
        %             e_val=e_gridvals_J(e_c,:,N_j);
        % 
        %             ReturnMatrix_d2ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, special_n_bothz, special_n_e, [d1_grid;d2_val], a_grid, z_val, e_val, ReturnFnParamsVec);
        % 
        %             entireRHS_d2ze=ReturnMatrix_d2ze+DiscountFactorParamsVec*entireEV_z; %*ones(1,N_a,1);
        % 
        %             %Calc the max and it's index
        %             [Vtemp,maxindex]=max(entireRHS_d2ze,[],1);
        % 
        %             V_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(Vtemp,1);
        %             Policy_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(maxindex,1);
        %         end
        %     end
        % end
        % % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        % [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        % V(:,:,:,N_j)=V_jj;
        % Policy4(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        % maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        % d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        % Policy4(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        % Policy4(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
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
    
    VKronNext_j=V(:,:,:,jj+1);
        
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0

        for d2_c=1:N_d2
            d12_gridvals=[d1_grid, d2_grid(d2_c)*ones(n_d1,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, n_bothz, n_e, [d1_grid; d2_gridvals(d2_c,:)'], a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);
            ReturnMatrix_d2=reshape(ReturnMatrix_d2,[N_d1,N_a,N_a,N_bothz,N_e]);
            % (d,aprime,a,z,e)

            EV=VKronNext_j.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*reshape(EV,[1,N_a,1,N_bothz]);

            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS,[],2); % second dimension, so this is the optimal aprime (on first layer, so on a_grid)

            % Now do the second layer for the interpolation

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV,aprime_grid); % CHECK RUNTIME!!!
            entireEVinterp=repelem(EVinterp,N_d1,1,1); % Note, this is only for later as it is the interpolated version
            % Note:  entireEVinterp is size [N_d1*n2aprime,1,N_semiz*N_z]

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, [n_d1,special_n_d2], n_bothz, n_e, d12_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            daprimez=(1:1:N_d1)'+N_d1*(aprimeindexes-1)+N_d1*n2aprime*shiftdim((0:1:N_bothz-1),-2); % the current aprime
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d1*n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);
            
            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind+N_d1*N_a*N_bothz*eind; % loweredge is n_d-by-1-by-n_a-by-n_bothz-by-n_e
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoint(allind));
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy4(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy4(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy4(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy4(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint

    elseif vfoptions.lowmemory==1
        % for d2_c=1:N_d2
        %     % Note: By definition V_Jplus1 does not depend on d (only aprime)
        %     pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
        % 
        %     EV=VKronNext_j.*shiftdim(pi_bothz',-1);
        %     EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        %     EV=sum(EV,2); % sum over z', leaving a singular second dimension
        % 
        %     entireEV=repelem(EV,N_d1,1,1);
        % 
        %     d2_val=d2_gridvals(d2_c,:)';
        % 
        %     for e_c=1:N_e
        %         e_val=e_gridvals_J(e_c,:,jj);
        %         ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, n_bothz, special_n_e, [d1_grid;d2_val], a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
        %         % (d,aprime,a,z)
        % 
        %         entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV; %.*ones(1,N_a,1);
        % 
        %         % Calc the max and it's index
        %         [Vtemp,maxindex]=max(entireRHS_e,[],1);
        % 
        %         V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
        %         Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);
        %     end
        % end
        % % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        % [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        % V(:,:,:,jj)=V_jj;
        % Policy4(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        % maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        % d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        % Policy4(1,:,:,:,jj)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        % Policy4(3,:,:,:,jj)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);
        
    elseif vfoptions.lowmemory==2
    %     for d2_c=1:N_d2
    %         % Note: By definition V_Jplus1 does not depend on d (only aprime)
    %         pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));
    % 
    %         d2_val=d2_gridvals(d2_c,:)';
    % 
    %         for z_c=1:N_bothz
    %             z_val=bothz_gridvals_J(z_c,:,jj);
    % 
    %             %Calc the condl expectation term (except beta) which depends on z but not control variables
    %             EV_z=VKronNext_j.*shiftdim(pi_bothz(z_c,:)',-1);
    %             EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    %             EV_z=sum(EV_z,2);
    %             entireEV_z=kron(EV_z,ones(N_d1,1));
    % 
    %             for e_c=1:N_e
    %                 e_val=e_gridvals_J(e_c,:,jj);
    % 
    %                 ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, special_n_bothz, special_n_e, [d1_grid;d2_val], a_grid, z_val, e_val, ReturnFnParamsVec);
    % 
    %                 entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*entireEV_z; %*ones(1,N_a,1);
    % 
    %                 %Calc the max and it's index
    %                 [Vtemp,maxindex]=max(entireRHS_ze,[],1);
    % 
    %                 V_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(Vtemp,1);
    %                 Policy_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(maxindex,1);
    %             end
    %         end
    %     end
    %     % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    %     [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
    %     V(:,:,:,jj)=V_jj;
    %     Policy4(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
    %     maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    %     d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
    %     Policy4(1,:,:,:,jj)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
    %     Policy4(3,:,:,:,jj)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);
    end

end

% Currently Policy(3,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(3,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy4(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy4(3,:,:,:,:)=Policy4(3,:,:,:,:)-adjust; % lower grid point
Policy4(4,:,:,:,:)=adjust.*Policy4(4,:,:,:,:)+(1-adjust).*(Policy4(4,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=squeeze(Policy4(1,:,:,:,:)+N_d1*(Policy4(2,:,:,:,:)-1)+N_d*(Policy4(3,:,:,:,:)-1)+N_d*N_a*(Policy4(4,:,:,:,:)-1));




end