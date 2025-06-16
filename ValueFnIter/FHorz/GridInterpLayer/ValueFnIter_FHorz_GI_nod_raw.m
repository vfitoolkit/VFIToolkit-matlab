function [V, Policy]=ValueFnIter_FHorz_GI_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(2,N_a,N_z,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)

%%
a_grid=gpuArray(a_grid);

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

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
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        %Calc the max and it's index
        [~,maxindex]=max(ReturnMatrix,[],1);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a-by-n_bothz-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=shiftdim(squeeze(midpoint),-1); % midpoint
        Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);
            %Calc the max and it's index
            [~,maxindex]=max(ReturnMatrix,[],1);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-n_bothz-by-n_e
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,N_j)=shiftdim(squeeze(midpoint),-1); % midpoint
            Policy(2,:,z_c,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0

        % Use sparse for a few lines until sum over zprime
        EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

        %Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],1);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a-by-n_z
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-1);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=shiftdim(squeeze(midpoint),-1); % midpoint
        Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);

            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=V_Jplus1.*pi_z_J(z_c,:,N_j);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            % Interpolate EV over aprime_grid
            EVinterp_z=interp1(a_grid,EV_z,aprime_grid);

            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z;

            %Calc the max and it's index
            [~,maxindex]=max(entireRHS_z,[],1);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-n_z
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
            ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountFactorParamsVec*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,N_j)=shiftdim(squeeze(midpoint),-1); % midpoint
            Policy(2,:,z_c,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
        end
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    VKronNext_j=V(:,:,jj+1);
    
    if vfoptions.lowmemory==0

        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0);
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV;

        %Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],1);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a-by-n_z
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*shiftdim((0:1:N_z-1),-1);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,jj)=shiftdim(squeeze(midpoint),-1); % midpoint
        Policy(2,:,:,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:);
            
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=VKronNext_j.*pi_z_J(z_c,:,jj);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            % Interpolate EV over aprime_grid
            EVinterp_z=interp1(a_grid,EV_z,aprime_grid);

            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z;

            %Calc the max and it's index
            [~,maxindex]=max(entireRHS_z,[],1);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-n_z
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
            ReturnMatrix_ii_z=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountFactorParamsVec*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,N_j)=shiftdim(squeeze(midpoint),-1); % midpoint
            Policy(2,:,z_c,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
        end
    end
end

% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust; % lower grid point
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=Policy(1,:,:,:)+N_a*(Policy(2,:,:,:)-1);


end