function [V,Policy]=ValueFnIter_FHorz_GI_e_raw(n_d,n_a,n_z,n_e,N_j, d_grid, a_grid, z_gridvals_J, e_gridvals_J,pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);


V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
d_gridvals=CreateGridvals(n_d,d_grid,1);

% Preallocate
if vfoptions.lowmemory>0 % loops over e
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1 % loops over z,e
    special_n_z=ones(1,length(n_z));
end

aind=0:1:N_a-1; % already includes -1
zind=shiftdim((0:1:N_z-1),-1); % already includes -1
eind=shiftdim((0:1:N_e-1),-2); % already includes -1

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

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % Calc the max and it's index
        [~,maxindex]=max(ReturnMatrix,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_z-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind+N_d*N_a*N_z*eind; % midpoint is n_d-by-1-by-n_a-by-n_z-by-n_e
        Policy(1,:,:,:,N_j)=d_ind; % d
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            % Calc the max and it's index
            [~,maxindex]=max(ReturnMatrix_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a-by-n_z
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zind; % midpoint is n_d-by-1-by-n_a-by-n_z
            Policy(1,:,:,e_c,N_j)=d_ind; % d
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec,1);
                % Calc the max and it's index
                [~,maxindex]=max(ReturnMatrix_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
                Policy(1,:,z_c,e_c,N_j)=d_ind; % d
                Policy(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
                Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
            end
        end

    end
else
    % Using V_Jplus1
    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=sum(EV.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % (d,aprime,a,z,e)
        
        EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        % entireEV=repelem(EV,N_d,1,1);

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);
        entireEVinterp=repelem(EVinterp,N_d,1,1);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);

        %Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);
        
        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_z-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        daprimez=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*shiftdim((0:1:N_z-1),-2);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d*n2long,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind+N_d*N_a*N_z*eind; % midpoint is n_d-by-1-by-n_a-by-n_z-by-n_e
        Policy(1,:,:,:,N_j)=d_ind; % d
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
        EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        % entireEV=repelem(EV,N_d,1,1);
        
        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);
        entireEVinterp=repelem(EVinterp,N_d,1,1);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            % (d,aprime,a,z)

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*shiftdim(EV,-1);

            %Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a-by-n_z
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*shiftdim((0:1:N_z-1),-2);
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d*n2long,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zind; % midpoint is n_d-by-1-by-n_a-by-n_z
            Policy(1,:,:,e_c,N_j)=d_ind; % d
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            
            %Calc the condl expectation term (except beta) which depends on z but not control variables
            EV_z=EV.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,N_j));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            % entireEV_z=kron(EV_z,ones(N_d,1));
            % Interpolate EV over aprime_grid
            EVinterp_z=interp1(a_grid,EV_z,aprime_grid);
            entireEVinterp_z=repelem(EVinterp_z,N_d,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec,1);
                
                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*shiftdim(EV_z,-1);

                %Calc the max and it's index
                [~,maxindex]=max(entireRHS_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                daprime=(1:1:N_d)'+N_d*(aprimeindexes-1);
                entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountFactorParamsVec*reshape(entireEVinterp_z(daprime(:)),[N_d*n2long,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
                Policy(1,:,z_c,e_c,N_j)=d_ind; % d
                Policy(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
                Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
            end
        end
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
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        % (d,aprime,a,z,e)
        
        EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        % entireEV=repelem(EV,N_d,1,1);

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);
        entireEVinterp=repelem(EVinterp,N_d,1,1);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);

        %Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);
        
        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_z-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        daprimez=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*shiftdim((0:1:N_z-1),-2);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d*n2long,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind+N_d*N_a*N_z*eind; % midpoint is n_d-by-1-by-n_a-by-n_z-by-n_e
        Policy(1,:,:,:,jj)=d_ind; % d
        Policy(2,:,:,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
        EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        % entireEV=repelem(EV,N_d,1,1);
        
        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);
        entireEVinterp=repelem(EVinterp,N_d,1,1);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);
            % (d,aprime,a,z)

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*shiftdim(EV,-1);

            %Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a-by-n_z
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),e_val,ReturnFnParamsVec,2);
            daprimez=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*shiftdim((0:1:N_z-1),-2);
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d*n2long,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            V(:,:,e_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zind; % midpoint is n_d-by-1-by-n_a-by-n_z
            Policy(1,:,:,e_c,jj)=d_ind; % d
            Policy(2,:,:,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            
            %Calc the condl expectation term (except beta) which depends on z but not control variables
            EV_z=EV.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            % entireEV_z=kron(EV_z,ones(N_d,1));
            % Interpolate EV over aprime_grid
            EVinterp_z=interp1(a_grid,EV_z,aprime_grid);
            entireEVinterp_z=repelem(EVinterp_z,N_d,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec,1);
                
                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*shiftdim(EV_z,-1);

                %Calc the max and it's index
                [~,maxindex]=max(entireRHS_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d,special_n_z,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                daprime=(1:1:N_d)'+N_d*(aprimeindexes-1);
                entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountFactorParamsVec*reshape(entireEVinterp_z(daprime(:)),[N_d*n2long,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                V(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
                Policy(1,:,z_c,e_c,jj)=d_ind; % d
                Policy(2,:,z_c,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
                Policy(3,:,z_c,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
            end
        end
    end

end

% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=Policy(1,:,:,:,:)+N_d*(Policy(2,:,:,:,:)-1)+N_d*N_a*(Policy(3,:,:,:,:)-1);


end
