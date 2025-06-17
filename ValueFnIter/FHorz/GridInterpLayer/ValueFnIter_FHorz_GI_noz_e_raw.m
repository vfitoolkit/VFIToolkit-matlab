function [V,Policy]=ValueFnIter_FHorz_GI_noz_e_raw(n_d,n_a,n_e,N_j, d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_e,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
d_gridvals=CreateGridvals(n_d,d_grid,1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end

aind=0:1:N_a-1; % already includes -1
eind=shiftdim((0:1:N_e-1),-1); % already includes -1

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
% n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_grid, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec);  % Because no z, can treat e like z and call Par2 rather than Par2e
        % Calc the max and it's index
        [~,maxindex]=max(ReturnMatrix,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*eind; % midpoint is n_d-by-1-by-n_a-by-n_e
        Policy(1,:,:,N_j)=d_ind; % d
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);  % Because no z, can treat e like z and call Par2 rather than Par2e
            % Calc the max and it's index
            [~,maxindex]=max(ReturnMatrix_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,e_c,N_j)=d_ind; % d
            Policy(2,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,e_c,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=sum(V_Jplus1.*pi_e_J(1,:,N_j),2);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_grid, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec);  % Because no z, can treat e like z and call Par2 rather than Par2e
        % (d,aprime,a,e)
        
        entireEV=repelem(EV,N_d,1,1);

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);
        entireEVinterp=repelem(EVinterp,N_d,1,1);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*entireEV;

        %Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);
        
        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        daprime=(1:1:N_d)'+N_d*(aprimeindexes-1);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprime(:)),[N_d*n2long,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*eind; % midpoint is n_d-by-1-by-n_a-by-n_e
        Policy(1,:,:,N_j)=d_ind; % d
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
        
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);
            % (d,aprime,a)
            
            entireEV=repelem(EV,N_d,1,1);

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV,aprime_grid);
            entireEVinterp=repelem(EVinterp,N_d,1,1);

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV;

            %Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            daprime=(1:1:N_d)'+N_d*(aprimeindexes-1);
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountFactorParamsVec*reshape(entireEVinterp(daprime(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,e_c,N_j)=d_ind; % d
            Policy(2,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,e_c,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
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
            
    EV=sum(V(:,:,jj+1).*pi_e_J(1,:,jj),2);
    
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_grid, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec);

        entireEV=repelem(EV,N_d,1,1);

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid);
        entireEVinterp=repelem(EVinterp,N_d,1,1);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*entireEV;

        %Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);
        
        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        daprime=(1:1:N_d)'+N_d*(aprimeindexes-1);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprime(:)),[N_d*n2long,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*eind; % midpoint is n_d-by-1-by-n_a-by-n_e
        Policy(1,:,:,jj)=d_ind; % d
        Policy(2,:,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);

            entireEV=repelem(EV,N_d,1,1);

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV,aprime_grid);
            entireEVinterp=repelem(EVinterp,N_d,1,1);

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV;

            %Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            daprime=(1:1:N_d)'+N_d*(aprimeindexes-1);
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountFactorParamsVec*reshape(entireEVinterp(daprime(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            V(:,e_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,e_c,jj)=d_ind; % d
            Policy(2,:,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,e_c,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind
        end
     
    end

end

% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=Policy(1,:,:,:)+N_d*(Policy(2,:,:,:)-1)+N_d*N_a*(Policy(3,:,:,:)-1);


end
