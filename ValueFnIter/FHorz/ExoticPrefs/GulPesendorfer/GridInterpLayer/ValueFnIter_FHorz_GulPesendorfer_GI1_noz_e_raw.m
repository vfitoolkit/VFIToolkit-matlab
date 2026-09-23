function [V,Policy]=ValueFnIter_FHorz_GulPesendorfer_GI1_noz_e_raw(n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Gul-Pesendorfer with the grid interpolation layer. The choice set is the fine grid, so the
% most-tempting term is the max of v over the FINE grid, found by the same two-stage scheme as
% the main max but around v's OWN coarse argmax (otherwise the chosen fine point could be more
% tempting than the coarse max of v, making the self-control cost negative). The main max is
% the standard GI two-stage on the tempted objective u+v+beta*EV, with the L2 -Inf flag based
% on u+v (the temptation fn is -Inf exactly where the return fn is, by user contract, but the
% sum makes the flag robust either way). The most-tempting term is subtracted after the max.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_e,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)
Policy(4,:,:,:)=2; % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
% When ReturnFn is -Inf on one of the course grid points, we will allow fine index between that and the neighbouring course grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iteration

%%
if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end

aind=gpuArray(0:1:N_a-1); % already includes -1
eind=shiftdim(gpuArray(0:1:N_e-1),-1); % already includes -1

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
% n2aprime=length(aprime_grid);

pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);  % Because no z, can treat e like z and call Par2 rather than Par2e
        TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);  % Because no z, can treat e like z and call Par2 rather than Par2e
        % Calc the max and it's index
        [~,maxindex]=max(ReturnMatrix+TemptationMatrix,[],2);

        % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a-1),2);
        aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexesT),a_grid,e_gridvals_J(:,:,N_j),TemptationFnParamsVec,2);
        MostTempting=max(TemptationMatrix_Tii,[],1);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),TemptationFnParamsVec,2);
        Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
        [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        d_ind = rem(maxindexL2-1,N_d)+1;
        L2offset = ceil(maxindexL2/N_d);
        linidx_lower = d_ind + N_d*n2long*aind + N_d*n2long*N_a*eind;
        linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*eind;
        isInfLower = (Ftemp_ii(linidx_lower) == -Inf);
        isInfUpper = (Ftemp_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(4,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V(:,:,N_j)=shiftdim(Vtempii-MostTempting,1);
        allind=d_ind+N_d*aind+N_d*N_a*eind; % midpoint is n_d-by-1-by-n_a-by-n_e
        Policy(1,:,:,N_j)=d_ind; % d
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,1);  % Because no z, can treat e like z and call Par2 rather than Par2e
            TemptationMatrix_e=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, TemptationFnParamsVec,1);  % Because no z, can treat e like z and call Par2 rather than Par2e
            % Calc the max and it's index
            [~,maxindex]=max(ReturnMatrix_e+TemptationMatrix_e,[],2);

            % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
            [~,maxindexT]=max(TemptationMatrix_e,[],2);
            midpointT=max(min(maxindexT,n_a-1),2);
            aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexesT),a_grid,e_val,TemptationFnParamsVec,2);
            MostTempting=max(TemptationMatrix_Tii,[],1);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            TemptationMatrix_ii_e=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,TemptationFnParamsVec,2);
            Ftemp_ii_e=ReturnMatrix_ii_e+TemptationMatrix_ii_e;
            [Vtempii,maxindexL2]=max(Ftemp_ii_e,[],1);

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            d_ind = rem(maxindexL2-1,N_d)+1;
            L2offset = ceil(maxindexL2/N_d);
            linidx_lower = d_ind + N_d*n2long*aind;
            linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
            isInfLower = (Ftemp_ii_e(linidx_lower) == -Inf);
            isInfUpper = (Ftemp_ii_e(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(4,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            V(:,e_c,N_j)=shiftdim(Vtempii-MostTempting,1);
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,e_c,N_j)=d_ind; % d
            Policy(2,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        end

    end
else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_e]).*pi_e_J(1,:,N_j+1),2);

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);  % Because no z, can treat e like z and call Par2 rather than Par2e
        TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);  % Because no z, can treat e like z and call Par2 rather than Par2e
        % (d,aprime,a,e)

        entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);

        %Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);

        % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a-1),2);
        aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexesT),a_grid,e_gridvals_J(:,:,N_j),TemptationFnParamsVec,2);
        MostTempting=max(TemptationMatrix_Tii,[],1);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),TemptationFnParamsVec,2);
        Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
        entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        d_ind = rem(maxindexL2-1,N_d)+1;
        L2offset = ceil(maxindexL2/N_d);
        linidx_lower = d_ind + N_d*n2long*aind + N_d*n2long*N_a*eind;
        linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*eind;
        isInfLower = (Ftemp_ii(linidx_lower) == -Inf);
        isInfUpper = (Ftemp_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(4,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V(:,:,N_j)=shiftdim(Vtempii-MostTempting,1);
        allind=d_ind+N_d*aind+N_d*N_a*eind; % midpoint is n_d-by-1-by-n_a-by-n_e
        Policy(1,:,:,N_j)=d_ind; % d
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,1);
            TemptationMatrix_e=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, TemptationFnParamsVec,1);
            % (d,aprime,a)

            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e+DiscountFactorParamsVec*shiftdim(EV,-1);

            %Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
            [~,maxindexT]=max(TemptationMatrix_e,[],2);
            midpointT=max(min(maxindexT,n_a-1),2);
            aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexesT),a_grid,e_val,TemptationFnParamsVec,2);
            MostTempting=max(TemptationMatrix_Tii,[],1);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            TemptationMatrix_ii_e=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,TemptationFnParamsVec,2);
            Ftemp_ii_e=ReturnMatrix_ii_e+TemptationMatrix_ii_e;
            entireRHS_ii_e=Ftemp_ii_e+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            d_ind = rem(maxindexL2-1,N_d)+1;
            L2offset = ceil(maxindexL2/N_d);
            linidx_lower = d_ind + N_d*n2long*aind;
            linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
            isInfLower = (Ftemp_ii_e(linidx_lower) == -Inf);
            isInfUpper = (Ftemp_ii_e(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(4,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            V(:,e_c,N_j)=shiftdim(Vtempii-MostTempting,1);
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,e_c,N_j)=d_ind; % d
            Policy(2,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
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
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=sum(V(:,:,jj+1).*pi_e_J(1,:,jj+1),2);

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,jj), TemptationFnParamsVec,1);

        entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);

        %Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);

        % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a-1),2);
        aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexesT),a_grid,e_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
        MostTempting=max(TemptationMatrix_Tii,[],1);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,jj),TemptationFnParamsVec,2);
        Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
        entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        d_ind = rem(maxindexL2-1,N_d)+1;
        L2offset = ceil(maxindexL2/N_d);
        linidx_lower = d_ind + N_d*n2long*aind + N_d*n2long*N_a*eind;
        linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*eind;
        isInfLower = (Ftemp_ii(linidx_lower) == -Inf);
        isInfUpper = (Ftemp_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(4,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V(:,:,jj)=shiftdim(Vtempii-MostTempting,1);
        allind=d_ind+N_d*aind+N_d*N_a*eind; % midpoint is n_d-by-1-by-n_a-by-n_e
        Policy(1,:,:,jj)=d_ind; % d
        Policy(2,:,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(3,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,1);
            TemptationMatrix_e=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, TemptationFnParamsVec,1);

            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e+DiscountFactorParamsVec*shiftdim(EV,-1);

            %Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
            [~,maxindexT]=max(TemptationMatrix_e,[],2);
            midpointT=max(min(maxindexT,n_a-1),2);
            aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexesT),a_grid,e_val,TemptationFnParamsVec,2);
            MostTempting=max(TemptationMatrix_Tii,[],1);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            TemptationMatrix_ii_e=CreateReturnFnMatrix_Disc_DC1(TemptationFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,TemptationFnParamsVec,2);
            Ftemp_ii_e=ReturnMatrix_ii_e+TemptationMatrix_ii_e;
            entireRHS_ii_e=Ftemp_ii_e+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            d_ind = rem(maxindexL2-1,N_d)+1;
            L2offset = ceil(maxindexL2/N_d);
            linidx_lower = d_ind + N_d*n2long*aind;
            linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
            isInfLower = (Ftemp_ii_e(linidx_lower) == -Inf);
            isInfUpper = (Ftemp_ii_e(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(4,:,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            V(:,e_c,jj)=shiftdim(Vtempii-MostTempting,1);
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,e_c,jj)=d_ind; % d
            Policy(2,:,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(3,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        end

    end

end


%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=Policy(3,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)


end
