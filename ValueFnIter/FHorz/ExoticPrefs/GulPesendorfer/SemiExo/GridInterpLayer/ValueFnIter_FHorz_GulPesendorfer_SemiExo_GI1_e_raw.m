function [V,Policy]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI1_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, n_e,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Gul-Pesendorfer with a semi-exogenous state and the grid interpolation layer. The tempted
% objective u+v+beta*EV goes through the standard per-d2 two-stage machinery (the coarse
% midpoint/argmax and the fine window are those of the TEMPTED objective, with the L2 -Inf
% flag based on u+v); the most-tempting term is the max of v over the FINE grid, found by the
% same two-stage scheme but around v's OWN coarse argmax (otherwise the chosen fine point
% could be more tempting than the coarse max of v, making the self-control cost negative): a
% per-d2 fine max is collected alongside each inner solve and the max over d2 is subtracted
% from V after the outer max (the '-max v' term is constant w.r.t. the choice given the
% state, so the subtraction after the d2-max is exact).

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
Policy=zeros(5,N_a,N_semiz*N_z,N_e,N_j,'gpuArray'); % First dimension: d1, d2, aprime, aprime2
Policy(5,:,:,:,:)=2; % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
% When ReturnFn is -Inf on one of the course grid points, we will allow fine index between that and the neighbouring course grid point, but we use L2flag to record this and so later avoid that -Inf point when simulating/iteration

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

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

aind=gpuArray(0:1:N_a-1); % already includes -1
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1 (used when vectorizing over semiz only)
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1

bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-2); % already includes -1
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-2); % already includes -1 (used when vectorizing over semiz only)

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
flag_ford2_jj=2*ones(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
MostTempting_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray'); % per-d2 fine (d1,aprime) max of the temptation

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
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_bothz, n_e, d_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        TemptationMatrix=CreateReturnFnMatrix_Disc_e(TemptationFn, n_d, n_a, n_bothz, n_e, d_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);
        % Treat standard problem as just being the first layer
        [~,maxindex]=max(ReturnMatrix+TemptationMatrix,[],2);

        % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a-1),2);
        aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, n_d, n_bothz, n_e, d_gridvals, aprime_grid(aprimeindexesT), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
        MostTempting=max(TemptationMatrix_Tii,[],1);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, n_d, n_bothz, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
        Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
        [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii-MostTempting,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*bothzind+N_d*N_a*N_bothz*eind; % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e

        % L2 flag: detect -Inf on the coarse neighbour we'd put weight on
        L2offset      = ceil(maxindexL2/N_d);
        linidx_lower  = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*bothzind + N_d*n2long*N_a*N_bothz*eind;
        linidx_upper  = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*bothzind + N_d*n2long*N_a*N_bothz*eind;
        isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
        isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(5,:,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        Policy(1,:,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy(2,:,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy(3,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
        Policy(4,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_bothz, special_n_e, d_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            TemptationMatrix_e=CreateReturnFnMatrix_Disc_e(TemptationFn, n_d, n_a, n_bothz, special_n_e, d_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_val, TemptationFnParamsVec,1);
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(ReturnMatrix_e+TemptationMatrix_e,[],2);

            % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
            [~,maxindexT]=max(TemptationMatrix_e,[],2);
            midpointT=max(min(maxindexT,n_a-1),2);
            aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, n_d, n_bothz, special_n_e, d_gridvals, aprime_grid(aprimeindexesT), a_grid, bothz_gridvals_J(:,:,N_j), e_val, TemptationFnParamsVec,2);
            MostTempting=max(TemptationMatrix_Tii,[],1);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a-by-n_bothz
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_bothz, special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
            TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, n_d, n_bothz, special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, TemptationFnParamsVec,2);
            Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
            [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii-MostTempting,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*bothzind; % midpoint is n_d-by-1-by-n_a-by-n_bothz

            % L2 flag: detect -Inf on the coarse neighbour we'd put weight on
            L2offset      = ceil(maxindexL2/N_d);
            linidx_lower  = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
            linidx_upper  = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
            isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
            isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(5,:,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            Policy(1,:,:,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy(2,:,:,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy(3,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
            Policy(4,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        end

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz

        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, [n_semiz,special_n_z], special_n_e, d_gridvals, a_grid, z_valblock, e_val, ReturnFnParamsVec,1);
                TemptationMatrix_ze=CreateReturnFnMatrix_Disc_e(TemptationFn, n_d, n_a, [n_semiz,special_n_z], special_n_e, d_gridvals, a_grid, z_valblock, e_val, TemptationFnParamsVec,1);
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(ReturnMatrix_ze+TemptationMatrix_ze,[],2);

                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix_ze,[],2);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, n_d, [n_semiz,special_n_z], special_n_e, d_gridvals, aprime_grid(aprimeindexesT), a_grid, z_valblock, e_val, TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a-by-n_semiz
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, [n_semiz,special_n_z], special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, e_val, ReturnFnParamsVec,2);
                TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, n_d, [n_semiz,special_n_z], special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, e_val, TemptationFnParamsVec,2);
                Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
                [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);
                V(:,semizblock,e_c,N_j)=shiftdim(Vtempii-MostTempting,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind+N_d*N_a*semizind; % midpoint is n_d-by-1-by-n_a-by-n_semiz

                % L2 flag: detect -Inf on the coarse neighbour we'd put weight on
                L2offset      = ceil(maxindexL2/N_d);
                linidx_lower  = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*semizind;
                linidx_upper  = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*semizind;
                isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
                isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(5,:,semizblock,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                Policy(1,:,semizblock,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
                Policy(2,:,semizblock,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
                Policy(3,:,semizblock,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
                Policy(4,:,semizblock,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e

        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_bothz, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,1);
                TemptationMatrix_ze=CreateReturnFnMatrix_Disc_e(TemptationFn, n_d, n_a, special_n_bothz, special_n_e, d_gridvals, a_grid, z_val, e_val, TemptationFnParamsVec,1);
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(ReturnMatrix_ze+TemptationMatrix_ze,[],2);

                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix_ze,[],2);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, n_d, special_n_bothz, special_n_e, d_gridvals, aprime_grid(aprimeindexesT), a_grid, z_val, e_val, TemptationFnParamsVec,2);
                MostTempting=max(TemptationMatrix_Tii,[],1);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, special_n_bothz, special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, e_val, ReturnFnParamsVec,2);
                TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, n_d, special_n_bothz, special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, e_val, TemptationFnParamsVec,2);
                Ftemp_ii=ReturnMatrix_ii+TemptationMatrix_ii;
                [Vtempii,maxindexL2]=max(Ftemp_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii-MostTempting,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a

                % L2 flag: detect -Inf on the coarse neighbour we'd put weight on
                L2offset      = ceil(maxindexL2/N_d);
                linidx_lower  = d_ind                  + N_d*n2long*aind;
                linidx_upper  = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
                isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
                isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(5,:,z_c,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                Policy(1,:,z_c,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
                Policy(2,:,z_c,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
                Policy(3,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % midpoint
                Policy(4,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
            end
        end
    end
else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);    % First, switch V_Jplus1 into Kron form
    EV=sum(EV.*pi_e_J(1,1,:,N_j+1),3);

    if vfoptions.lowmemory==0

        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            TemptationMatrix_d2=CreateReturnFnMatrix_Disc_e(TemptationFn, special_n_d, n_a, n_bothz, n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);
            entireRHS=ReturnMatrix_d2+TemptationMatrix_d2+DiscountFactorParamsVec*shiftdim(EV_d2,-1);
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS,[],2); % second dimension, so this is the optimal aprime (on first layer, so on a_grid)

            % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
            [~,maxindexT]=max(TemptationMatrix_d2,[],2);
            midpointT=max(min(maxindexT,n_a-1),2);
            aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexesT), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
            MostTempting_ford2_jj(:,:,:,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,aprime) max for this d2

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
            Ftemp_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii;
            aprimez=aprimeindexes+n2aprime*bothzBind;
            entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind+N_d1*N_a*N_bothz*eind; % loweredge is n_d-by-1-by-n_a-by-n_bothz-by-n_e
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoint(allind));

            % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
            L2offset      = ceil(maxindex/N_d1);
            linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
            isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford2_jj(:,:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj-max(MostTempting_ford2_jj,[],4); % subtract the most-tempting term
        Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,N_j)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy(4,:,:,:,N_j)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,N_j)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, special_n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
                TemptationMatrix_e=CreateReturnFnMatrix_Disc_e(TemptationFn, special_n_d, n_a, n_bothz, special_n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_val, TemptationFnParamsVec,1);
                entireRHS_e=ReturnMatrix_e+TemptationMatrix_e+DiscountFactorParamsVec*shiftdim(EV_d2,-1);
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(entireRHS_e,[],2); % second dimension, so this is the optimal aprime (on first layer, so on a_grid)

                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix_e,[],2);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexesT), a_grid, bothz_gridvals_J(:,:,N_j), e_val, TemptationFnParamsVec,2);
                MostTempting_ford2_jj(:,:,e_c,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,aprime) max for this d2

                % Now do the second layer for the interpolation

                % Turn maxindex into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a-by-n_bothz
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, TemptationFnParamsVec,2);
                Ftemp_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii;
                aprimez=aprimeindexes+n2aprime*bothzBind;
                entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_bothz]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind; % loweredge is n_d-by-1-by-n_a-by-n_bothz
                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoint(allind));

                % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
                L2offset      = ceil(maxindex/N_d1);
                linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
                isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford2_jj(:,:,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj-max(MostTempting_ford2_jj,[],4); % subtract the most-tempting term
        Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,N_j)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy(4,:,:,:,N_j)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,N_j)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

                EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % Interpolate EV over aprime_grid
                EVinterp=interp1(a_grid,EV_d2z,aprime_grid);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, z_valblock, e_val, ReturnFnParamsVec,1);
                    TemptationMatrix_ze=CreateReturnFnMatrix_Disc_e(TemptationFn, special_n_d, n_a, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, z_valblock, e_val, TemptationFnParamsVec,1);
                    entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze+DiscountFactorParamsVec*shiftdim(EV_d2z,-1);
                    % Treat standard problem as just being the first layer
                    [~,maxindex]=max(entireRHS_ze,[],2); % second dimension, so this is the optimal aprime (on first layer, so on a_grid)

                    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                    [~,maxindexT]=max(TemptationMatrix_ze,[],2);
                    midpointT=max(min(maxindexT,n_a-1),2);
                    aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, aprime_grid(aprimeindexesT), a_grid, z_valblock, e_val, TemptationFnParamsVec,2);
                    MostTempting_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,aprime) max for this d2

                    % Now do the second layer for the interpolation

                    % Turn maxindex into the 'midpoint'
                    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                    % midpoint is n_d-by-1-by-n_a-by-n_semiz
                    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                    % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
                    ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, e_val, ReturnFnParamsVec,2);
                    TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, e_val, TemptationFnParamsVec,2);
                    Ftemp_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii;
                    aprimez=aprimeindexes+n2aprime*semizBind;
                    entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_semiz]);
                    [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                    V_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(maxindex,1);

                    d1_ind=rem(maxindex-1,N_d1)+1;
                    allind=d1_ind+N_d1*aind+N_d1*N_a*semizind; % loweredge is n_d-by-1-by-n_a-by-n_semiz
                    midpoint_ford2_jj(:,semizblock,e_c,d2_c)=squeeze(midpoint(allind));

                    % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
                    L2offset      = ceil(maxindex/N_d1);
                    linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                    linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                    isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
                    isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj-max(MostTempting_ford2_jj,[],4); % subtract the most-tempting term
        Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,N_j)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy(4,:,:,:,N_j)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,N_j)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2);

                % Interpolate EV over aprime_grid
                EVinterp=interp1(a_grid,EV_d2z,aprime_grid);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, special_n_bothz, special_n_e, d12c_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,1);
                    TemptationMatrix_ze=CreateReturnFnMatrix_Disc_e(TemptationFn, special_n_d, n_a, special_n_bothz, special_n_e, d12c_gridvals, a_grid, z_val, e_val, TemptationFnParamsVec,1);
                    entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze+DiscountFactorParamsVec*shiftdim(EV_d2z,-1);
                    % Treat standard problem as just being the first layer
                    [~,maxindex]=max(entireRHS_ze,[],2); % second dimension, so this is the optimal aprime (on first layer, so on a_grid)

                    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                    [~,maxindexT]=max(TemptationMatrix_ze,[],2);
                    midpointT=max(min(maxindexT,n_a-1),2);
                    aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexesT), a_grid, z_val, e_val, TemptationFnParamsVec,2);
                    MostTempting_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,aprime) max for this d2

                    % Now do the second layer for the interpolation

                    % Turn maxindex into the 'midpoint'
                    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                    % midpoint is n_d-by-1-by-n_a
                    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                    % aprime possibilities are n_d-by-n2long-by-n_a
                    ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, e_val, ReturnFnParamsVec,2);
                    TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, e_val, TemptationFnParamsVec,2);
                    Ftemp_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii;
                    entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes),[N_d1*n2long,N_a]);
                    [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                    V_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(maxindex,1);

                    d1_ind=rem(maxindex-1,N_d1)+1;
                    allind=d1_ind+N_d1*aind; % loweredge is n_d-by-1-by-n_a
                    midpoint_ford2_jj(:,z_c,e_c,d2_c)=squeeze(midpoint(allind));

                    % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
                    L2offset      = ceil(maxindex/N_d1);
                    linidx_lower  = d1_ind                   + N_d1*n2long*aind;
                    linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                    isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
                    isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford2_jj(:,z_c,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj-max(MostTempting_ford2_jj,[],4); % subtract the most-tempting term
        Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,N_j)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy(4,:,:,:,N_j)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,N_j)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
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

    EV=V(:,:,:,jj+1);
    EV=sum(EV.*pi_e_J(1,1,:,jj+1),3);

    if vfoptions.lowmemory==0

        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
            TemptationMatrix_d2=CreateReturnFnMatrix_Disc_e(TemptationFn, special_n_d, n_a, n_bothz, n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
            entireRHS=ReturnMatrix_d2+TemptationMatrix_d2+DiscountFactorParamsVec*shiftdim(EV_d2,-1);
            % Treat standard problem as just being the first layer
            [~,maxindex]=max(entireRHS,[],2); % second dimension, so this is the optimal aprime (on first layer, so on a_grid)

            % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
            [~,maxindexT]=max(TemptationMatrix_d2,[],2);
            midpointT=max(min(maxindexT,n_a-1),2);
            aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
            TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexesT), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), TemptationFnParamsVec,2);
            MostTempting_ford2_jj(:,:,:,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,aprime) max for this d2

            % Now do the second layer for the interpolation

            % Turn maxindex into the 'midpoint'
            midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), TemptationFnParamsVec,2);
            Ftemp_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii;
            aprimez=aprimeindexes+n2aprime*bothzBind;
            entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind+N_d1*N_a*N_bothz*eind; % loweredge is n_d-by-1-by-n_a-by-n_bothz-by-n_e
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoint(allind));

            % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
            L2offset      = ceil(maxindex/N_d1);
            linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
            isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford2_jj(:,:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj-max(MostTempting_ford2_jj,[],4); % subtract the most-tempting term
        Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,jj)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, special_n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);
                TemptationMatrix_e=CreateReturnFnMatrix_Disc_e(TemptationFn, special_n_d, n_a, n_bothz, special_n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,jj), e_val, TemptationFnParamsVec,1);
                entireRHS_e=ReturnMatrix_e+TemptationMatrix_e+DiscountFactorParamsVec*shiftdim(EV_d2,-1);
                % Treat standard problem as just being the first layer
                [~,maxindex]=max(entireRHS_e,[],2); % second dimension, so this is the optimal aprime (on first layer, so on a_grid)

                % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                [~,maxindexT]=max(TemptationMatrix_e,[],2);
                midpointT=max(min(maxindexT,n_a-1),2);
                aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexesT), a_grid, bothz_gridvals_J(:,:,jj), e_val, TemptationFnParamsVec,2);
                MostTempting_ford2_jj(:,:,e_c,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,aprime) max for this d2

                % Now do the second layer for the interpolation

                % Turn maxindex into the 'midpoint'
                midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-by-1-by-n_a-by-n_bothz
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_val, TemptationFnParamsVec,2);
                Ftemp_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii;
                aprimez=aprimeindexes+n2aprime*bothzBind;
                entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_bothz]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind; % loweredge is n_d-by-1-by-n_a-by-n_bothz
                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoint(allind));

                % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
                L2offset      = ceil(maxindex/N_d1);
                linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
                isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford2_jj(:,:,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj-max(MostTempting_ford2_jj,[],4); % subtract the most-tempting term
        Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,jj)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

                EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % Interpolate EV over aprime_grid
                EVinterp=interp1(a_grid,EV_d2z,aprime_grid);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, z_valblock, e_val, ReturnFnParamsVec,1);
                    TemptationMatrix_ze=CreateReturnFnMatrix_Disc_e(TemptationFn, special_n_d, n_a, [n_semiz,special_n_z], special_n_e, d12c_gridvals, a_grid, z_valblock, e_val, TemptationFnParamsVec,1);
                    entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze+DiscountFactorParamsVec*shiftdim(EV_d2z,-1);
                    % Treat standard problem as just being the first layer
                    [~,maxindex]=max(entireRHS_ze,[],2); % second dimension, so this is the optimal aprime (on first layer, so on a_grid)

                    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                    [~,maxindexT]=max(TemptationMatrix_ze,[],2);
                    midpointT=max(min(maxindexT,n_a-1),2);
                    aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, aprime_grid(aprimeindexesT), a_grid, z_valblock, e_val, TemptationFnParamsVec,2);
                    MostTempting_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,aprime) max for this d2

                    % Now do the second layer for the interpolation

                    % Turn maxindex into the 'midpoint'
                    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                    % midpoint is n_d-by-1-by-n_a-by-n_semiz
                    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                    % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
                    ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, e_val, ReturnFnParamsVec,2);
                    TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, [n_semiz,special_n_z], special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, e_val, TemptationFnParamsVec,2);
                    Ftemp_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii;
                    aprimez=aprimeindexes+n2aprime*semizBind;
                    entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_semiz]);
                    [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                    V_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(maxindex,1);

                    d1_ind=rem(maxindex-1,N_d1)+1;
                    allind=d1_ind+N_d1*aind+N_d1*N_a*semizind; % loweredge is n_d-by-1-by-n_a-by-n_semiz
                    midpoint_ford2_jj(:,semizblock,e_c,d2_c)=squeeze(midpoint(allind));

                    % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
                    L2offset      = ceil(maxindex/N_d1);
                    linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                    linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                    isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
                    isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj-max(MostTempting_ford2_jj,[],4); % subtract the most-tempting term
        Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,jj)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

                EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2);

                % Interpolate EV over aprime_grid
                EVinterp=interp1(a_grid,EV_d2z,aprime_grid);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, special_n_bothz, special_n_e, d12c_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,1);
                    TemptationMatrix_ze=CreateReturnFnMatrix_Disc_e(TemptationFn, special_n_d, n_a, special_n_bothz, special_n_e, d12c_gridvals, a_grid, z_val, e_val, TemptationFnParamsVec,1);
                    entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze+DiscountFactorParamsVec*shiftdim(EV_d2z,-1);
                    % Treat standard problem as just being the first layer
                    [~,maxindex]=max(entireRHS_ze,[],2); % second dimension, so this is the optimal aprime (on first layer, so on a_grid)

                    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
                    [~,maxindexT]=max(TemptationMatrix_ze,[],2);
                    midpointT=max(min(maxindexT,n_a-1),2);
                    aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexesT), a_grid, z_val, e_val, TemptationFnParamsVec,2);
                    MostTempting_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,aprime) max for this d2

                    % Now do the second layer for the interpolation

                    % Turn maxindex into the 'midpoint'
                    midpoint=max(min(maxindex,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                    % midpoint is n_d-by-1-by-n_a
                    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                    % aprime possibilities are n_d-by-n2long-by-n_a
                    ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, e_val, ReturnFnParamsVec,2);
                    TemptationMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, special_n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, e_val, TemptationFnParamsVec,2);
                    Ftemp_ii=ReturnMatrix_d2ii+TemptationMatrix_d2ii;
                    entireRHS_ii=Ftemp_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes),[N_d1*n2long,N_a]);
                    [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                    V_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(maxindex,1);

                    d1_ind=rem(maxindex-1,N_d1)+1;
                    allind=d1_ind+N_d1*aind; % loweredge is n_d-by-1-by-n_a
                    midpoint_ford2_jj(:,z_c,e_c,d2_c)=squeeze(midpoint(allind));

                    % L2 flag (per d2): detect -Inf on the coarse neighbour we'd put weight on
                    L2offset      = ceil(maxindex/N_d1);
                    linidx_lower  = d1_ind                   + N_d1*n2long*aind;
                    linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                    isInfLower    = (Ftemp_ii(linidx_lower) == -Inf);
                    isInfUpper    = (Ftemp_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford2_jj(:,z_c,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj-max(MostTempting_ford2_jj,[],4); % subtract the most-tempting term
        Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]); %aprimeL2ind
        Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]); % midpoint
        Policy(5,:,:,:,jj)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    end

end

%% Currently Policy(3,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(3,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust; % lower grid point
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% Policy=squeeze(Policy(1,:,:,:,:)+N_d1*(Policy(2,:,:,:,:)-1)+N_d*(Policy(3,:,:,:,:)-1)+N_d*N_a*(Policy(4,:,:,:,:)-1)+N_d*N_a*(n2short+2)*(Policy(5,:,:,:,:)-1));




end