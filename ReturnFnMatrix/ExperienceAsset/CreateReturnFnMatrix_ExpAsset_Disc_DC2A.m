function Fmatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1prime_grid, a2prime_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_gridvals, ReturnFnParamsVec, Level)
% Divide-and-conquer in the first standard endogenous state (a1, single dim).
% Folded standard middle endogenous states (a2, may be multi-dim, l_a2 up to 3).
% Last endogenous state is an experience asset (a3, may be multi-dim, l_a3 in {1,2}).
%
% Note: d_gridvals is both d1 and d2 (unless n_d1=0 so there is no d1, in which case is just d2)
% a1: standard endogenous state which will have DC applied
% a2: standard endogenous state
% a3: experienceasset
%
% Level==1: n-monotonicity sweep (a1prime full, a1 at level1 subset).
%   a1prime_grid input: column [N_a1prime, 1] (= a1_grid full).
%   Output shape: [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_a3, N_z].
% Level==2: narrow-band sweep, direct max (j=N_j no-V_Jplus1 case).
%   a1prime_grid input: [N_d, maxgap+1, N_a2prime, 1, N_a2, N_a3, N_z].
%   Output shape: [N_d*N_a1prime*N_a2prime, N_a1*N_a2*N_a3, N_z].
% Level==3: narrow-band sweep, multi-D output (for broadcast with DiscountedEV before max).
%   a1prime_grid input: same as Level==2.
%   Output shape: [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_a3, N_z].

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

if n_d1(1)==0
    n_d=n_d2;
else
    n_d=[n_d1,n_d2];
end
N_d=prod(n_d);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_z=prod(n_z);

l_d=length(n_d);
l_a2=length(n_a2);
l_a3=length(n_a3);
l_z=length(n_z);

if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variables (you have length(n_d1)+length(n_d2)>4)')
end
if l_a2>3
    error('Using GPU for the return fn does not allow for more than three of folded a2 variables (DC2A_nod1 ExpAsset)')
end
if l_a3>2
    error('experienceasset currently supports length(n_a3) in {1,2}')
end
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variables (you have length(n_z)>4)')
end

if Level==1
    N_a1prime=size(a1prime_grid,1); % column [N_a1prime, 1]
    a1prime_grid=shiftdim(a1prime_grid,-1); % column -> dim 2
elseif Level==2 || Level==3
    N_a1prime=size(a1prime_grid,2); % already laid out as [N_d, maxgap+1, ...]
end
N_a2prime=N_a2;
N_a1=size(a1_grid,1);

% a2prime at dim 3 (one per folded component)
if l_a2>=1
    a2prime1vals=shiftdim(a2prime_gridvals(:,1),-2);
    if l_a2>=2
        a2prime2vals=shiftdim(a2prime_gridvals(:,2),-2);
        if l_a2>=3
            a2prime3vals=shiftdim(a2prime_gridvals(:,3),-2);
        end
    end
end

% a1 at dim 4
a1vals=shiftdim(a1_grid,-3);

% a2 at dim 5 (one per folded component)
if l_a2>=1
    a21vals=shiftdim(a2_gridvals(:,1),-4);
    if l_a2>=2
        a22vals=shiftdim(a2_gridvals(:,2),-4);
        if l_a2>=3
            a23vals=shiftdim(a2_gridvals(:,3),-4);
        end
    end
end

% a3 (expasset) at dim 6
if l_a3==1
    a3vals=shiftdim(a3_gridvals(:,1),-5);
elseif l_a3==2
    a3vals_1=shiftdim(a3_gridvals(:,1),-5);
    a3vals_2=shiftdim(a3_gridvals(:,2),-5);
end

% z at dim 7 (one per component)
if l_z>=1
    z1vals=shiftdim(z_gridvals(:,1),-6);
    if l_z>=2
        z2vals=shiftdim(z_gridvals(:,2),-6);
        if l_z>=3
            z3vals=shiftdim(z_gridvals(:,3),-6);
            if l_z>=4
                z4vals=shiftdim(z_gridvals(:,4),-6);
            end
        end
    end
end

if l_z==1
    if l_d==1
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==2
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==3
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==4
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals, ReturnFnParamsCell{:});
            end
        end
    end
elseif l_z==2
    if l_d==1
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==2
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==3
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==4
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals, ReturnFnParamsCell{:});
            end
        end
    end
elseif l_z==3
    if l_d==1
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==2
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==3
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==4
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
            end
        end
    end
elseif l_z==4
    if l_d==1
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==2
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==3
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_d==4
        if l_a2==1
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==2
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        elseif l_a2==3
            if l_a3==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            elseif l_a3==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
            end
        end
    end
end

if Level==1 || Level==3
    Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a2prime,N_a1,N_a2,N_a3,N_z]);
elseif Level==2
    Fmatrix=reshape(Fmatrix,[N_d*N_a1prime*N_a2prime,N_a1*N_a2*N_a3,N_z]);
end

end
