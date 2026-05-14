function Fmatrix=CreateReturnFnMatrix_Case1_fastOLG_ExpAsset_Disc_Par2(ReturnFn, n_d1, n_d2, n_a1prime, n_a1,n_a2, n_z,N_j, d_gridvals, a1prime_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J, ReturnFnParamsAgeMatrix,Level,Refine) % Refine is an optional input
% Note: d_gridvals is both d1 and d2 (unless n_d1=1 so there is no d1, in which case is just d2)
% Level and Refine are about different shapes of inputs/output
% Set Level=0, unless using Divide-and-Conquer
% When Level=1 or 2, Refine is ignored

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-4)};
end

if n_d1(1)==0
    n_d=n_d2;
else
    n_d=[n_d1,n_d2]; % Almost everything is done without distinguishing d1 and d2, just for some reshapes at the end
end
N_d=prod(n_d);
N_a1prime=prod(n_a1prime);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_z=prod(n_z);

l_d=length(n_d);
l_a1=length(n_a1);
l_a2=length(n_a2);
l_z=length(n_z);
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a1>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_a2>1
    error('Using GPU for the return fn does not allow for more than one experience asset variable')
end
if l_z>8
    error('Using GPU for the return fn does not allow for more than eight of semiz and z variables')
end

if Level==0 || Level==1
    if l_a1>=1
        a1prime1vals=shiftdim(a1prime_gridvals(:,1),-1);
        if l_a1>=2
            a1prime2vals=shiftdim(a1prime_gridvals(:,2),-1);
            if l_a1>=3
                a1prime3vals=shiftdim(a1prime_gridvals(:,3),-1);
                if l_a1>=4
                    a1prime4vals=shiftdim(a1prime_gridvals(:,4),-1);
                end
            end
        end
    end
elseif Level==2 || Level==3
    if l_a1>=1
        a1prime1vals=a1prime_gridvals;
        if l_a1>=2
            error('Cannot yet do divide-and-conquer with experienceasset when there is more than one standard asset')
        end
    end
end
if l_a1>=1
    a1vals=shiftdim(a1_gridvals(:,1),-2);
    if l_a1>=2
        a2vals=shiftdim(a1_gridvals(:,2),-2);
        if l_a1>=3
            a3vals=shiftdim(a1_gridvals(:,3),-2);
            if l_a1>=4
                a4vals=shiftdim(a1_gridvals(:,4),-2);
            end
        end
    end
end
expassetvals=shiftdim(a2_gridvals,-3);
% fastOLG: z_gridvals_J is (1,1,1,1,j,N_z,l_z) for fastOLG with ExpAsset
if l_z>=1
    z1vals=z_gridvals_J(1,1,1,1,:,:,1);
    if l_z>=2
        z2vals=z_gridvals_J(1,1,1,1,:,:,2);
        if l_z>=3
            z3vals=z_gridvals_J(1,1,1,1,:,:,3);
            if l_z>=4
                z4vals=z_gridvals_J(1,1,1,1,:,:,4);
                if l_z>=5
                    z5vals=z_gridvals_J(1,1,1,1,:,:,5);
                    if l_z>=6
                        z6vals=z_gridvals_J(1,1,1,1,:,:,6);
                        if l_z>=7
                            z7vals=z_gridvals_J(1,1,1,1,:,:,7);
                            if l_z>=8
                                z8vals=z_gridvals_J(1,1,1,1,:,:,8);
                            end
                        end
                    end
                end
            end
        end
    end
end


if l_z==1
    if l_d==1
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals, a1vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals, a1vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals, a1vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals, a1vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, ReturnFnParamsCell{:});
        end
    end
elseif l_z==2
    if l_d==1
        if l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals, a1vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_a1==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_a1==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals, a1vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals, a1vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals, a1vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, ReturnFnParamsCell{:});
        end
    end
elseif l_z==3
    if l_d==1
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        end
    end
elseif l_z==4
    if l_d==1
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    end
elseif l_z==5
    if l_d==1
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if  l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
        end
    end
elseif l_z==6
    if l_d==1
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if  l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, ReturnFnParamsCell{:});
        end
    end
elseif l_z==7
    if l_d==1
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if  l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, ReturnFnParamsCell{:});
        end
    end
elseif l_z==8
    if l_d==1
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if  l_a1==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        elseif l_a1==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, ReturnFnParamsCell{:});
        end
    end
end


% Note: cannot have N_d=0 with experience asset
if Level==0
    N_d1=prod(n_d1);
    if Refine==0 || N_d1==0
        Fmatrix=reshape(Fmatrix,[N_d*N_a1prime,N_a1*N_a2,N_j,N_z]);
    elseif Refine==1
        N_d2=prod(n_d2);
        Fmatrix=reshape(Fmatrix,[N_d1,N_d2*N_a1prime,N_a1*N_a2,N_j,N_z]);  % want to refine away d1
    end
elseif Level==1 || Level==3
    Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a1,N_a2,N_j,N_z]);
elseif Level==2 % For level 2
    Fmatrix=reshape(Fmatrix,[N_d*N_a1prime,N_a1*N_a2,N_j,N_z]);
end


end


