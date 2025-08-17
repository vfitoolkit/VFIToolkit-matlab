function Fmatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1, n_d2, n_a1prime, n_a1,n_a2, n_z,n_e, d_gridvals, a1prime_gridvals, a1_gridvals, a2_gridvals, z_gridvals, e_gridvals, ReturnFnParams,Level,Refine) % Refine is an optional input
% Note: d_gridvals is both d1 and d2 (unless n_d1=1 so there is no d1, in which case is just d2)
% Level and Refine are about different shapes of inputs/output
% Set Level=0, unless using Divide-and-Conquer
% When Level=1 or 2, Refine is ignored


ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    ParamCell(ii,1)={ReturnFnParams(ii)};
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
N_e=prod(n_e);

l_d=length(n_d);
l_a1=length(n_a1); 
l_a2=length(n_a2); 
l_z=length(n_z);
l_e=length(n_e);
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
if l_e>4
    error('Using GPU for the return fn does not allow for more than four of e variable (you have length(n_e)>4)')
end


if l_d>=1
    d1vals=d_gridvals(:,1); 
    if l_d>=2
        d2vals=d_gridvals(:,2);
        if l_d>=3
            d3vals=d_gridvals(:,3);
            if l_d>=4
                d4vals=d_gridvals(:,4);
            end
        end
    end
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
elseif Level==2
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
if l_z>=1
    z1vals=shiftdim(z_gridvals(:,1),-4);
    if l_z>=2
        z2vals=shiftdim(z_gridvals(:,2),-4);
        if l_z>=3
            z3vals=shiftdim(z_gridvals(:,3),-4);
            if l_z>=4
                z4vals=shiftdim(z_gridvals(:,4),-4);
                if l_z>=5
                    z5vals=shiftdim(z_gridvals(:,5),-4);
                    if l_z>=6
                        z6vals=shiftdim(z_gridvals(:,6),-4);
                        if l_z>=7
                            z7vals=shiftdim(z_gridvals(:,7),-4);
                            if l_z>=8
                                z8vals=shiftdim(z_gridvals(:,8),-4);
                            end
                        end
                    end
                end
            end
        end
    end
end
if l_e>=1
    e1vals=shiftdim(e_gridvals(:,1),-5);
    if l_e>=2
        e2vals=shiftdim(e_gridvals(:,2),-5);
        if l_e>=3
            e3vals=shiftdim(e_gridvals(:,3),-5);
            if l_e>=4
                e4vals=shiftdim(e_gridvals(:,4),-5);
                if l_e>=5
                    e5vals=shiftdim(e_gridvals(:,5),-5);
                end
            end
        end
    end
end

if l_e==1
    if l_z==1
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, ParamCell{:});
            end
        end
    elseif l_z==2
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, ParamCell{:});
            end
        end
    elseif l_z==3
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            end
        end
    elseif l_z==4
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            end
        end
    elseif l_z==5
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
            end
        end
    elseif l_z==6
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, ParamCell{:});
            end
        end
    elseif l_z==7
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, ParamCell{:});
            end
        end
    elseif l_z==8
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, ParamCell{:});
            end
        end
    end
elseif l_e==2
    if l_z==1
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, ParamCell{:});
            end
        end
    elseif l_z==2
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
            end
        end
    elseif l_z==3
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
            end
        end
    elseif l_z==4
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
            end
        end
    elseif l_z==5
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, ParamCell{:});
            end
        end
    elseif l_z==6
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, ParamCell{:});
            end
        end
    elseif l_z==7
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, ParamCell{:});
            end
        end
    elseif l_z==8
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, ParamCell{:});
            end
        end
    end
elseif l_e==3
    if l_z==1
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        end
    elseif l_z==2
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        end
    elseif l_z==3
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        end
    elseif l_z==4
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        end
    elseif l_z==5
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        end
    elseif l_z==6
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        end
    elseif l_z==7
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        end
    elseif l_z==8
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, ParamCell{:});
            end
        end
    end
elseif l_e==4
    if l_z==1
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        end
    elseif l_z==2
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        end
    elseif l_z==3
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        end
    elseif l_z==4
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        end
    elseif l_z==5
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        end
    elseif l_z==6
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        end
    elseif l_z==7
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        end
    elseif l_z==8
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
            end
        end
    end
elseif l_e==5
    if l_z==1
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        end
    elseif l_z==2
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        end
    elseif l_z==3
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        end
    elseif l_z==4
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==4
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        end
    elseif l_z==5
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        end
    elseif l_z==6
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        end
    elseif l_z==7
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        end
    elseif l_z==8
        if l_d==1
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==2
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==3
            if l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        elseif l_d==4
            if  l_a1==1
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==2
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==3
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            elseif l_a1==4
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1prime1vals,a1prime2vals,a1prime3vals,a1prime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals, e1vals, e2vals, e3vals, e4vals,e5vals, ParamCell{:});
            end
        end
    end
end


% Note: cannot have N_d=0 with experience asset
if Level==0
    N_d1=prod(n_d1);
    if Refine==0 || N_d1==0
        Fmatrix=reshape(Fmatrix,[N_d*N_a1,N_a1*N_a2,N_z,N_e]);
    elseif Refine==1
        N_d2=prod(n_d2);
        Fmatrix=reshape(Fmatrix,[N_d1,N_d2*N_a1,N_a1*N_a2,N_z,N_e]);  % want to refine away d1
    end
elseif Level==1
    Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a1,N_a2,N_z,N_e]);
elseif Level==2 % For level 2
    Fmatrix=reshape(Fmatrix,[N_d*N_a1prime,N_a1*N_a2,N_z,N_e]);
end


end


