function Values=EvalFnOnAgentDist_Grid_Case1e(FnToEvaluate,FnToEvaluateParams,PolicyValuesPermute,n_d,n_a,n_z,n_e,a_grid,z_grid,e_grid,Parallel)

if Parallel~=2
    fprintf('Need to use EvalFnOnAgentDist_Grid_Case1_cpu instead of EvalFnOnAgentDist_Grid_Case1 \n')
    dbstack
    return
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
l_e=length(n_e);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

ParamCell=cell(length(FnToEvaluateParams),1);
for ii=1:length(FnToEvaluateParams)
    ParamCell(ii,1)={FnToEvaluateParams(ii)};
end

% if l_d>4
%     error
% end
% if l_a>4
%     error
% end
% if l_z>4
%     error
% end
% if l_e>4
%     error
% end

if l_a>=1
    a1vals=a_grid(1:n_a(1));
    if l_a>=2
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1);
        if l_a>=3
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-2);
            if l_a>=4
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-3);
            end
        end
    end
end
if all(size(z_grid)==[sum(n_z),1]) % kroneker product z_grid
    z_shift=l_z;
    if l_z>=1
        z1vals=shiftdim(z_grid(1:n_z(1)),-l_a);
        if l_z>=2
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-l_a-1);
            if l_z>=3
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-l_a-2);
                if l_z>=4
                    z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-l_a-3);
                    if l_z>=5
                        z5vals=shiftdim(z_grid(sum(n_z(1:4))+1:sum(n_z(1:5))),-l_a-4);
                    end
                end
            end
        end
    end
elseif all(size(z_grid)==[prod(n_z),l_z]) % joint z_grid
    z_shift=1;
    if l_z>=1
        z1vals=shiftdim(z_grid(:,1),-l_a);
        if l_z>=2
            z2vals=shiftdim(z_grid(:,2),-l_a);
            if l_z>=3
                z3vals=shiftdim(z_grid(:,3),-l_a);
                if l_z>=4
                    z4vals=shiftdim(z_grid(:,4),-l_a);
                    if l_z>=5
                        z5vals=shiftdim(z_grid(:,5),-l_a);
                    end
                end
            end
        end
    end
end
if all(size(e_grid)==[sum(n_e),1]) % kroneker product e_grid
    if l_e>=1
        e1vals=shiftdim(e_grid(1:n_e(1)),-l_a-z_shift);
        if l_e>=2
            e2vals=shiftdim(e_grid(n_e(1)+1:n_e(1)+n_e(2)),-l_a-z_shift-1);
            if l_e>=3
                e3vals=shiftdim(e_grid(sum(n_e(1:2))+1:sum(n_e(1:3))),-l_a-z_shift-2);
                if l_e>=4
                    e4vals=shiftdim(e_grid(sum(n_e(1:3))+1:sum(n_e(1:4))),-l_a-z_shift-3);
                    if l_e>=5
                        e5vals=shiftdim(e_grid(sum(n_e(1:4))+1:sum(n_e(1:5))),-l_a-z_shift-4);
                    end
                end
            end
        end
    end
elseif all(size(e_grid)==[prod(n_e),l_e]) % joint e_grid
    if l_e>=1
        e1vals=shiftdim(e_grid(:,1),-l_a-z_shift);
        if l_e>=2
            e2vals=shiftdim(e_grid(:,2),-l_a-z_shift);
            if l_e>=3
                e3vals=shiftdim(e_grid(:,3),-l_a-z_shift);
                if l_e>=4
                    e4vals=shiftdim(e_grid(:,4),-l_a-z_shift);
                    if l_e>=5
                        e5vals=shiftdim(e_grid(:,5),-l_a-z_shift);
                    end
                end
            end
        end
    end
end

if l_a+l_z+l_e==2
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,l_d+4);
                end
            end
        end
    end
elseif l_a+l_z+l_e==3
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,l_d+4);
                end
            end
        end
    end
elseif l_a+l_z+l_e==4
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,:,l_d+4);
                end
            end
        end
    end
elseif l_a+l_z+l_e==5
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,:,:,l_d+4);
                end
            end
        end
    end
elseif l_a+l_z+l_e==6
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,:,:,:,l_d+4);
                end
            end
        end
    end
elseif l_a+l_z+l_e==7
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,:,:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,:,:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,:,:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,:,:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,:,:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,:,:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,:,:,:,:,l_d+4);
                end
            end
        end
    end
elseif l_a+l_z+l_e==8
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,l_d+4);
                end
            end
        end
    end
elseif l_a+l_z+l_e==9
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,l_d+4);
                end
            end
        end
    end
elseif l_a+l_z+l_e==10
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,:,:,:,:,:,:,:,l_d+4);
                end
            end
        end
    end
end

if all(size(z_grid)==[prod(n_z),l_z]) && all(size(e_grid)==[prod(n_e),l_e])  % joint z grid and joint e_grid
    if l_d>=1
        d1vals=reshape(d1vals,[n_a,N_z,N_e]);
        if l_d>=2
            d2vals=reshape(d2vals,[n_a,N_z,N_e]);
            if l_d>=3
                d3vals=reshape(d3vals,[n_a,N_z,N_e]);
                if l_d>=4
                    d4vals=reshape(d4vals,[n_a,N_z,N_e]);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=reshape(a1primevals,[n_a,N_z,N_e]);
        if l_a>=2
            a2primevals=reshape(a2primevals,[n_a,N_z,N_e]);
            if l_a>=3
                a3primevals=reshape(a3primevals,[n_a,N_z,N_e]);
                if l_a>=4
                    a4primevals=reshape(a4primevals,[n_a,N_z,N_e]);
                end
            end
        end
    end
elseif all(size(z_grid)==[prod(n_z),l_z]) % joint z grid
    if l_d>=1
        d1vals=reshape(d1vals,[n_a,N_z,n_e]);
        if l_d>=2
            d2vals=reshape(d2vals,[n_a,N_z,n_e]);
            if l_d>=3
                d3vals=reshape(d3vals,[n_a,N_z,n_e]);
                if l_d>=4
                    d4vals=reshape(d4vals,[n_a,N_z,n_e]);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=reshape(a1primevals,[n_a,N_z,n_e]);
        if l_a>=2
            a2primevals=reshape(a2primevals,[n_a,N_z,n_e]);
            if l_a>=3
                a3primevals=reshape(a3primevals,[n_a,N_z,n_e]);
                if l_a>=4
                    a4primevals=reshape(a4primevals,[n_a,N_z,n_e]);
                end
            end
        end
    end
elseif all(size(e_grid)==[prod(n_e),l_e]) % joint e grid
    if l_d>=1
        d1vals=reshape(d1vals,[n_a,n_z,N_e]);
        if l_d>=2
            d2vals=reshape(d2vals,[n_a,n_z,N_e]);
            if l_d>=3
                d3vals=reshape(d3vals,[n_a,n_z,N_e]);
                if l_d>=4
                    d4vals=reshape(d4vals,[n_a,n_z,N_e]);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=reshape(a1primevals,[n_a,n_z,N_e]);
        if l_a>=2
            a2primevals=reshape(a2primevals,[n_a,n_z,N_e]);
            if l_a>=3
                a3primevals=reshape(a3primevals,[n_a,n_z,N_e]);
                if l_a>=4
                    a4primevals=reshape(a4primevals,[n_a,n_z,N_e]);
                end
            end
        end
    end
end

if l_e==1
    if l_d==0 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
    end
elseif l_e==2
    if l_d==0 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, ParamCell{:});
    end
elseif l_e==3
    if l_d==0 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, ParamCell{:});
    end
elseif l_e==4
    if l_d==0 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==1 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==2 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==3 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==4 
        Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
    end
end

Values=reshape(Values,[N_a,N_z*N_e]);


end