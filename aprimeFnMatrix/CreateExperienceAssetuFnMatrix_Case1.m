function [aprimeIndexes,aprimeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d, n_a, n_u, d_grid, a_grid, u_grid, aprimeFnParams, aprimeIndexAsColumn)
% Note: aprimeIndex is [N_d*N_a*N_u,1], whereas aprimeProbs is [N_d,N_a,N_u]
%
% Creates the grid points and their 'interpolation' probabilities
% Note: aprimeIndexes is always the 'lower' point (the upper points are
% just aprimeIndexes+1, so no need to waste memory storing them), and the
% aprimeProbs are the probability of this lower point (prob of upper point
% is just 1 minus this).

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if size(aprimeFnParams(ii))~=[1,1]
        error('Using GPU for the return fn does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_d=prod(n_d);
N_a=prod(n_a);
N_u=prod(n_u);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a=length(n_a);
if l_d>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
l_u=length(n_u);

if nargin(aprimeFn)~=l_d+l_a+l_u+length(aprimeFnParams)
    error('ERROR: Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if l_d>=1
    d1vals=d_grid(1:n_d(1));
    if l_d>=2
        d2vals=shiftdim(d_grid(n_d(1)+1:sum(n_d(1:2))),-1);
        if l_d>=3
            d3vals=shiftdim(d_grid(sum(n_d(1:2))+1:sum(n_d(1:3))),-2);
            if l_d>=4
                d4vals=shiftdim(d_grid(sum(n_d(1:3))+1:sum(n_d(1:4))),-3);
            end
        end
    end
end
if l_a>=1
    a1vals=shiftdim(a_grid(1:n_a(1)),-l_d);
    if l_a>=2
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_d-1);
        if l_a>=3
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_d-2);
            if l_a>=4
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_d-3);
            end
        end
    end
end
if all(size(u_grid)==[sum(n_u),1]) % kroneker product u_grid
    if l_u>=1
        u1vals=shiftdim(u_grid(1:n_u(1)),-l_d-l_a);
        if l_u>=2
            u2vals=shiftdim(u_grid(n_u(1)+1:n_u(1)+n_u(2)),-l_d-l_a-1);
            if l_u>=3
                u3vals=shiftdim(u_grid(sum(n_u(1:2))+1:sum(n_u(1:3))),-l_d-l_a-2);
                if l_u>=4
                    u4vals=shiftdim(u_grid(sum(n_u(1:3))+1:sum(n_u(1:4))),-l_d-l_a-3);
                    if l_u>=5
                        u5vals=shiftdim(u_grid(sum(n_u(1:4))+1:sum(n_u(1:5))),-l_d-l_a-4);
                    end
                end
            end
        end
    end
elseif all(size(u_grid)==[prod(n_u),l_u]) % joint u_grid
    if l_u>=1
        u1vals=shiftdim(u_grid(:,1),-l_d-l_a);
        if l_u>=2
            u2vals=shiftdim(u_grid(:,2),-l_d-l_a);
            if l_u>=3
                u3vals=shiftdim(u_grid(:,3),-l_d-l_a);
                if l_u>=4
                    u4vals=shiftdim(u_grid(:,4),-l_d-l_a);
                    if l_u>=5
                        u5vals=shiftdim(u_grid(:,5),-l_d-l_a);
                    end
                end
            end
        end
    end
end

if l_u==1
    if l_d==1
        if l_a==1
            d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals, u1vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals, u1vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals, u1vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals,a4vals, u1vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals, ParamCell{:});
        end
    elseif l_d==2
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals, u1vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals, u1vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals, u1vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, u1vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals, ParamCell{:});
        end
    elseif l_d==3
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals, u1vals, ParamCell{:});
        elseif  l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals, u1vals, ParamCell{:});
        elseif  l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, u1vals, ParamCell{:});
        elseif  l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, u1vals, ParamCell{:});
        elseif  l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals, ParamCell{:});
        end
    elseif l_d==4
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals, u1vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, u1vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, u1vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, u1vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals, ParamCell{:});
        end
    end
elseif l_u==2
    if l_d==1
        if l_a==1
            d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals, ParamCell{:});
        end
    elseif l_d==2
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals, ParamCell{:});
        end
    elseif l_d==3
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals, u1vals,u2vals, ParamCell{:});
        elseif  l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals, u1vals,u2vals, ParamCell{:});
        elseif  l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, u1vals,u2vals, ParamCell{:});
        elseif  l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals, ParamCell{:});
        elseif  l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals, ParamCell{:});
        end
    elseif l_d==4
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals, ParamCell{:});
        end
    end
elseif l_u==3
    if l_d==1
        if l_a==1
            d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals, ParamCell{:});
        end
    elseif l_d==2
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals, ParamCell{:});
        end
    elseif l_d==3
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif  l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif  l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif  l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif  l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals, ParamCell{:});
        end
    elseif l_d==4
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals, ParamCell{:});
        end
    end
elseif l_u==4
    if l_d==1
        if l_a==1
            d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        end
    elseif l_d==2
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        end
    elseif l_d==3
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif  l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif  l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif  l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif  l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        end
    elseif l_d==4
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
        end
    end
elseif l_u==5
    if l_d==1
        if l_a==1
            d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        end
    elseif l_d==2
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        end
    elseif l_d==3
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif  l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif  l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif  l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif  l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        end
    elseif l_d==4
        if l_a==1
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==2
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==3
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==4
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        elseif l_a==5
            aprimeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals,a5vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
        end
    end
end

%% Calcuate grid indexes and probs from the values
if l_a==1
    aprimeVals=reshape(aprimeVals,[1,N_d*N_a*N_u]);

    a_griddiff=a_grid(2:end)-a_grid(1:end-1); % Distance between point and the next point
    
    temp=a_grid-aprimeVals;
    temp(temp>0)=1; % Equals 1 when a_grid is greater than aprimeVals
    
    [~,aprimeIndexes]=max(temp,[],1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
    % Note, this is going to find the 'first' grid point such that aprimeVals is smaller than or equal to that grid point
    % This is the 'upper' grid point
        
    % Switch to lower grid point index
    aprimeIndexes=aprimeIndexes-1;
    aprimeIndexes(aprimeIndexes==0)=1;
        
    % Now, find the probabilities
    aprime_residual=aprimeVals'-a_grid(aprimeIndexes);
    % Probability of the 'lower' points
    aprimeProbs=1-aprime_residual./a_griddiff(aprimeIndexes);
        
    % Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
    offTopOfGrid=(aprimeVals>=a_grid(end));
    aprimeProbs(offTopOfGrid)=0;
    % Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
    offBottomOfGrid=(aprimeVals<=a_grid(1));
    aprimeProbs(offBottomOfGrid)=1;
    
    if aprimeIndexAsColumn==1 % value fn codes without z want column
%     aprimeIndexes=reshape(aprimeIndexes,[N_d*N_a*N_u,1]);
        aprimeIndexes=aprimeIndexes'; % This is just doing the commented out reshape above
    elseif aprimeIndexAsColumn==3 % value fn with another asset uses 3
        aprimeIndexes=reshape(aprimeIndexes,[N_d*N_a,N_u]);
    else % aprimeIndexAsColumn==2 % value fn codes with z, and simulation codes want matrix 
        aprimeIndexes=reshape(aprimeIndexes,[N_d,N_a,N_u]);
    end
    aprimeProbs=reshape(aprimeProbs,[N_d,N_a,N_u]);
end


