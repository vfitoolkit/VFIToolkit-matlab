function [rprimeIndexes, rprimeProbs]=CreaterprimePolicyResidualAsset_Case1(Policy,rprimeFn, n_d, n_a, n_r, n_z, d_grid, a_grid, r_grid, z_grid, rprimeFnParamsVec)
% Note: rprimeIndex is [N_a*N_r,N_z], whereas rprimeProbs is [N_a*N_r,N_z]
%
% Creates the grid points and their 'interpolation' probabilities based on
% Policy.
% Note: rprimeIndexes is always the 'lower' point (the upper points are
% just rprimeIndexes+1, so no need to waste memory storing them), and the
% rprimeProbs are the probability of this lower point (prob of upper point
% is just 1 minus this).

ParamCell=cell(length(rprimeFnParamsVec),1);
for ii=1:length(rprimeFnParamsVec)
    if size(rprimeFnParamsVec(ii))~=[1,1]
        error('Using GPU for the return fn does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={rprimeFnParamsVec(ii)};
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_r=length(n_r);
l_z=length(n_z);

N_a=prod(n_a);
N_r=prod(n_r);
N_z=prod(n_z);

Policy=reshape(Policy,[size(Policy,1),N_a*N_r*N_z])'; % so they can then be easily used as index

if l_d>=1
    d1grid=d_grid(1:n_d(1));
    d1vals=reshape(d1grid(Policy(:,1)),[n_a,n_r,n_z]);
    if l_d>=2
        d2grid=d_grid(sum(n_d(1))+1:sum(n_d(1:2)));
        d2vals=reshape(d2grid(Policy(:,2)),[n_a,n_r,n_z]);
        if l_d>=3
            d3grid=d_grid(sum(n_d(1:2))+1:sum(n_d(1:3)));
            d3vals=reshape(d3grid(Policy(:,3)),[n_a,n_r,n_z]);
            if l_d>=4
                d4grid=d_grid(sum(n_d(1:3))+1:sum(n_d(1:4)));
                d4vals=reshape(d4grid(Policy(:,4)),[n_a,n_r,n_z]);
            end
        end
    end
end
if l_a>=1
    a1grid=a_grid(1:n_a(1));
    a1vals=a1grid;
    aprime1vals=reshape(a1grid(Policy(:,l_d+1)),[n_a,n_r,n_z]);
    if l_a>=2
        a2grid=a_grid(sum(n_a(1))+1:sum(n_a(1:2)));
        a2vals=shiftdim(a2grid,-1);
        aprime2vals=reshape(a2grid(Policy(:,l_d+2)),[n_a,n_r,n_z]);
        if l_a>=3
            a3grid=a_grid(sum(n_a(1:2))+1:sum(n_a(1:3)));
            a3vals=shiftdim(a3grid,-2);
            aprime3vals=reshape(a3grid(Policy(:,l_d+3)),[n_a,n_r,n_z]);
            if l_a>=1
                a4grid=a_grid(sum(n_a(1:3))+1:sum(n_a(1:4)));
                a4vals=shiftdim(a4grid,-3);
                aprime4vals=reshape(a4grid(Policy(:,l_d+4)),[n_a,n_r,n_z]);
            end
        end
    end
end
if l_z>=1
    z1grid=z_grid(1:n_z(1));
    z1vals=shiftdim(z1grid,-l_a-l_r);
    if l_z>=2
        z2grid=z_grid(sum(n_z(1))+1:sum(n_z(1:2)));
        z2vals=shiftdim(z2grid,-l_a-l_r-1);
        if l_z>=3
            z3grid=z_grid(sum(n_z(1:2))+1:sum(n_z(1:3)));
            z3vals=shiftdim(z3grid,-l_a-l_r-2);
            if l_z>=4
                z4grid=z_grid(sum(n_z(1:3))+1:sum(n_z(1:4)));
                z4vals=shiftdim(z4grid,-l_a-l_r-3);
            end
        end
    end
end



if l_d==0 && l_a==1 && l_z==1
    rprimeVals=arrayfun(rprimeFn, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==2
    rprimeVals=arrayfun(rprimeFn, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==3
    rprimeVals=arrayfun(rprimeFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==4
    rprimeVals=arrayfun(rprimeFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==5
    rprimeVals=arrayfun(rprimeFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==1
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==2
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==3
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==4
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==5
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==1
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==2
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==3
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==4
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==5
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==1
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==2
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==3
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==4
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==5
    rprimeVals=arrayfun(rprimeFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==1
    d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==1
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==2
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==3
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==4
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==5
    rprimeVals=arrayfun(rprimeFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
end

rprimeVals=reshape(rprimeVals,[1,N_a*N_r*N_z]);

%% Calcuate grid indexes and probs from the values
if l_r==1
    % rprimeVals=reshape(rprimeVals,[1,N_a*N_r*N_z]);

    r_griddiff=r_grid(2:end)-r_grid(1:end-1); % Distance between point and the next point
    
    temp=r_grid-rprimeVals;
    temp(temp>0)=1; % Equals 1 when a_grid is greater than aprimeVals
    
    [~,rprimeIndexes]=max(temp,[],1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
    % Note, this is going to find the 'first' grid point such that aprimeVals is smaller than or equal to that grid point
    % This is the 'upper' grid point
        
    % Switch to lower grid point index
    rprimeIndexes=rprimeIndexes-1;
    rprimeIndexes(rprimeIndexes==0)=1;
        
    % Now, find the probabilities
    rprime_residual=rprimeVals'-r_grid(rprimeIndexes);
    % Probability of the 'lower' points
    rprimeProbs=1-rprime_residual./r_griddiff(rprimeIndexes);
        
    % Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
    offTopOfGrid=(rprimeVals>=r_grid(end));
    rprimeProbs(offTopOfGrid)=0;
    % Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
    offBottomOfGrid=(rprimeVals<=r_grid(1));
    rprimeProbs(offBottomOfGrid)=1;
    
    rprimeIndexes=reshape(rprimeIndexes,[N_a*N_r,N_z]);
    rprimeProbs=reshape(rprimeProbs,[N_a*N_r,N_z,]);
else
    error('Need to implement this before you can use 2 residual assets')
end


end
