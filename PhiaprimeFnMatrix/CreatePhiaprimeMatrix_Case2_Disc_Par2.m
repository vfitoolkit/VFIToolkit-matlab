function Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime,Case2_Type,n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec,n_zprime,zprime_grid)
% n_zprime, zprime_grid are only inputted when z_grid varies with age. When
% there is no age dependence they are just equal to n_z and z_grid
if ninputs==9
    n_zprime=n_z;
    zprime_grid=z_grid;
end

ParamCell=cell(length(PhiaprimeParamsVec),1);
for ii=1:length(PhiaprimeParamsVec)
    if size(PhiaprimeParamsVec(ii))~=[1,1]
        disp('ERROR: Using GPU for the return fn does not allow for and of Phi_aprimeFnParams to be anything but a scalar')
    end
    ParamCell(ii,1)={PhiaprimeParamsVec(ii)};
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_zprime=prod(n_zprime);

l_d=length(n_d);
l_a=length(n_a); 
l_z=length(n_z);
if l_d>3
    disp('ERROR: Using GPU for the phi_aprime fn does not yet allow for more than three of d variable (you have length(n_d)>3): (in CreatePhi_aprimeFnMatrix_Case1_Disc_Par2)')
end
if l_a>2
    disp('ERROR: Using GPU for the phi_aprime fn does not yet allow for more than two of a variable (you have length(n_a)>2): (in CreatePhi_aprimeFnMatrix_Case1_Disc_Par2)')
end
if l_z>2
    disp('ERROR: Using GPU for the phi_aprime fn does not yet allow for more than two of z variable (you have length(n_z)>2): (in CreatePhi_aprimeFnMatrix_Case1_Disc_Par2)')
end

% if nargin(Phi_aprime)~=l_d+l_a+l_z+length(Phi_aprimeParamsVec)
%     disp('ERROR: Number of inputs to Phi_aprime function does not fit with size of Phi_aprimeParamNames')
% end

Phi_aprimeMatrix=zeros(1,1,'gpuArray');
if Case2_Type==1 % (d,a,z',z)
    if l_d==1 && l_a==1 && l_z==1
        dvals=d_grid; dvals(1,1,1)=d_grid(1);
        avals=shiftdim(a_grid,-1);
        zpvals=shiftdim(zprime_grid,-2);
        zvals=shiftdim(z_grid,-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, avals,zpvals,zvals,ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1
        dvals=d_grid;
        a1vals=shiftdim(a_grid(1:n_a(1)),-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
        zpvals=shiftdim(zprime_grid,-3);
        zvals=shiftdim(z_grid,-4);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, a1vals,a2vals,zpvals,zvals,ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        avals=shiftdim(a_grid,-2);
        zpvals=shiftdim(zprime_grid,-3);
        zvals=shiftdim(z_grid,-4);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, avals,zpvals,zvals,ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        a1vals=shiftdim(a_grid(1:n_a(1)),-2);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
        zpvals=shiftdim(zprime_grid,-4);
        zvals=shiftdim(z_grid,-5);
        Phi_aprimeMatrix=arrayfun(Phi_aprime,d1vals,d2vals, a1vals,a2vals,zpvals,zvals,ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        avals=shiftdim(a_grid,-3);
        zpvals=shiftdim(zprime_grid,-4);
        zvals=shiftdim(z_grid,-5);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, avals,zpvals,zvals,ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        a1vals=shiftdim(a_grid(1:n_a(1)),-3);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-4);
        zpvals=shiftdim(zprime_grid,-5);
        zvals=shiftdim(z_grid,-6);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals,zpvals,zvals,ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2
        dvals=d_grid; dvals(1,1,1)=d_grid(1);
        avals=shiftdim(a_grid,-1);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-2);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-3);
        z1vals=shiftdim(z_grid(1:n_z(1)),-4);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-5);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, avals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2
        dvals=d_grid;
        a1vals=shiftdim(a_grid(1:n_a(1)),-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-3);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-4);
        z1vals=shiftdim(z_grid(1:n_z(1)),-5);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-6);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, a1vals,a2vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        avals=shiftdim(a_grid,-2);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-3);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-4);
        z1vals=shiftdim(z_grid(1:n_z(1)),-5);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-6);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, avals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        a1vals=shiftdim(a_grid(1:n_a(1)),-2);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-4);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-5);
        z1vals=shiftdim(z_grid(1:n_z(1)),-6);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-7);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a2vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        avals=shiftdim(a_grid,-3);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-4);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-5);
        z1vals=shiftdim(z_grid(1:n_z(1)),-6);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-7);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, avals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        a1vals=shiftdim(a_grid(1:n_a(1)),-3);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-4);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-5);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-6);
        z1vals=shiftdim(z_grid(1:n_z(1)),-7);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-8);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_a,N_z,N_zprime]);
elseif Case2_Type==11  || Case2_Type==12 % (d,a,z') || (d,a,z)
    if Case2_Type==12 % The codes is all using zpvals, but for Case2_Type to these need to be zvals
        N_zprime=N_z;
        n_zprime=n_z;
        zprime_grid=z_grid;        
    end
    if l_d==1 && l_a==1 && l_z==1
        dvals=d_grid; dvals(1,1,1)=d_grid(1);
        avals=shiftdim(a_grid,-1);
        zpvals=shiftdim(zprime_grid,-2);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, avals,zpvals,ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1
        dvals=d_grid;
        a1vals=shiftdim(a_grid(1:n_a(1)),-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
        zpvals=shiftdim(zprime_grid,-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, a1vals,a2vals,zpvals,ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        avals=shiftdim(a_grid,-2);
        zpvals=shiftdim(zprime_grid,-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, avals,zpvals,ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        a1vals=shiftdim(a_grid(1:n_a(1)),-2);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
        zpvals=shiftdim(zprime_grid,-4);
        Phi_aprimeMatrix=arrayfun(Phi_aprime,d1vals,d2vals, a1vals,a2vals,zpvals,ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        avals=shiftdim(a_grid,-3);
        zpvals=shiftdim(zprime_grid,-4);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, avals,zpvals,ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        a1vals=shiftdim(a_grid(1:n_a(1)),-3);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-4);
        zpvals=shiftdim(zprime_grid,-5);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals,zpvals,ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2
        dvals=d_grid; dvals(1,1,1)=d_grid(1);
        avals=shiftdim(a_grid,-1);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-2);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, avals, zp1vals,zp2vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2
        dvals=d_grid;
        a1vals=shiftdim(a_grid(1:n_a(1)),-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-3);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-4);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, a1vals,a2vals, zp1vals,zp2vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        avals=shiftdim(a_grid,-2);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-3);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-4);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, avals, zp1vals,zp2vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        a1vals=shiftdim(a_grid(1:n_a(1)),-2);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-4);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-5);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a2vals, zp1vals,zp2vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        avals=shiftdim(a_grid,-3);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-4);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-5);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, avals, zp1vals,zp2vals,ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        a1vals=shiftdim(a_grid(1:n_a(1)),-3);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-4);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-5);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-6);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals, zp1vals,zp2vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_a,N_zprime]);
if Case2_Type==2 % (d,z',z)
    if l_d==1 && l_z==1
        dvals=d_grid; dvals(1,1,1)=d_grid(1);
        zpvals=shiftdim(zprime_grid,-1);
        zvals=shiftdim(z_grid,-2);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, zpvals, zvals,ParamCell{:});
    elseif l_d==1 && l_z==2
        dvals=d_grid;
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-1);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-2);
        z1vals=shiftdim(z_grid(1:n_z(1)),-3);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-4);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==2 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        zpvals=shiftdim(zprime_grid,-2);
        zvals=shiftdim(z_grid,-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,zpvals, zvals,ParamCell{:});
    elseif l_d==2 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-2);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-3);
        z1vals=shiftdim(z_grid(1:n_z(1)),-4);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-5);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==3 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        zpvals=shiftdim(zprime_grid,-3);
        zvals=shiftdim(z_grid,-4);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, zpvals, zvals,ParamCell{:});
    elseif l_d==3 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-3);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-4);
        z1vals=shiftdim(z_grid(1:n_z(1)),-5);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-6);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_zprime,N_z]);
elseif Case2_Type==3 % (d,z')
    if l_d==1 && l_z==1
        dvals=d_grid; dvals(1,1,1)=d_grid(1);
        zpvals=shiftdim(zprime_grid,-1);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, zpvals,ParamCell{:});
    elseif l_d==1 && l_z==2
        dvals=d_grid;
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-1);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-2);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, zp1vals,zp2vals,ParamCell{:});
    elseif l_d==2 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        zpvals=shiftdim(zprime_grid,-2);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, zpvals,ParamCell{:});
    elseif l_d==2 && l_z==2
        d1vals=d_grid(1:n_d(1),-1);
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-2);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:sum(n_zprime(1:2))),-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, zp1vals,zp2vals,ParamCell{:});
    elseif l_d==2 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-2);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, zp1vals,zp2vals,ParamCell{:});
    elseif l_d==3 && l_z==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        zpvals=shiftdim(zprime_grid,-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, zpvals,ParamCell{:});
    elseif l_d==3 && l_z==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-3);
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-4);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, zp1vals,zp2vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_zprime]);
elseif Case2_Type==4 % (d,a)
    if l_d==1 && l_a==1
        dvals=d_grid; dvals(1,1,1)=d_grid(1);
        avals=shiftdim(a_grid,-1);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, avals,ParamCell{:});
    elseif l_d==1 && l_a==2
        dvals=d_grid;
        a1vals=shiftdim(a_grid(1:n_a(1)),-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, a1vals,a2vals,ParamCell{:});
    elseif l_d==2 && l_a==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        avals=shiftdim(a_grid,-2);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, avals,ParamCell{:});
    elseif l_d==2 && l_a==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        a1vals=shiftdim(a_grid(1:n_a(1)),-2);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a2vals,ParamCell{:});
    elseif l_d==3 && l_a==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        avals=shiftdim(a_grid,-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, avals,ParamCell{:});
    elseif l_d==3 && l_a==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        a1vals=shiftdim(a_grid(1:n_a(1)),-3);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-4);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_a]);
elseif Case2_Type==5 % (d)
    if l_d==1
        dvals=d_grid; dvals(1,1,1)=d_grid(1);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, dvals, avals,ParamCell{:});
    elseif l_d==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,ParamCell{:});
    elseif l_d==3
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,1]);
end


end


