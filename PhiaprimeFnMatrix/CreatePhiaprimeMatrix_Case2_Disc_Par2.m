function Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime,Case2_Type,n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec,n_zprime,zprime_grid)
% n_zprime, zprime_grid are only inputted when z_grid varies with age. When
% there is no age dependence they are just equal to n_z and z_grid
if nargin==9
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
if l_a>3
    disp('ERROR: Using GPU for the phi_aprime fn does not yet allow for more than three of a variable (you have length(n_a)>3): (in CreatePhi_aprimeFnMatrix_Case1_Disc_Par2)')
end
if l_z>3
    disp('ERROR: Using GPU for the phi_aprime fn does not yet allow for more than three of z variable (you have length(n_z)>3): (in CreatePhi_aprimeFnMatrix_Case1_Disc_Par2)')
end

% if nargin(Phi_aprime)~=l_d+l_a+l_z+length(Phi_aprimeParamsVec)
%     disp('ERROR: Number of inputs to Phi_aprime function does not fit with size of Phi_aprimeParamNames')
% end

Phi_aprimeMatrix=zeros(1,1,'gpuArray');
if Case2_Type==1 % (d,a,z',z)
    if l_d>=1
        d1vals=d_grid(1:n_d(1));
    end
    if l_d>=2
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
    end
    if l_d>=3
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
    end
    if l_a>=1
        a1vals=shiftdim(a_grid(1:n_a(1)),-l_d);
    end
    if l_a>=2
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-l_d-1);
    end
    if l_a>=3
        a3vals=shiftdim(a_grid(n_a(1)+n_a(2)+1:n_a(1)+n_a(2)+n_a(3)),-l_d-2);
    end
    if l_z>=1
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-l_d-l_a);
        z1vals=shiftdim(zprime_grid(1:n_zprime(1)),-l_d-l_a-l_z);
    end
    if l_z>=2
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-l_d-l_a-1);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-l_d-l_a-l_z-1);
    end
    if l_z>=3
        zp3vals=shiftdim(zprime_grid(n_zprime(1)+n_zprime(2)+1:n_zprime(1)+n_zprime(2)+n_zprime(2)),-l_d-l_a-2);
        z3vals=shiftdim(z_grid(n_z(1)+n_z(2)+1:n_z(1)+n_z(2)+n_z(3)),-l_d-l_a-l_z-2);
    end
    if l_d==1 && l_a==1 && l_z==1
        d1vals(1,1,1)=d_grid(1); % Requires special treatment
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,zp1vals,z1vals,ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals,zp1vals,z1vals,ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals,a3vals,zp1vals,z1vals,ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,zp1vals,z1vals,ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime,d1vals,d2vals, a1vals,a2vals,zp1vals,z1vals,ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime,d1vals,d2vals, a1vals,a2vals,a3vals,zp1vals,z1vals,ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,zp1vals,z1vals,ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals,zp1vals,z1vals,ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,zp1vals,z1vals,ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals,a3vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a2vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a3vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals,a3vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a2vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a3vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_a,N_z,N_zprime]);
elseif Case2_Type==11  || Case2_Type==12 % (d,a,z') || (d,a,z)
    if Case2_Type==12 % The codes is all using zpvals, but for Case2_Type to these need to be zvals
        N_zprime=N_z;
        n_zprime=n_z;
        zprime_grid=z_grid;        
    end
    if l_d>=1
        d1vals=d_grid(1:n_d(1));
    end
    if l_d>=2
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
    end
    if l_d>=3
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
    end
    if l_a>=1
        a1vals=shiftdim(a_grid(1:n_a(1)),-l_d);
    end
    if l_a>=2
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-l_d-1);
    end
    if l_a>=3
        a3vals=shiftdim(a_grid(n_a(1)+n_a(2)+1:n_a(1)+n_a(2)+n_a(3)),-l_d-2);
    end
    if l_z>=1
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-l_d-l_a);
    end
    if l_z>=2
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-l_d-l_a-1);
    end
    if l_z>=3
        zp3vals=shiftdim(zprime_grid(n_zprime(1)+n_zprime(2)+1:n_zprime(1)+n_zprime(2)+n_zprime(2)),-l_d-l_a-2);
    end
    if l_d==1 && l_a==1 && l_z==1
        d1vals(1,1,1)=d_grid(1); % Requires special treatment
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,zp1vals,ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals,zp1vals,ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals,a3vals,zp1vals,ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,zp1vals,ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime,d1vals,d2vals, a1vals,a2vals,zp1vals,ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime,d1vals,d2vals, a1vals,a2vals,a3vals,zp1vals,ParamCell{:});    
    elseif l_d==3 && l_a==1 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,zp1vals,ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals,zp1vals,ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,zp1vals,ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals, zp1vals,zp2vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals, zp1vals,zp2vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals,a3vals, zp1vals,zp2vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals, zp1vals,zp2vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a2vals, zp1vals,zp2vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a2vals,a3vals, zp1vals,zp2vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals, zp1vals,zp2vals,ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals, zp1vals,zp2vals,ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, zp1vals,zp2vals,ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals, zp1vals,zp2vals,zp3vals, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals, zp1vals,zp2vals,zp3vals, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, a1vals,a2vals,a3vals, zp1vals,zp2vals,zp3vals, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals, zp1vals,zp2vals,zp3vals, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a2vals, zp1vals,zp2vals,zp3vals, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, a1vals,a2vals,a3vals, zp1vals,zp2vals,zp3vals, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals, zp1vals,zp2vals,zp3vals,ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals, zp1vals,zp2vals,zp3vals,ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, zp1vals,zp2vals,zp3vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_a,N_zprime]);
if Case2_Type==2 % (d,z',z)
    if l_d>=1
        d1vals=d_grid(1:n_d(1));
    end
    if l_d>=2
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
    end
    if l_d>=3
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
    end
    if l_z>=1
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-l_d);
        z1vals=shiftdim(zprime_grid(1:n_zprime(1)),-l_d-l_z);
    end
    if l_z>=2
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-l_d-1);
        z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-l_d-l_z-1);
    end
    if l_z>=3
        zp3vals=shiftdim(zprime_grid(n_zprime(1)+n_zprime(2)+1:n_zprime(1)+n_zprime(2)+n_zprime(2)),-l_d-2);
        z3vals=shiftdim(z_grid(n_z(1)+n_z(2)+1:n_z(1)+n_z(2)+n_z(3)),-l_d-l_z-2);
    end
    if l_d==1 && l_z==1
        d1vals(1,1,1)=d_grid(1); % Requires special treatment
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, zp1vals, z1vals,ParamCell{:});
    elseif l_d==2 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,zp1vals, z1vals,ParamCell{:});
    elseif l_d==3 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, zp1vals, z1vals,ParamCell{:});
    elseif l_d==1 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==2 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==3 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, zp1vals,zp2vals, z1vals,z2vals,ParamCell{:});
    elseif l_d==1 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_d==2 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_d==3 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, zp1vals,zp2vals,zp3vals, z1vals,z2vals,z3vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_zprime,N_z]);
elseif Case2_Type==3 % (d,z')
    if l_d>=1
        d1vals=d_grid(1:n_d(1));
    end
    if l_d>=2
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
    end
    if l_d>=3
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
    end
    if l_z>=1
        zp1vals=shiftdim(zprime_grid(1:n_zprime(1)),-l_d);
    end
    if l_z>=2
        zp2vals=shiftdim(zprime_grid(n_zprime(1)+1:n_zprime(1)+n_zprime(2)),-l_d-1);
    end
    if l_z>=3
        zp3vals=shiftdim(zprime_grid(n_zprime(1)+n_zprime(2)+1:n_zprime(1)+n_zprime(2)+n_zprime(2)),-l_d-2);
    end
    if l_d==1 && l_z==1
        d1vals(1,1,1)=d_grid(1); % Requires special treatment
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, zp1vals,ParamCell{:});
    elseif l_d==2 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, zp1vals,ParamCell{:});
    elseif l_d==3 && l_z==1
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, zp1vals,ParamCell{:});
    elseif l_d==1 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, zp1vals,zp2vals,ParamCell{:});
    elseif l_d==2 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, zp1vals,zp2vals,ParamCell{:});
    elseif l_d==3 && l_z==2
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, zp1vals,zp2vals,ParamCell{:});
    elseif l_d==1 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals, zp1vals,zp2vals,zp3vals,ParamCell{:});
    elseif l_d==2 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals, zp1vals,zp2vals,zp3vals,ParamCell{:});
    elseif l_d==3 && l_z==3
        Phi_aprimeMatrix=arrayfun(Phi_aprime, d1vals,d2vals,d3vals, zp1vals,zp2vals,zp3vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_zprime]);
end


end


