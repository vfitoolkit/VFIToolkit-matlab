function Phi_aprimeMatrix=CreatePhi_aprimeFnMatrix_Case2_Disc_Par2(Phi_aprimeFn,n_d, n_a, n_z, d_grid, a_grid, z_grid,Phi_aprimeFnParams,Case2_Type)
% CreatePhi_aprimeFnMatrix_Case2_Disc_Par2(Phi_aprimeFn, n_d, n_a, 0, d_grid, a_grid,0,Phi_aprimeFnParams,Case2_Type);
%If there is no d variable, just input n_d=0 and d_grid=0

ParamCell=cell(length(Phi_aprimeFnParams),1);
for ii=1:length(Phi_aprimeFnParams)
    if size(Phi_aprimeFnParams(ii))~=[1,1]
        disp('ERROR: Using GPU for the return fn does not allow for and of Phi_aprimeFnParams to be anything but a scalar')
    end
    ParamCell(ii,1)={Phi_aprimeFnParams(ii)};
end

% No need to input ReturnFn as it has been created as an m-file called
% TempReturnFn.m

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

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

if nargin(Phi_aprimeFn)~=l_d+l_a+l_z+length(Phi_aprimeFnParams)
    disp('ERROR: Number of inputs to Phi_aprimeFn does not fit with size of Phi_aprimeFnParams')
end

Phi_aprimeMatrix=zeros(1,1,'gpuArray');
if Case2Type==1
    disp('ERROR: CreatePhi_aprimeFnMatrix_Case2_Disc_Par2 not yet implemented for Case2Type=1')
elseif Case2Type==2
    disp('ERROR: CreatePhi_aprimeFnMatrix_Case2_Disc_Par2 not yet implemented for Case2Type=1')
elseif Case2Type==3
    disp('ERROR: CreatePhi_aprimeFnMatrix_Case2_Disc_Par2 not yet implemented for Case2Type=1')
elseif Case2Type==4 % (d,a)
    if l_d==1 && l_a==1
        dvals=d_grid; dvals(1,1,1)=d_grid(1);
        avals=shiftdim(a_grid,-1);
        Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, dvals, avals,ParamCell{:});
    elseif l_d==1 && l_a==2
        dvals=d_grid;
        a1vals=shiftdim(a_grid(1:n_a(1)),-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
        Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, dvals, a1vals,a2vals,ParamCell{:});
    elseif l_d==2 && l_a==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        avals=shiftdim(a_grid,-2);
        Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals, avals,ParamCell{:});
    elseif l_d==2 && l_a==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        a1vals=shiftdim(a_grid(1:n_a(1)),-2);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals, a1vals,a2vals,ParamCell{:});
    elseif l_d==3 && l_a==1
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        avals=shiftdim(a_grid,-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals,d3vals, avals,ParamCell{:});
    elseif l_d==3 && l_a==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
        a1vals=shiftdim(a_grid(1:n_a(1)),-2);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
        Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals,ParamCell{:});
    end
    Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_a]);
end

%% General version
% 
% if l_d==1 && l_a==1 && l_z==1
%     dvals=d_grid; dvals(1,1,1)=d_grid(1);
%     avals=shiftdim(a_grid,-1);
%     zvals=shiftdim(z_grid,-2);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, dvals, avals, zvals,ParamCell{:});
% elseif l_d==1 && l_a==1 && l_z==2
%     dvals=d_grid;
%     avals=shiftdim(a_grid,-1);
%     z1vals=shiftdim(z_grid(1:n_z(1)),-2);
%     z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-3);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, dvals, avals, z1vals,z2vals,ParamCell{:});
% elseif l_d==1 && l_a==2 && l_z==1
%     dvals=d_grid;
%     a1vals=shiftdim(a_grid(1:n_a(1)),-1);
%     a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
%     zvals=shiftdim(z_grid,-3);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, dvals, a1vals,a2vals, zvals,ParamCell{:});
% elseif l_d==1 && l_a==2 && l_z==2
%     dvals=d_grid;
%     a1vals=shiftdim(a_grid(1:n_a(1)),-1);
%     a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
%     z1vals=shiftdim(z_grid(1:n_z(1)),-3);
%     z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-4);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, dvals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
% elseif l_d==2 && l_a==1 && l_z==1
%     d1vals=d_grid(1:n_d(1));
%     d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
%     avals=shiftdim(a_grid,-2);
%     zvals=shiftdim(z_grid,-3);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals, avals, zvals,ParamCell{:});
% elseif l_d==2 && l_a==1 && l_z==2
%     d1vals=d_grid(1:n_d(1));
%     d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
%     avals=shiftdim(a_grid,-2);
%     z1vals=shiftdim(z_grid(1:n_z(1)),-3);
%     z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-4);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals, avals, z1vals,z2vals,ParamCell{:});
% elseif l_d==2 && l_a==2 && l_z==1
%     d1vals=d_grid(1:n_d(1));
%     d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
%     a1vals=shiftdim(a_grid(1:n_a(1)),-2);
%     a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
%     zvals=shiftdim(z_grid,-4);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals, a1vals,a2vals, zvals,ParamCell{:});
% elseif l_d==2 && l_a==2 && l_z==2
%     d1vals=d_grid(1:n_d(1));
%     d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
%     a1vals=shiftdim(a_grid(1:n_a(1)),-2);
%     a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
%     z1vals=shiftdim(z_grid(1:n_z(1)),-4);
%     z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-5);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
% elseif l_d==3 && l_a==1 && l_z==1
%     d1vals=d_grid(1:n_d(1));
%     d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
%     d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
%     avals=shiftdim(a_grid,-3);
%     zvals=shiftdim(z_grid,-4);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals,d3vals, avals, zvals,ParamCell{:});
% elseif l_d==3 && l_a==1 && l_z==2
%     d1vals=d_grid(1:n_d(1));
%     d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
%     d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
%     avals=shiftdim(a_grid,-2);
%     z1vals=shiftdim(z_grid(1:n_z(1)),-3);
%     z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-4);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals,d3vals, avals, z1vals,z2vals,ParamCell{:});
% elseif l_d==3 && l_a==2 && l_z==1
%     d1vals=d_grid(1:n_d(1));
%     d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
%     d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
%     a1vals=shiftdim(a_grid(1:n_a(1)),-2);
%     a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
%     zvals=shiftdim(z_grid,-4);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals, zvals,ParamCell{:});
% elseif l_d==3 && l_a==2 && l_z==2
%     d1vals=d_grid(1:n_d(1));
%     d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
%     d3vals=shiftdim(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),-2);
%     a1vals=shiftdim(a_grid(1:n_a(1)),-2);
%     a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
%     z1vals=shiftdim(z_grid(1:n_z(1)),-4);
%     z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-5);
%     Phi_aprimeMatrix=arrayfun(Phi_aprimeFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
% end
% 
% Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d,N_a,N_z]);



end


