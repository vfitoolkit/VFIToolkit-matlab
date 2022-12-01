function Fmatrix=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG_noz(ReturnFn, n_d, n_a, N_j, d_grid, a_grid, ReturnFnParamsAgeMatrix)
%If there is no d variable, just input n_d=0 and d_grid=0

N_d=prod(n_d);
N_a=prod(n_a);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a=length(n_a); 
if l_d>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4): (in CreateReturnFnMatrix_Case1_Disc_Par2)')
end
if l_a>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4): (in CreateReturnFnMatrix_Case1_Disc_Par2)')
end

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);

if nargin(ReturnFn)~=l_d+l_a+l_a+nReturnFnParams
    disp('ERROR: Number of inputs to ReturnFn does not fit with size of ReturnFnParams')
end

ParamCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
%     if ~prod(size(ReturnFnParamsAgeMatrix(:,ii))==[N_j,1]) % ~prod(size(ReturnFnParamsAgeMatrix(:,ii))==[1,1]) && 
%         disp('ERROR: Using GPU for the fastOLG return fn does not allow for any of ReturnFnParams to be anything but a vector dependent on age)')
%     end
    ParamCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-l_d-l_a-l_a)};
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
    aprime1vals=shiftdim(a_grid(1:n_a(1)),-l_d);
    a1vals=shiftdim(a_grid(1:n_a(1)),-l_a-l_d);
    if l_a>=2
        aprime2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_d-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_a-l_d-1);
        if l_a>=3
            aprime3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_d-2);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_a-l_d-2);
            if l_a>=4
                aprime4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_d-3);
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_a-l_d-3);
            end
        end
    end
end

% Note that the dimensions of the following are d-aprime-a-j

if l_d==0 && l_a==1 
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, ParamCell{:});
elseif l_d==0 && l_a==2 
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, ParamCell{:});
elseif l_d==0 && l_a==3 
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==0 && l_a==4 
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==1 && l_a==1 
    d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, ParamCell{:});
elseif l_d==1 && l_a==2 
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, ParamCell{:});
elseif l_d==1 && l_a==3 
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==1 && l_a==4 
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==2 && l_a==1 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, ParamCell{:});
elseif l_d==2 && l_a==2 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, ParamCell{:});
elseif l_d==2 && l_a==3 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==2 && l_a==4 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==3 && l_a==1 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, ParamCell{:});
elseif l_d==3 && l_a==2 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, ParamCell{:});
elseif l_d==3 && l_a==3 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==3 && l_a==4 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==4 && l_a==1 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, ParamCell{:});
elseif l_d==4 && l_a==2 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, ParamCell{:});
elseif l_d==4 && l_a==3 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==4 && l_a==4 
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
end

if l_d==0
    Fmatrix=reshape(Fmatrix,[N_a,N_a,N_j]);
else
    Fmatrix=reshape(Fmatrix,[N_d*N_a,N_a,N_j]);
end


end


