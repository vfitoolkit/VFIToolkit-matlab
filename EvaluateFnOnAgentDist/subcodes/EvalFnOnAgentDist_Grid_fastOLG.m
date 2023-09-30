function Values=EvalFnOnAgentDist_Grid_fastOLG(FnToEvaluate,ParamCell,PolicyValues,l_d,l_a,l_z,a_gridvals,z_gridvals)

% PolicyValues is [N_a,1,N_z,l_d+l_aprime]
% a_gridvals is [N_a,l_a]
% z_gridvals [1,1,N_z,l_z]

% Values is created as [N_a,1,N_z]

if l_d==0 && l_a==1 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1), a_gridvals(:,1), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1), a_gridvals(:,1), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});    
elseif l_d==0 && l_a==1 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1), a_gridvals(:,1), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1), a_gridvals(:,1), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});  
elseif l_d==0 && l_a==3 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2), a_gridvals(:,1), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2), a_gridvals(:,1), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2), a_gridvals(:,1), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2), a_gridvals(:,1), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3), a_gridvals(:,1), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6), a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==1
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7),PolicyValues(:,:,:,8), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1), ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==2
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7),PolicyValues(:,:,:,8), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==3
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7),PolicyValues(:,:,:,8), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==4
    Values=arrayfun(FnToEvaluate, PolicyValues(:,:,:,1),PolicyValues(:,:,:,2),PolicyValues(:,:,:,3),PolicyValues(:,:,:,4),PolicyValues(:,:,:,5),PolicyValues(:,:,:,6),PolicyValues(:,:,:,7),PolicyValues(:,:,:,8), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), ParamCell{:});
end

% Values is [N_a,1,N_z]

end