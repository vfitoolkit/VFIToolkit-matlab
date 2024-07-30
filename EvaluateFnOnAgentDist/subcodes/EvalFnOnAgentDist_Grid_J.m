function Values=EvalFnOnAgentDist_Grid_J(FnToEvaluate,CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J)
% Note: This also handles e and semiz, just put them together with the z as for this function there is no difference
% Note: using a_gridvals and z_gridvals_J

N_z=prod(n_z);
l_a=length(n_a);
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end

if l_z>0
    % assumes joint z_grid
    z_gridvals_J=shiftdim(permute(z_gridvals_J,[1,3,2]),-1); % (1,z,j,l_z)
end
% Note: CellOverAgeOfParamValues should be set up so that the age-dependence is in the third dimension if l_z>0 (second dimension if l_z=0).


if l_daprime==1
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1), a_gridvals(:,1), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1), a_gridvals(:,1),a_gridvals(:,2), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    end
elseif l_daprime==2
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2), a_gridvals(:,1), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2), a_gridvals(:,1),a_gridvals(:,2), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    end
elseif l_daprime==3
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3), a_gridvals(:,1), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3), a_gridvals(:,1),a_gridvals(:,2), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    end
elseif l_daprime==4
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3),PolicyValuesPermute(:,:,4), a_gridvals(:,1), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3),PolicyValuesPermute(:,:,4), a_gridvals(:,1),a_gridvals(:,2), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3),PolicyValuesPermute(:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3),PolicyValuesPermute(:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    end
elseif l_daprime==5
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3),PolicyValuesPermute(:,:,4),PolicyValuesPermute(:,:,5), a_gridvals(:,1), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==1 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3),PolicyValuesPermute(:,:,4),PolicyValuesPermute(:,:,5), a_gridvals(:,1),a_gridvals(:,2), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==2 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3),PolicyValuesPermute(:,:,4),PolicyValuesPermute(:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==3 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,1),PolicyValuesPermute(:,:,2),PolicyValuesPermute(:,:,3),PolicyValuesPermute(:,:,4),PolicyValuesPermute(:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==5
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5), CellOverAgeOfParamValues{:});
    elseif l_a==4 && l_z==6
        Values=arrayfun(FnToEvaluate, PolicyValuesPermute(:,:,:,1),PolicyValuesPermute(:,:,:,2),PolicyValuesPermute(:,:,:,3),PolicyValuesPermute(:,:,:,4),PolicyValuesPermute(:,:,:,5), a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals_J(1,:,:,1),z_gridvals_J(1,:,:,2),z_gridvals_J(1,:,:,3),z_gridvals_J(1,:,:,4),z_gridvals_J(1,:,:,5),z_gridvals_J(1,:,:,6), CellOverAgeOfParamValues{:});
    end
end

% Note: No need to reshape the output as it is already the appropriate size
% if N_z==0
%     Values=reshape(Values,[N_a,N_j]);
% else
%     Values=reshape(Values,[N_a,N_z,N_j]);
% end



end