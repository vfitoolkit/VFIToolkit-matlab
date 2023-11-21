function Values=EvalFnOnSimPanelIndex(FnToEvaluate,ParamCell,daprime_val,a_val,z_val,l_daprime,l_a,l_z)
% Note: This also handles e and semiz, just put them together with the z as for this function there is no difference
% Note: z_grid needs to be a joint-grid

if l_daprime>=1
    daprime1vals=daprime_val(:,1);
    if l_daprime>=2
        daprime2vals=daprime_val(:,2);
        if l_daprime>=3
            daprime3vals=daprime_val(:,3);
            if l_daprime>=4
                daprime4vals=daprime_val(:,4);
                if l_daprime>=5
                    daprime5vals=daprime_val(:,5);
                end
            end
        end
    end
end


if l_daprime==1
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1), ParamCell{:});
    elseif l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1), z_val(:,1), ParamCell{:});
    elseif l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2), ParamCell{:});
    elseif l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2), z_val(:,1), ParamCell{:});
    elseif l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2),a_val(:,3), ParamCell{:});
    elseif l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1), ParamCell{:});
    elseif l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), ParamCell{:});
    elseif l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1), ParamCell{:});
    elseif l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    end
elseif l_daprime==2
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1), ParamCell{:});
    elseif l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1), z_val(:,1), ParamCell{:});
    elseif l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2), ParamCell{:});
    elseif l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2), z_val(:,1), ParamCell{:});
    elseif l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2),a_val(:,3), ParamCell{:});
    elseif l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1), ParamCell{:});
    elseif l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), ParamCell{:});
    elseif l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1), ParamCell{:});
    elseif l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    end
elseif l_daprime==3
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1), ParamCell{:});
    elseif l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1), z_val(:,1), ParamCell{:});
    elseif l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2), ParamCell{:});
    elseif l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2), z_val(:,1), ParamCell{:});
    elseif l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2),a_val(:,3), ParamCell{:});
    elseif l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1), ParamCell{:});
    elseif l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), ParamCell{:});
    elseif l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1), ParamCell{:});
    elseif l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    end
elseif l_daprime==4
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1), ParamCell{:});
    elseif l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1), z_val(:,1), ParamCell{:});
    elseif l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2), ParamCell{:});
    elseif l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2), z_val(:,1), ParamCell{:});
    elseif l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2),a_val(:,3), ParamCell{:});
    elseif l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1), ParamCell{:});
    elseif l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), ParamCell{:});
    elseif l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1), ParamCell{:});
    elseif l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    end
elseif l_daprime==5
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1), ParamCell{:});
    elseif l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1), z_val(:,1), ParamCell{:});
    elseif l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2), ParamCell{:});
    elseif l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2), z_val(:,1), ParamCell{:});
    elseif l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2),a_val(:,3), ParamCell{:});
    elseif l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1), ParamCell{:});
    elseif l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2),a_val(:,3), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    elseif l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), ParamCell{:});
    elseif l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1), ParamCell{:});
    elseif l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2), ParamCell{:});
    elseif l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2),z_val(:,3), ParamCell{:});
    elseif l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_val(:,1),a_val(:,2),a_val(:,3),a_val(:,4), z_val(:,1),z_val(:,2),z_val(:,3),z_val(:,4), ParamCell{:});
    end
end

% Note: No need to reshape the output as it is already the appropriate size
% if N_z==0
%     Values=reshape(Values,[N_a,1]);
% else
%     Values=reshape(Values,[N_a,N_z]);
% end



end