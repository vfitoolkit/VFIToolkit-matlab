function [vfoptions,simoptions]=SetupNonStandardEndoStates_InfHorz_TPath(n_d,n_a,d_grid,a_grid,vfoptions,simoptions)

%% experienceasset
if vfoptions.experienceasset==1
    % To be able to pass them to internal commands, we store all the setup in vfoptions and simoptions.
    if ~isfield(vfoptions,'l_dexperienceasset')
        vfoptions.l_dexperienceasset=1; % by default, only one decision variable influences the experienceasset
    end
    vfoptions.l_d2=vfoptions.l_dexperienceasset;
    
    % Split decision variables into the standard ones and the one relevant to the experience asset
    if length(n_d)>vfoptions.l_d2
        n_d1=n_d(1:end-vfoptions.l_d2);
    else
        n_d1=0;
    end
    n_d2=n_d(end-vfoptions.l_d2+1:end); % n_d2 is the decision variable that influences next period vale of the experience asset
    % d1_grid=d_grid(1:sum(n_d1));
    d2_grid=d_grid(sum(n_d1)+1:end);
    % Split endogenous assets into the standard ones and the experience asset
    if isscalar(n_a)
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_a2=n_a(end); % n_a2 is the experience asset
    a1_grid=a_grid(1:sum(n_a1));
    a2_grid=a_grid(sum(n_a1)+1:end);

    if isfield(vfoptions,'aprimeFn')
        aprimeFn=vfoptions.aprimeFn;
    else
        error('To use an experience asset you must define vfoptions.aprimeFn')
    end

    % aprimeFnParamNames in same fashion
    l_d2=length(n_d2);
    l_a2=length(n_a2);
    temp=getAnonymousFnInputNames(aprimeFn);
    if length(temp)>(l_d2+l_a2)
        aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
    else
        aprimeFnParamNames={};
    end

    % N_d1=prod(n_d1);
    N_a1=prod(n_a1);

    % Note: divide-and-conquer is only possible with a1
    if N_a1>0 % set up for divide-and-conquer
        if vfoptions.divideandconquer==1
            if ~isfield(vfoptions,'level1n')
                vfoptions.level1n=max(ceil(n_a1(1)/50),5); % minimum of 5
                if n_a1(1)<5
                    error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
                end
                if vfoptions.verbose==1
                    fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
                end
            end
            vfoptions.level1n=min(vfoptions.level1n,n_a1); % Otherwise causes errors
        end
    end

    if N_a1>0
        a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
    else
        a1_gridvals=[];
    end
    d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
    % d_gridvals is anyway created
    % if N_d1>0
    %     d_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
    % else
    %     d_gridvals=[]; % not used
    % end

    % Now, store all of these in vfoptions and simoptions, as appropriate
    vfoptions.setup_experienceasset.l_dexperienceasset=vfoptions.l_dexperienceasset;
    vfoptions.setup_experienceasset.n_d1=n_d1;
    vfoptions.setup_experienceasset.n_d2=n_d2;
    vfoptions.setup_experienceasset.n_a1=n_a1;
    vfoptions.setup_experienceasset.n_a2=n_a2;
    vfoptions.setup_experienceasset.d2_gridvals=d2_gridvals;
    vfoptions.setup_experienceasset.a1_gridvals=a1_gridvals;
    vfoptions.setup_experienceasset.a2_grid=a2_grid;
    vfoptions.setup_experienceasset.aprimeFn=aprimeFn; % Note: bit silly, as is already in vfoptions.aprimeFn, but just makes things easier to remember as everything is vfoptions.setup_experienceasset
    vfoptions.setup_experienceasset.aprimeFnParamNames=aprimeFnParamNames;

    simoptions.setup_experienceasset.aprimeFn=aprimeFn; % Note: bit silly, as is already in simoptions.aprimeFn, but just makes things easier to remember as everything is simoptions.setup_experienceasset
    simoptions.setup_experienceasset.aprimeFnParamNames=aprimeFnParamNames;
    simoptions.setup_experienceasset.l_dexperienceasset=vfoptions.l_dexperienceasset;
    simoptions.setup_experienceasset.n_a1=n_a1;
    simoptions.setup_experienceasset.N_a1=N_a1;
    simoptions.setup_experienceasset.n_a2=n_a2;
    simoptions.setup_experienceasset.a2_grid=a2_grid;
    simoptions.setup_experienceasset.d_grid=d_grid; % Note: bit silly, as is already in simoptions.d_grid
end