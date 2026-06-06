function [n_z_temp,z_grid_temp,pi_z_temp,options_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,pi_z,options_temp,structordim)
% Shock dependence on ptype can be done as structure, or as a trailing dimension of length N_i.
%
% structordim controls which forms are resolved:
%   1 = struct only (lenient: pass through if struct lacks the field)
%   2 = trailing-dim only (pass struct through)
%   3 = both, struct strict (matches single-level PType convention)
%
% Pass options_temp positionally as either vfoptions_temp (ValueFnIter) or
% simoptions_temp (StationaryDist, AggVars, AllStats, LifeCycleProfiles).

%% n_z (no meaningful trailing-dim form, so struct-only regardless of mode)
if structordim==1
    if isstruct(n_z) && isfield(n_z,iistr)
        n_z_temp=n_z.(iistr);
    else
        n_z_temp=n_z;
    end
elseif structordim==2
    n_z_temp=n_z;
elseif structordim==3
    if isstruct(n_z)
        n_z_temp=n_z.(iistr);
    else
        n_z_temp=n_z;
    end
end

%% z_grid
if structordim==1
    if isstruct(z_grid) && isfield(z_grid,iistr)
        z_grid_temp=z_grid.(iistr);
    else
        z_grid_temp=z_grid;
    end
elseif structordim==2
    if isstruct(z_grid)
        z_grid_temp=z_grid;
    else
        nn=size(z_grid,ndims(z_grid));
        if nn==N_i
            otherdims=repmat({':'},1,ndims(z_grid)-1);
            z_grid_temp=z_grid(otherdims{:},ii);
        else
            z_grid_temp=z_grid;
        end
    end
elseif structordim==3
    if isstruct(z_grid)
        z_grid_temp=z_grid.(iistr);
    else
        nn=size(z_grid,ndims(z_grid));
        if nn==N_i
            otherdims=repmat({':'},1,ndims(z_grid)-1);
            z_grid_temp=z_grid(otherdims{:},ii);
        else
            z_grid_temp=z_grid;
        end
    end
end

%% pi_z
if structordim==1
    if isstruct(pi_z) && isfield(pi_z,iistr)
        pi_z_temp=pi_z.(iistr);
    else
        pi_z_temp=pi_z;
    end
elseif structordim==2
    if isstruct(pi_z)
        pi_z_temp=pi_z;
    else
        nn=size(pi_z,ndims(pi_z));
        if nn==N_i
            otherdims=repmat({':'},1,ndims(pi_z)-1);
            pi_z_temp=pi_z(otherdims{:},ii);
        else
            pi_z_temp=pi_z;
        end
    end
elseif structordim==3
    if isstruct(pi_z)
        pi_z_temp=pi_z.(iistr);
    else
        nn=size(pi_z,ndims(pi_z));
        if nn==N_i
            otherdims=repmat({':'},1,ndims(pi_z)-1);
            pi_z_temp=pi_z(otherdims{:},ii);
        else
            pi_z_temp=pi_z;
        end
    end
end

%% e
% e_grid/pi_e: PType_Options/PType_Options_2L already resolved any struct
% form, so structordim==3 only applies the trailing-dim slice (struct case
% is a silent passthrough, preserving prior single-level behavior).
if prod(options_temp.n_e)>0
    % e_grid
    if structordim==1
        if isstruct(options_temp.e_grid) && isfield(options_temp.e_grid,iistr)
            options_temp.e_grid=options_temp.e_grid.(iistr);
        end
    elseif structordim==2
        if ~isstruct(options_temp.e_grid)
            nn=size(options_temp.e_grid,ndims(options_temp.e_grid));
            if nn==N_i
                otherdims=repmat({':'},1,ndims(options_temp.e_grid)-1);
                options_temp.e_grid=options_temp.e_grid(otherdims{:},ii);
            end
        end
    elseif structordim==3
        if ~isstruct(options_temp.e_grid)
            nn=size(options_temp.e_grid,ndims(options_temp.e_grid));
            if nn==N_i
                otherdims=repmat({':'},1,ndims(options_temp.e_grid)-1);
                options_temp.e_grid=options_temp.e_grid(otherdims{:},ii);
            end
        end
    end
    % pi_e
    if structordim==1
        if isstruct(options_temp.pi_e) && isfield(options_temp.pi_e,iistr)
            options_temp.pi_e=options_temp.pi_e.(iistr);
        end
    elseif structordim==2
        if ~isstruct(options_temp.pi_e)
            nn=size(options_temp.pi_e,ndims(options_temp.pi_e));
            if nn==N_i
                otherdims=repmat({':'},1,ndims(options_temp.pi_e)-1);
                options_temp.pi_e=options_temp.pi_e(otherdims{:},ii);
            end
        end
    elseif structordim==3
        if ~isstruct(options_temp.pi_e)
            nn=size(options_temp.pi_e,ndims(options_temp.pi_e));
            if nn==N_i
                otherdims=repmat({':'},1,ndims(options_temp.pi_e)-1);
                options_temp.pi_e=options_temp.pi_e(otherdims{:},ii);
            end
        end
    end
end

%% semiz
if prod(options_temp.n_semiz)>0
    % semiz_grid
    if structordim==1
        if isstruct(options_temp.semiz_grid) && isfield(options_temp.semiz_grid,iistr)
            options_temp.semiz_grid=options_temp.semiz_grid.(iistr);
        end
    elseif structordim==2
        if ~isstruct(options_temp.semiz_grid)
            nn=size(options_temp.semiz_grid,ndims(options_temp.semiz_grid));
            if nn==N_i
                otherdims=repmat({':'},1,ndims(options_temp.semiz_grid)-1);
                options_temp.semiz_grid=options_temp.semiz_grid(otherdims{:},ii);
            end
        end
    elseif structordim==3
        if ~isstruct(options_temp.semiz_grid)
            nn=size(options_temp.semiz_grid,ndims(options_temp.semiz_grid));
            if nn==N_i
                otherdims=repmat({':'},1,ndims(options_temp.semiz_grid)-1);
                options_temp.semiz_grid=options_temp.semiz_grid(otherdims{:},ii);
            end
        end
    end
    % pi_semiz (optional field — only present if model uses an explicit transition rather than SemiExoShockFn)
    if isfield(options_temp,'pi_semiz')
        if structordim==1
            if isstruct(options_temp.pi_semiz) && isfield(options_temp.pi_semiz,iistr)
                options_temp.pi_semiz=options_temp.pi_semiz.(iistr);
            end
        elseif structordim==2
            if ~isstruct(options_temp.pi_semiz)
                nn=size(options_temp.pi_semiz,ndims(options_temp.pi_semiz));
                if nn==N_i
                    otherdims=repmat({':'},1,ndims(options_temp.pi_semiz)-1);
                    options_temp.pi_semiz=options_temp.pi_semiz(otherdims{:},ii);
                end
            end
        elseif structordim==3
            if ~isstruct(options_temp.pi_semiz)
                nn=size(options_temp.pi_semiz,ndims(options_temp.pi_semiz));
                if nn==N_i
                    otherdims=repmat({':'},1,ndims(options_temp.pi_semiz)-1);
                    options_temp.pi_semiz=options_temp.pi_semiz(otherdims{:},ii);
                end
            end
        end
    end
end

end
