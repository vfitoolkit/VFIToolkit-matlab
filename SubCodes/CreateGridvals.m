function x_gridvals=CreateGridvals(n_x,x_grid,MatrixOrCell)
% Creates the 'gridvals' versions of the standard grids. These allow for
% easier evaluation of functions on the grids via the EvalFnOnAgentDist
% commands.
% x here stands for any one of d,a,or z.
% For MatrixOrCell=1, output takes form of matrices
% For MatrixOrCell=2, output takes form of cells.
%
% x_gridvals contain N_x rows, and (e.g., for x_gridvals) the columns for a given row contain
% all the values of all the 'a' variables. (ie. x_gridvals is N_a-by-l_a)
% These contain no more information than the standard grid format (e.g.,
% x_grid), but are substantially larger (use more memory), however for
% certain purposes they are much easier to use quickly or in parallel.

l_x=length(n_x);
% N_x=prod(n_x);

% Create x_gridvals.
if l_x==1
    x_gridvals=x_grid;
else
    if l_x==2
        x_gridvals=[repmat(x_grid(1:n_x(1)),n_x(2),1), repelem(x_grid(n_x(1)+1:end),n_x(1),1)];
    elseif l_x==3
        x_gridvals=[repmat(x_grid(1:n_x(1)),n_x(2)*n_x(3),1), repmat(repelem(x_grid(n_x(1)+1:n_x(1)+n_x(2)),n_x(1),1),n_x(3),1), repelem(x_grid(n_x(1)+n_x(2)+1:end),n_x(1)*n_x(2),1)];
    elseif l_x==4
        x_gridvals=[repmat(x_grid(1:n_x(1)),n_x(2)*n_x(3)*n_x(4),1), repmat(repelem(x_grid(n_x(1)+1:n_x(1)+n_x(2)),n_x(1),1),n_x(3)*n_x(4),1), repmat(repelem(x_grid(n_x(1)+n_x(2)+1:n_x(1)+n_x(2)+n_x(3)),n_x(1)*n_x(2),1),n_x(4),1), repelem(x_grid(n_x(1)+n_x(2)+n_x(3)+1:end),n_x(1)*n_x(2)*n_x(3),1)];
    elseif l_x==5
        x_gridvals=[repmat(x_grid(1:n_x(1)),n_x(2)*n_x(3)*n_x(4)*n_x(5),1), repmat(repelem(x_grid(n_x(1)+1:n_x(1)+n_x(2)),n_x(1),1),n_x(3)*n_x(4)*n_x(5),1), repmat(repelem(x_grid(n_x(1)+n_x(2)+1:n_x(1)+n_x(2)+n_x(3)),n_x(1)*n_x(2),1),n_x(4)*n_x(5),1), repmat(repelem(x_grid(n_x(1)+n_x(2)+n_x(3)+1:n_x(1)+n_x(2)+n_x(3)+n_x(4)),n_x(1)*n_x(2),1),n_x(4)*n_x(5),1), repelem(x_grid(n_x(1)+n_x(2)+n_x(3)+n_x(4)+1:end),n_x(1)*n_x(2)*n_x(3)*n_x(4),1)];
    else
        error('Cannot handle length(n_x)>5. Please email me if you need this functionality')
    end
end
if MatrixOrCell==2
    x_gridvals=num2cell(x_gridvals);
end



end
