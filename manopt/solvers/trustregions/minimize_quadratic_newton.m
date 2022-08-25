function [s, iter, lambda, status] = minimize_quadratic_newton(H, g, Delta, options)
    n = size(H, 1);
        
    % Compute the smallest eigenvalue of H, as we know the target lambda
    % must be at least as large as the negative of that, so that the
    % shifted H will be positive semidefinite.
    % 
    % Since H ought to be sparse and tridiagonal, and since we only need
    % its smallest eigenvalue, this computation could be sped up
    % significantly. It does not appear to be a bottleneck, and eig is
    % simple and reliable, so we keep this for now.

    [eigenvec_min, lambda_min] = eigs(H, 1, 'smallestreal');

    left_barrier = max(0, -lambda_min);
    if left_barrier == 0
        lambda = 0;
    else
        lambda = -lambda_min + (1-lambda_min)*sqrt(eps(1));
    end
    H_shifted = H + lambda*speye(n);

    % Counter 'iter' holds the number of fully executed Newton iterations.
    iter = 0;

    s = -(H_shifted\g);
    snorm = norm(s);
    if snorm <= Delta
        if ~(lambda == 0) && ~(snorm == Delta)
            alpha = max(real(roots([snorm, 2* eigenvec_min.*s, norm(eigenvec_min)-Delta^2])));
            s = s + alpha * eigenvec_min;
        end
        
        return;
    end
    
    while true

        if iter >= options.maxiter_newton
            % Iterations exceeded maximum number allowed.
            status = -1;
            return;
        end

        norm_del_diff = snorm - Delta;

        % Check if it is close enough to zero to stop.
        if abs(snorm - Delta) <= options.tol_newton*Delta
            status = 0;
            return;
        end

        % factorize H_shifted with cholesky factorization
        L = chol(H_shifted);

        w = L \ s;

        del_lambda = ((snorm - Delta)/Delta) * (snorm^2/norm(w)^2);
        
        % If the Newton step would bring us left of the left barrier, jump
        % instead to the midpoint between the left barrier and the current
        % lambda.
        if lambda + del_lambda <= left_barrier
            del_lambda = -.5*(lambda - left_barrier);
        end
        
        % If the step is so small that it numerically does not make a
        % difference when added to the current lambda, we stop.
        if abs(del_lambda) <= eps(lambda)
            status = 1;
            return;
        end

        lambda = lambda + del_lambda;
        H_shifted = H_shifted + del_lambda*speye(n);

        s = -(H_shifted\g);
        snorm = norm(s);

        if options.verbosity >= 6
            fprintf(['lambda %.12e, ||s|| %.12e, ||s|| - Delta %.12e' ...
                     '\n\n'], lambda, snorm, norm_del_diff);
        end
        iter = iter + 1;
    end
end
