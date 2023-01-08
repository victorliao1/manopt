function [eta, Heta, print_str, stats] = trs_lanczos(problem, subprobleminput, options, storedb, key)

if nargin == 3
    % trustregions.m only wants default values for stats.
    eta = [];
    Heta = [];
    print_str = sprintf('%9s   %9s   %s', 'numinner', 'hessvec','stopreason');
    stats = struct('numinner', 0, 'hessvecevals', 0, 'limitedbyTR', false);
    return;
end

x = subprobleminput.x;
Delta = subprobleminput.Delta;
grad = subprobleminput.fgradx;

if problem.M.norm(x, grad) == 0
    eta = problem.M.zerovec(x);
    Heta = eta;
    print_str = 'Cost gradient is zero';
    stats = struct('numinner', 0, 'hessvecevals', 0, 'limitedbyTR', false);
    return;
end

M = problem.M;
inner   = @(u, v) M.inner(x, u, v);
tangent = @(u) M.tangent(x, u);
n = M.dim();

% Set local defaults here
localdefaults.kappa = 0.1;
localdefaults.theta = 0.1;
localdefaults.mininner = 1;
localdefaults.maxinner = M.dim();
% The following are here for the Newton solver called below
localdefaults.maxiter_newton = 100;
localdefaults.tol_newton = 1e-10;

% Merge local defaults with user options, if any
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

theta = options.theta;
kappa = options.kappa;

% Vectors where we keep track of the Newton root finder's work, the
% gradient norm, and the function values at each inner iteration
newton_iterations = zeros(n, 1);

% Lanczos iteratively produces an orthonormal basis of tangent vectors
% which tridiagonalize the Hessian. The corresponding tridiagonal
% matrix is preallocated here as a sparse matrix.
T = spdiags(ones(n, 3), -1:1, n, n);

% The orthonormal basis (n tangent vectors at x) is stored in this cell
Q = cell(n, 1);

% returned boolean to trustregions.m. true if we are limited by the TR
% boundary (returns boundary solution). Otherwise false.
limitedbyTR = false;

eta = M.zerovec(x);
Heta = M.zerovec(x);
r = grad;
e_Pe = 0;

% Precondition the residual.
z = getPrecon(problem, x, r, storedb, key);

% Compute z'*r.
z_r = inner(z, r);
z_r0 = z_r;

gamma_0 = sqrt(z_r);
gamma = gamma_0;
interior = true;
sigma = 1;
prevalpha = 0;

d_Pd = z_r;

% Initial search direction (we maintain -delta in memory, called mdelta, to
% avoid a change of sign of the tangent vector.)
mdelta = z;
e_Pd = 0;

% If the Hessian or a linear Hessian approximation is in use, it is
% theoretically guaranteed that the model value decreases strictly
% with each iteration of tCG. Hence, there is no need to monitor the model
% value. But, when a nonlinear Hessian approximation is used (such as the
% built-in finite-difference approximation for example), the model may
% increase. It is then important to terminate the tCG iterations and return
% the previous (the best-so-far) iterate. The variable below will hold the
% model value.
%
% This computation could be further improved based on Section 17.4.1 in
% Conn, Gould, Toint, Trust Region Methods, 2000.
% If we make this change, then also modify trustregions to gather this
% value from tCG rather than recomputing it itself.
model_fun = @(eta, Heta) inner(eta, grad) + .5*inner(eta, Heta);
model_fun_h = @(h, Hh, g) dot(h,g) + .5* dot(h, Hh);

model_value = 0;
true_model_value = 0;

first_newton_complete = false;
% Pre-assume termination because j == end.
stopreason_str = 'maximum inner iterations';

% B = tangentorthobasis(M, x);
% H = hessianmatrix(problem, x, B);

% Begin inner/tCG loop.
for j = 1 : min(options.maxinner, n) - 1
    % This call is the computationally expensive step.
    Hmdelta = getHessian(problem, x, mdelta, storedb, key);
    
    % Compute curvature (often called kappa).
    d_Hd = inner(mdelta, Hmdelta);

    % Note that if d_Hd == 0, we will exit at the next "if" anyway.
    alpha = z_r/d_Hd;

    q = M.lincomb(x, sigma/sqrt(z_r), z);
    q = tangent(q);
    Q{j} = q;

    sigma = -sign(alpha) * sigma;

    if j == 1
        T(j, j) = 1/alpha;
    else
        T(j-1, j) = gamma;     %#ok<SPRIX>
        T(j, j-1) = gamma;     %#ok<SPRIX>
        T(j, j) = 1/alpha + beta/prevalpha;  %#ok<SPRIX>
    end
        
    % Check against negative curvature and trust-region radius violation.
    % If either condition triggers, we bail out.
    if interior 
        % This will not be accurate once interior = false but it will not be
        % needed in that case anyways.
        % <neweta,neweta>_P =
        % <eta,eta>_P + 2*alpha*<eta,delta>_P + alpha*alpha*<delta,delta>_P
        e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd;
        if (d_Hd <= 0 || e_Pe_new >= Delta^2)
            interior = false;
            limitedbyTR = true;
        end
    end
    
    if interior
        e_Pe = e_Pe_new;
        % No negative curvature and eta_prop inside TR: accept it.
        new_eta  = M.lincomb(x, 1,  eta, -alpha,  mdelta);
        
        % If only a nonlinear Hessian approximation is available, this is
        % only approximately correct, but saves an additional Hessian call.
        % TODO: this computation is redundant with that of r, L241. Clean up.
        new_Heta = M.lincomb(x, 1, Heta, -alpha, Hmdelta);
        
        % Verify that the model cost decreased in going from eta to new_eta. If
        % it did not (which can only occur if the Hessian approximation is
        % nonlinear or because of numerical errors), then we return the
        % previous eta (which necessarily is the best reached so far, according
        % to the model cost). Otherwise, we accept the new eta and go on.
        new_model_value = model_fun(new_eta, new_Heta);
        if new_model_value >= model_value
            stopreason_str = 'model increased CG';
            break;
        end
        
        eta = new_eta;
        Heta = new_Heta;
        model_value = new_model_value; %% added Feb. 17, 2015
    else
%         [new_h, newton_iter,~,status] = minimize_quadratic_newton(T(1:j, 1:j), ...
%                                  gamma_0*eye(j, 1), Delta, options);
        [new_h, limitedbyTR] = TRSgep(T(1:j, 1:j), gamma_0*eye(j, 1), Delta);
%         Heta_vec = T(1:j, 1:j)*eta_vec;
        new_Hh = T(1:j, 1:j)*new_h;
        new_model_value = model_fun_h(new_h, new_Hh, gamma_0*eye(j, 1));
%         true_new_model_value = model_fun_h(eta_vec, Heta_vec, gamma_0*eye(j, 1));
%         if abs(true_new_model_value - new_model_value) > 1e-8
%             disp(status);
%             disp('gg');
%         end

%         newton_iterations(j) = newton_iter;
        h = new_h;
        first_newton_complete = true;
        model_value = new_model_value; %% added Feb. 17, 2015
    end
    
    % Update the residual.
    r = M.lincomb(x, 1, r, -alpha, Hmdelta);

    % Precondition the residual.
    z = getPrecon(problem, x, r, storedb, key);
    
    % Save the old z'*r.
    zold_rold = z_r;
    
    % Compute new z'*r.
    z_r = inner(z, r);
        
    beta = z_r/zold_rold;
    prevalpha = alpha;

    % Check kappa/theta stopping criterion.
    % Note that it is somewhat arbitrary whether to check this stopping
    % criterion on the r's (the gradients) or on the z's (the
    % preconditioned gradients). [CGT2000], page 206, mentions both as
    % acceptable criteria.
    if j >= options.mininner
        if interior && z_r <= z_r0*min(z_r0^theta, kappa)
            % Residual is small enough to quit
            if kappa < z_r0^theta
                stopreason_str = 'reached target residual-kappa (linear)';
            else
                stopreason_str = 'reached target residual-theta (superlinear)';
            end
            break;  
        elseif ~interior && (gamma * abs(h(j))) < z_r0*min(z_r0^theta, kappa)
            if kappa < z_r0^theta
                stopreason_str = 'lanczos reached target residual-kappa (linear)';
            else
                stopreason_str = 'lanczos reached target residual-theta (superlinear)';
            end
            break;
        end
    end

    gamma = sqrt(beta)/abs(prevalpha);

    % Compute new search direction.
    mdelta = M.lincomb(x, 1, z, beta, mdelta);
    
    % Since mdelta is passed to getHessian, which is the part of the code
    % we have least control over from here, we want to make sure mdelta is
    % a tangent vector up to numerical errors that should remain small.
    % For this reason, we re-project mdelta to the tangent space.
    % In limited tests, it was observed that it is a good idea to project
    % at every iteration rather than only every k iterations, the reason
    % being that loss of tangency can lead to more inner iterations being
    % run, which leads to an overall higher computational cost.
    mdelta = tangent(mdelta);

    if interior
        % Update new P-norms and P-dots [CGT2000, eq. 7.5.6 & 7.5.7].
        e_Pd = beta*(e_Pd + alpha*d_Pd);
        d_Pd = z_r + beta*beta*d_Pd;
    end
    
end  % of tCG loop

if ~interior && first_newton_complete
    % Construct the tangent vector eta as a linear combination of the basis
    % vectors
    eta = lincomb(M, x, Q(1:numel(h)), h);
end
% make sure eta is tangent up to numerical accuracy.
eta = tangent(eta);

% In principle we could avoid this call by computing an appropriate
% linear combination of available vectors. For now at least, we favor
% this numerically safer approach.
Heta = getHessian(problem, x, eta, storedb, key);

print_str = sprintf('%9d   %9d   %s', j, j, stopreason_str);

stats = struct('numinner', j, 'hessvecevals', j + 1, 'limitedbyTR', limitedbyTR);
end
