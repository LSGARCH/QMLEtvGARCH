function tvGARCH21_local_estimation
    rng(12345);

    Ns = [1000, 2000, 5000];
    for k = 1:length(Ns)
        N = Ns(k);
        [x, u] = simulate_tvgarch21(N);
        u_points = linspace(0.02, 0.98, 50);
        h = 0.08;

        % Estimated parameters
        omega_hat_lap = zeros(size(u_points));
        alpha1_hat_lap = zeros(size(u_points));
        alpha2_hat_lap = zeros(size(u_points));
        beta_hat_lap   = zeros(size(u_points));

        omega_hat_gauss = zeros(size(u_points));
        alpha1_hat_gauss = zeros(size(u_points));
        alpha2_hat_gauss = zeros(size(u_points));
        beta_hat_gauss   = zeros(size(u_points));

        for i = 1:length(u_points)
            up = u_points(i);
            w = gaussian_kernel((u - up)/h);
            w = w(:); w = w / sum(w);
            x = x(:);

            % Initial guess
            wmean = sum(w .* (x.^2)); wmean = sum(wmean);
            init_omega = max(0.01, 0.25 * wmean);
            init_alpha1 = 0.1; init_alpha2 = 0.05; init_beta = 0.7;
            p0 = [log(init_omega);
                  log(init_alpha1/(0.999 - init_alpha1));
                  log(init_alpha2/(0.999 - init_alpha2));
                  log(init_beta/(0.999 - init_beta))];

            opts = optimoptions('fminunc','Algorithm','quasi-newton','Display','off');

            % Laplace
            fun_lap = @(p) weighted_laplace_nll(p, x, w);
            try
                pstar = fminunc(fun_lap, p0, opts);
            catch
                pstar = fminsearch(fun_lap, p0);
            end
            [om,a1,a2,b1] = transform_from_p(pstar);
            omega_hat_lap(i)=om; alpha1_hat_lap(i)=a1; alpha2_hat_lap(i)=a2; beta_hat_lap(i)=b1;

            % Gaussian
            fun_g = @(p) weighted_gauss_nll(p, x, w);
            try
                pstar = fminunc(fun_g, p0, opts);
            catch
                pstar = fminsearch(fun_g, p0);
            end
            [om,a1,a2,b1] = transform_from_p(pstar);
            omega_hat_gauss(i)=om; alpha1_hat_gauss(i)=a1; alpha2_hat_gauss(i)=a2; beta_hat_gauss(i)=b1;
        end

        % True parameters
        omega_true_at = omega_true(u_points);
        alpha1_true_at = alpha1_true(u_points);
        alpha2_true_at = alpha2_true(u_points);
        beta_true_at   = beta_true(u_points);

        fprintf('\n=== N = %d ===\n', N);
        fprintf('SRMISE(omega):  Laplace = %.4f, Gaussian = %.4f\n', ...
            srmise(omega_hat_lap, omega_true_at), srmise(omega_hat_gauss, omega_true_at));
        fprintf('SRMISE(alpha1): Laplace = %.4f, Gaussian = %.4f\n', ...
            srmise(alpha1_hat_lap, alpha1_true_at), srmise(alpha1_hat_gauss, alpha1_true_at));
        fprintf('SRMISE(alpha2): Laplace = %.4f, Gaussian = %.4f\n', ...
            srmise(alpha2_hat_lap, alpha2_true_at), srmise(alpha2_hat_gauss, alpha2_true_at));
        fprintf('SRMISE(beta):   Laplace = %.4f, Gaussian = %.4f\n', ...
            srmise(beta_hat_lap, beta_true_at), srmise(beta_hat_gauss, beta_true_at));

        % ===== PLOTS =====
        figure('Name',sprintf('tvGARCH(2,1) — N=%d',N));
        subplot(2,3,1);
        plot(u_points, omega_true_at,'k','LineWidth',2); hold on;
        plot(u_points, omega_hat_lap,'b--','LineWidth',1.3);
        plot(u_points, omega_hat_gauss,'r:','LineWidth',1.3);
        title('\omega(u)'); legend('True','Laplace','Gaussian'); grid on;

        subplot(2,3,2);
        plot(u_points, alpha1_true_at,'k','LineWidth',2); hold on;
        plot(u_points, alpha1_hat_lap,'b--','LineWidth',1.3);
        plot(u_points, alpha1_hat_gauss,'r:','LineWidth',1.3);
        title('\alpha_1(u)'); legend('True','Laplace','Gaussian'); grid on;

        subplot(2,3,3);
        plot(u_points, alpha2_true_at,'k','LineWidth',2); hold on;
        plot(u_points, alpha2_hat_lap,'b--','LineWidth',1.3);
        plot(u_points, alpha2_hat_gauss,'r:','LineWidth',1.3);
        title('\alpha_2(u)'); legend('True','Laplace','Gaussian'); grid on;

        subplot(2,3,4);
        plot(u_points, beta_true_at,'k','LineWidth',2); hold on;
        plot(u_points, beta_hat_lap,'b--','LineWidth',1.3);
        plot(u_points, beta_hat_gauss,'r:','LineWidth',1.3);
        title('\beta(u)'); legend('True','Laplace','Gaussian'); grid on;

        subplot(2,3,[5 6]);
        plot(u_points, alpha1_true_at + alpha2_true_at + beta_true_at,'k','LineWidth',2); hold on;
        plot(u_points, alpha1_hat_lap + alpha2_hat_lap + beta_hat_lap,'b--','LineWidth',1.3);
        plot(u_points, alpha1_hat_gauss + alpha2_hat_gauss + beta_hat_gauss,'r:','LineWidth',1.3);
        title('\alpha_1+\alpha_2+\beta'); legend('True','Laplace','Gaussian'); grid on;
    end
end

% ==== helper functions ====

function y = gaussian_kernel(x)
    y = exp(-x.^2 / 2) ./ sqrt(2*pi);
end

function [x, u] = simulate_tvgarch21(N)
    u = (1:N)'/N;
    omega = omega_true(u);
    a1 = alpha1_true(u);
    a2 = alpha2_true(u);
    b1 = beta_true(u);
    df = 3;
    z = trnd(df, N, 1);
    x = zeros(N,1);
    h2 = zeros(N,1);
    h2(1:2) = omega(1)/(1 - a1(1) - a2(1) - b1(1) + 1e-4);
    x(1:2) = sqrt(h2(1:2)).*z(1:2);
    for t=3:N
        h2(t) = max(omega(t) + a1(t)*x(t-1)^2 + a2(t)*x(t-2)^2 + b1(t)*h2(t-1), 1e-8);
        x(t) = sqrt(h2(t))*z(t);
    end
end

% === True parameter functions ===
function v = omega_true(u)
    v = 1 + 0.5*sin(5*u);
end
function v = alpha1_true(u)
    v = 0.1 + 0.4*(cos(4*u)).^2;
end
function v = alpha2_true(u)
    v = 0.05 + 0.2*(u.^2);
end
function v = beta_true(u)
    v = 0.1 + 0.04*u;
end

% === Likelihoods ===
function val = weighted_gauss_nll(p, x, w)
    [omega,a1,a2,b1] = transform_from_p(p);
    s2 = compute_s2(omega,a1,a2,b1,x);
    term = 0.5*log(s2) + 0.5*(x.^2)./s2;
    val = sum(w .* term);
end

function val = weighted_laplace_nll(p, x, w)
    [omega,a1,a2,b1] = transform_from_p(p);
    s2 = compute_s2(omega,a1,a2,b1,x);
    sigma = sqrt(s2);
    term = log(2.*sigma) + abs(x)./sigma;
    val = sum(w .* term);
end

% === Recursion for h^2 ===
function s2 = compute_s2(omega,a1,a2,b1,x)
    N = length(x);
    s2 = zeros(N,1);
    s2(1:2) = omega/(1 - a1 - a2 - b1 + 1e-4);
    for t=3:N
        s2(t) = max(omega + a1*x(t-1)^2 + a2*x(t-2)^2 + b1*s2(t-1), 1e-10);
    end
end

% === Transformations and metrics ===
function [omega,a1,a2,b1] = transform_from_p(p)
    omega = exp(p(1));
    a1 = 0.999 * (1 / (1 + exp(-p(2))));
    a2 = 0.999 * (1 / (1 + exp(-p(3))));
    b1 = 0.999 * (1 / (1 + exp(-p(4))));
end

function val = srmise(est, truev)
    val = sqrt(mean((est - truev).^2)) / mean(truev);
end
