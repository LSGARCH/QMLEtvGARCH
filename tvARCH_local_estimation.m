function tvARCH_local_estimation
    rng(12345);

    Ns = [1000, 2000, 5000];
    for k = 1:length(Ns)
        N = Ns(k);
        [x, u, omega_true_vec, alpha_true_vec] = simulate_tvarch1(N);
        u_points = linspace(0.02, 0.98, 50);
        h = 0.08;
        omega_hat_lap = zeros(size(u_points));
        omega_hat_gauss = zeros(size(u_points));
        alpha_hat_lap = zeros(size(u_points));
        alpha_hat_gauss = zeros(size(u_points));

        for i = 1:length(u_points)
            up = u_points(i);
            w = gaussian_kernel((u - up)/h);
            w = w(:);              
            w = w / sum(w);

            x = x(:);              
            % ----- initial guesses -----
            wmean = sum(w .* (x.^2)); 
            wmean = sum(wmean);       % scalar
            init_omega = max(0.01, 0.25 * wmean);
            init_alpha = 0.2;
            p0 = [log(init_omega); log(init_alpha/(0.999 - init_alpha) + 1e-12)];
            p0 = p0(:);

            % ----- Laplace -----
            fun_lap = @(p) weighted_laplace_nll(p, x, w);
            opts = optimoptions('fminunc','Algorithm','quasi-newton','Display','off');
            try
                pstar = fminunc(fun_lap, p0, opts);
            catch
                pstar = fminsearch(fun_lap, p0);
            end
            [om, al] = transform_from_p(pstar);
            omega_hat_lap(i) = om;
            alpha_hat_lap(i) = al;

            % ----- Gaussian -----
            fun_g = @(p) weighted_gauss_nll(p, x, w);
            try
                pstar = fminunc(fun_g, p0, opts);
            catch
                pstar = fminsearch(fun_g, p0);
            end
            [om, al] = transform_from_p(pstar);
            omega_hat_gauss(i) = om;
            alpha_hat_gauss(i) = al;
        end

        omega_true_at = omega_true(u_points);
        alpha_true_at = alpha_true(u_points);

    
        fprintf('\n=== N = %d ===\n', N);
        fprintf('SRMISE(omega): Laplace = %.4f, Gaussian = %.4f\n', ...
            srmise(omega_hat_lap, omega_true_at), srmise(omega_hat_gauss, omega_true_at));
        fprintf('SRMISE(alpha): Laplace = %.4f, Gaussian = %.4f\n', ...
            srmise(alpha_hat_lap, alpha_true_at), srmise(alpha_hat_gauss, alpha_true_at));

        
        figure;
        subplot(2,1,1);
        plot(u_points, omega_true_at, 'k', 'LineWidth', 2); hold on;
        plot(u_points, omega_hat_lap, 'b--', 'LineWidth', 1.5);
        plot(u_points, omega_hat_gauss, 'r:', 'LineWidth', 1.5);
        title(sprintf('N = %d — ω(u)', N));
        legend('True','Laplace','Gaussian');
        grid on;

        subplot(2,1,2);
        plot(u_points, alpha_true_at, 'k', 'LineWidth', 2); hold on;
        plot(u_points, alpha_hat_lap, 'b--', 'LineWidth', 1.5);
        plot(u_points, alpha_hat_gauss, 'r:', 'LineWidth', 1.5);
        title(sprintf('N = %d — α(u)', N));
        legend('True','Laplace','Gaussian');
        grid on;
    end
end



function y = gaussian_kernel(x)
    y = exp(-x.^2 / 2) ./ sqrt(2*pi);
end

function [x, u, omega, alpha] = simulate_tvarch1(N)
    u = (1:N) ./ N;
    omega = omega_true(u);
    alpha = alpha_true(u);
    df = 3; % heavy tails
    z = trnd(df, N, 1);  
    x = zeros(N,1);
    x(1) = z(1) * sqrt(max(omega(1)/(1-alpha(1)),1e-6));
    for t = 2:N
        s2 = omega(t) + alpha(t) * x(t-1)^2;
        x(t) = z(t) * sqrt(max(s2,1e-8));
    end
end

function v = omega_true(u)
    v = 1 + 0.5 * sin(5*u);
end

function v = alpha_true(u)
    v = 0.1 + 0.4 * cos(u);
end

function val = weighted_gauss_nll(p, x, w)
    [omega, alpha] = transform_from_p(p);
    s2 = compute_s2(omega, alpha, x);
    term = 0.5*log(s2) + 0.5 * (x.^2) ./ s2;
    val = sum(w .* term);
    val = sum(val);
end

function val = weighted_laplace_nll(p, x, w)
    [omega, alpha] = transform_from_p(p);
    s2 = compute_s2(omega, alpha, x);
    sigma = sqrt(s2);
    term = log(2.*sigma) + abs(x) ./ sigma;
    val = sum(w .* term);
    val = sum(val); 
end

function s2 = compute_s2(omega, alpha, x)
    N = length(x);
    s2 = zeros(N,1);
    s2(1) = omega;
    for t = 2:N
        s2(t) = max(omega + alpha * x(t-1)^2, 1e-10);
    end
end

function [omega, alpha] = transform_from_p(p)
    a = p(1); b = p(2);
    omega = exp(a);
    alpha = 0.999 * (1 / (1 + exp(-b)));
end

function val = srmise(est, truev)
    val = sqrt(mean((est - truev).^2)) / mean(truev);
end

