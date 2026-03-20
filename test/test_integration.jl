# test/test_integration.jl
using Test
using LoopToVecs

@testset "Integration: setup.jl patterns" begin
    num_regions, num_sectors, num_s = 3, 2, 2
    num_x = num_regions * num_sectors * num_s

    # tau_bar pattern: @t A[n,j,s,np,jp,sp] += 5.0*(n!=np)
    tau_bar = zeros(num_regions, num_sectors, num_s, num_regions, num_sectors, num_s)
    @t tau_bar[n, j, s, np, jp, sp] += 5.0 * (n != np)
    for n in 1:num_regions, np in 1:num_regions, j in 1:num_sectors, s in 1:num_s,
        jp in 1:num_sectors, sp in 1:num_s
        @test tau_bar[n, j, s, np, jp, sp] ≈ (n != np ? 5.0 : 0.0)
    end

    # ln_d pattern: @t A[n,i,j] += 1.0*(n!=i)
    ln_d = zeros(num_regions, num_regions, num_sectors)
    @t ln_d[n, i, j] += 1.0 * (n != i)
    for n in 1:num_regions, i in 1:num_regions, j in 1:num_sectors
        @test ln_d[n, i, j] ≈ (n != i ? 1.0 : 0.0)
    end
end

@testset "Integration: solve_decision.jl patterns" begin
    num_a, num_s = 5, 2
    aR = 3
    ss_bar = 0.3
    Tss_bar = 0.1

    # @t ss_as[a,s] = ss_bar*(a<aR)*(s==2)
    ss_as = zeros(num_a, num_s)
    @t ss_as[a, s] = ss_bar * (a < aR) * (s == 2)
    for a in 1:num_a, s in 1:num_s
        @test ss_as[a, s] ≈ ss_bar * (a < aR) * (s == 2)
    end

    # @t Tss_as[a,s] = Tss_bar*(a>=aR)*(s==2)
    Tss_as = zeros(num_a, num_s)
    @t Tss_as[a, s] = Tss_bar * (a >= aR) * (s == 2)
    for a in 1:num_a, s in 1:num_s
        @test Tss_as[a, s] ≈ Tss_bar * (a >= aR) * (s == 2)
    end

    # $var indexing inside a "loop"
    num_x = 6
    psi1_ax = zeros(num_a, num_x)
    y_ax = rand(num_a, num_x)
    eta = 1.0

    # @t psi1_ax[$num_a, x] = 1/eta*exp(-eta*y_ax[$num_a, x])
    @t psi1_ax[$num_a, x] = 1 / eta * exp(-eta * y_ax[$num_a, x])
    @test psi1_ax[num_a, :] ≈ (1 / eta) .* exp.(-eta .* y_ax[num_a, :])
    @test psi1_ax[1, :] == zeros(num_x)  # other rows untouched
end

@testset "Integration: solve_aggregates.jl patterns" begin
    num_a, num_x = 4, 3

    # @t N_ax[1,x] = N1_x[x]
    N_ax = zeros(num_a, num_x)
    N1_x = rand(num_x)
    @t N_ax[1, x] = N1_x[x]
    @test N_ax[1, :] == N1_x
    @test N_ax[2, :] == zeros(num_x)

    # @t Tau_ax[$num_a, a] = 0.0
    Tau_ax = rand(num_a, num_a)
    old_row1 = copy(Tau_ax[1, :])
    @t Tau_ax[$num_a, a] = 0.0
    @test Tau_ax[num_a, :] == zeros(num_a)
    @test Tau_ax[1, :] == old_row1
end

@testset "Integration: solve_gN.jl patterns" begin
    # @t eq1[a] := N[a+1]*(1+gN) - zeta_a[a]*N[a]
    num_a = 5
    N = rand(num_a)
    zeta_a = rand(num_a - 1)
    gN = 0.02

    @t eq1[a] := N[a+1] * (1 + gN) - zeta_a[a] * N[a]
    @test length(eq1) == num_a - 1
    for a in 1:(num_a-1)
        @test eq1[a] ≈ N[a+1] * (1 + gN) - zeta_a[a] * N[a]
    end

    # scalar reduction: @t (+) eq3 += N[a]
    eq3 = -1.0
    @t (+) eq3 += N[a]
    @test eq3 ≈ -1.0 + sum(N)
end

@testset "Integration: calibration.jl patterns" begin
    num_regions = 3

    # @t tau_bar_ni[n,i] += 5.0*(n!=i)
    tau_bar_ni = zeros(num_regions, num_regions)
    @t tau_bar_ni[n, i] += 5.0 * (n != i)
    for n in 1:num_regions, i in 1:num_regions
        @test tau_bar_ni[n, i] == (n != i ? 5.0 : 0.0)
    end

    # @t resid_mu[n,n] = tau_bar_ni[n,n]
    resid_mu = zeros(num_regions, num_regions)
    @t resid_mu[n, n] = tau_bar_ni[n, n]
    @test diag(resid_mu) ≈ diag(tau_bar_ni)
    @test resid_mu[1, 2] == 0.0
end

@testset "Integration: solve_eq.jl patterns" begin
    num_regions, num_sectors = 3, 2
    num_a, num_s = 5, 2

    # @t (+) Tss_numer := ss_bar*N_a_njs[a,n,j,s]*W_nj[n,j]*e_aj[a,j]*(a<aR)*(s==2)
    N_a_njs = rand(num_a, num_regions, num_sectors, num_s)
    W_nj = rand(num_regions, num_sectors)
    e_aj = rand(num_a, num_sectors)
    ss_bar = 0.1
    aR = 3
    @t (+) Tss_numer := ss_bar * N_a_njs[a, n, j, s] * W_nj[n, j] * e_aj[a, j] * (a < aR) * (s == 2)
    expected = sum(
        ss_bar * N_a_njs[a, n, j, s] * W_nj[n, j] * e_aj[a, j] * (a < aR) * (s == 2)
        for a in 1:num_a, n in 1:num_regions, j in 1:num_sectors, s in 1:num_s
    )
    @test Tss_numer ≈ expected

    # *= pattern
    implied_W = rand(num_regions, num_sectors)
    GDP = 10.0
    orig = copy(implied_W)
    @t implied_W[n, j] *= 1 / GDP
    @test implied_W ≈ orig ./ GDP
end

@testset "Integration: calc_stats.jl patterns (high-dim reduction)" begin
    num_a, num_regions, num_sectors, num_s = 3, 2, 2, 2

    N_a_njs = rand(num_a, num_regions, num_sectors, num_s)

    # @t (+) denom_mu_n[n] := N_a_njs[a,n,j,s]  — reduce over a,j,s
    @t (+) denom_mu_n[n] := N_a_njs[a, n, j, s]
    for n in 1:num_regions
        @test denom_mu_n[n] ≈ sum(N_a_njs[:, n, :, :])
    end
end

@testset "Integration: 3+ dim permutation with different sizes" begin
    # Regression test: permutation must be correct when 3+ dimensions
    # have different sizes (not just transpositions).
    # Uses different sizes to catch perm vs invperm confusion.
    num_a, num_n, num_j = 2, 3, 4

    A = rand(num_a, num_n, num_j)

    # Reduction over a: B[n,j] = sum_a A[a,n,j]
    @t (+) B[n, j] := A[a, n, j]
    for n in 1:num_n, j in 1:num_j
        @test B[n, j] ≈ sum(A[:, n, j])
    end

    # Pure permutation (no reduction): C[n,j,a] = A[a,n,j]
    @t C[n, j, a] := A[a, n, j]
    for a in 1:num_a, n in 1:num_n, j in 1:num_j
        @test C[n, j, a] ≈ A[a, n, j]
    end

    # 4D pattern matching solve_eq.jl: e_aj[a,j]*N_a_njs[a,n,j,s]
    num_s = 2
    N_a_njs = rand(num_a, num_n, num_j, num_s)
    e_aj = rand(num_a, num_j)
    @t (+) W_denom[n, j] := e_aj[a, j] * N_a_njs[a, n, j, s]
    for n in 1:num_n, j in 1:num_j
        expected = sum(e_aj[a, j] * N_a_njs[a, n, j, s]
                       for a in 1:num_a, s in 1:num_s)
        @test W_denom[n, j] ≈ expected
    end
end

@testset "Integration: \$a inside for loop (solve_aggregates adaptation)" begin
    # Patterns from code_samples that use Tullio's implicit reduction
    # must be adapted with explicit (+) for LoopToVecs.
    # For-loop variables require $a prefix.

    num_a, num_x = 4, 3
    N_ax = zeros(num_a, num_x)
    N_ax[1, :] .= 1.0
    zeta = [0.9, 0.8, 0.7]
    mu = ones(num_a, num_x, num_x) ./ num_x  # uniform transitions

    for a = 1:num_a-1
        @t (+) N_ax[$a+1, xp] = zeta[$a] * N_ax[$a, x] * mu[$a, x, xp]
    end
    # verify reduction over x worked: sum_x(zeta[1] * 1.0 * (1/num_x)) = zeta[1]
    @test N_ax[2, :] ≈ fill(zeta[1], num_x)
end
