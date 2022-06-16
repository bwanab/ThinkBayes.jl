module Joint3Tst

using ThinkBayes
using Test

@testset "Test Joint3" begin
#d1, d2, d3 = 20, 30, 50
d1, d2, d3 = 3, 4, 5
pmf1 = pmf_from_seq(1:d1, normalize(rand(d1)))
pmf2 = pmf_from_seq(1:d2, normalize(rand(d2)))
pmf3 = pmf_from_seq(1:d3, normalize(rand(d3)))
joint2 = make_joint(*, pmf2, pmf3)
vs1, probs1 = stack(joint2)
joint2_pmf = pmf_from_seq(vs1, probs1)
joint3 = make_joint(*, pmf1, joint2_pmf)
vs2, probs2 = stack(joint3)
joint3_pmf = pmf_from_seq(vs2, probs2)
m1, m2, m3 = marginals3(joint3_pmf)
@test m1 ≈ pmf1
@test m2 ≈ pmf2
@test m3 ≈ pmf3
"""
Note that the following is a set of experiments to think about how to 
make a Joint of 3 (or more) parameters/dimensions work.
"""
M3 = reshape(probs(joint3_pmf), d3,d2,d1)
#@test reshape(sum(sum(M3, dims=1), dims=2), d3) ≈ probs(m1)
#@test reshape(sum(sum(M3, dims=1), dims=3), d2) ≈ probs(m2)
#@test reshape(sum(sum(M3, dims=2), dims=3), d1) ≈ probs(m3)
@test [sum(M3[:,:,x]) for x in 1:d1] ≈ probs(m1)
@test [sum(M3[:,x,:]) for x in 1:d2] ≈ probs(m2)
@test [sum(M3[x,:,:]) for x in 1:d3] ≈ probs(m3)

"""
Using BenchmarkTools, I've determined that for arrays of significant
size the view method works best.
"""

@test [sum(view(M3, :,:,x)) for x in 1:d1] ≈ probs(m1)
@test [sum(view(M3, :,x,:)) for x in 1:d2] ≈ probs(m2)
@test [sum(view(M3, x,:,:)) for x in 1:d3] ≈ probs(m3)

"""
This is how one might compute the 3rd parameter 3d matrix of a given Joint and a Pmf.
"""

M4 = permutedims(reshape(outer(*, reshape(joint2.M, prod(size(joint2.M))), probs(pmf1)), d1,d3,d2), (1,3,2))
#@test joint3.M == M4
@test [sum(view(M4, :,:,x)) for x in 1:d3] ≈ probs(m3)
@test [sum(view(M4, x,:,:)) for x in 1:d1] ≈ probs(m1)
@test [sum(view(M4, :,x,:)) for x in 1:d2] ≈ probs(m2)

joint3_new = make_joint(joint2, pmf1)
end
end
