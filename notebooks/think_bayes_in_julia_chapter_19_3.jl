### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 6d67c447-e64b-4506-bc55-30ba9f29bef5
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/ThinkBayes.jl")
	using ThinkBayes
end


# ╔═╡ 1bd08068-0aa0-11ed-28f7-0bc0172f07d2
using Distributions, Plots

# ╔═╡ 1213cab7-6156-4609-84e9-5de35447f62a



# ╔═╡ 570e848e-40c5-484a-a67d-2a3293f127d0
lam = Gamma(1.4, 1)

# ╔═╡ ad5ddc2b-3ff9-4691-afde-da5b5f72c9fd
function model(alpha, beta, n)
	lam = Gamma(alpha, beta)
	[rand(Poisson(x)) for x in rand(lam, n)]
end

# ╔═╡ 50451c2a-ec77-45b1-9a40-fb516fbd7bb1
gs = model(1.4, 1, 1000)

# ╔═╡ fbd6e222-c0a3-49a4-aa32-a5d25e4e9c61
plot(cdf_from_seq(gs), seriestype=:step, xaxis=[0,2,4,6,8,10])

# ╔═╡ Cell order:
# ╠═6d67c447-e64b-4506-bc55-30ba9f29bef5
# ╠═1213cab7-6156-4609-84e9-5de35447f62a
# ╠═1bd08068-0aa0-11ed-28f7-0bc0172f07d2
# ╠═570e848e-40c5-484a-a67d-2a3293f127d0
# ╠═ad5ddc2b-3ff9-4691-afde-da5b5f72c9fd
# ╠═50451c2a-ec77-45b1-9a40-fb516fbd7bb1
# ╠═fbd6e222-c0a3-49a4-aa32-a5d25e4e9c61
