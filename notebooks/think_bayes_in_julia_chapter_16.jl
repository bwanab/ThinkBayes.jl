### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 75ff3f3e-e68d-11ec-3089-63c586f05505
begin
	import Pkg
	Pkg.develop(path="/Users/williamallen/src/ThinkBayes.jl")
	using ThinkBayes
end

# ╔═╡ 20a227a8-dece-403f-af26-92225880e2b8
using DataFrames, CSV, Plots, Distributions, PlutoUI

# ╔═╡ f1325978-b2d0-4ec8-ad45-7761510280f4
TableOfContents()

# ╔═╡ 860ea21a-13fb-44d7-b3dc-1fe39e4ce945
md"## Log Odds"

# ╔═╡ bfc2cf2d-b039-4f55-9bb4-cd5d49a86283
prob(o) = o / (o + 1)

# ╔═╡ ca4f274a-a896-4abe-87c6-afc50f9fd9c6


# ╔═╡ 20d6dee6-244d-420a-86ab-bf080d186db6
begin
	table = DataFrame(Index=["prior", "1 student", "2 students", "3 students"])
	table.odds = [10/3^x for x in 0:3]
	table.prob = 100 .* prob.(table.odds)
	table.prob_diff = vcat([missing], diff(table.prob))
	table
end

# ╔═╡ 7140af96-dfa0-4439-81a8-4bc9b128d0c3
begin
	table.log_odds = log.(table.odds)
	table.log_odds_diff = vcat([missing], diff(table.log_odds))
	table
end

# ╔═╡ fb12a0a8-5e2b-4e84-b9b3-28de8d6371ae
md"## The Space Shuttle Problem"

# ╔═╡ 6eae5309-b4b0-4a5c-850d-f5dd3c8aad1e
data = DataFrame(CSV.File(download("https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter2_MorePyMC/data/challenger_data.csv"), footerskip=1, missingstring=["NA"], dateformat="mm/dd/yyyy"))

# ╔═╡ c736605d-3b33-4ae7-9406-9830828b8584
begin
	dropmissing!(data)
	rename!(data, "Damage Incident" => :Damage)
end

# ╔═╡ e30170b5-302f-434d-a800-0c49039238e1
data

# ╔═╡ Cell order:
# ╠═20a227a8-dece-403f-af26-92225880e2b8
# ╠═f1325978-b2d0-4ec8-ad45-7761510280f4
# ╠═75ff3f3e-e68d-11ec-3089-63c586f05505
# ╟─860ea21a-13fb-44d7-b3dc-1fe39e4ce945
# ╠═bfc2cf2d-b039-4f55-9bb4-cd5d49a86283
# ╠═ca4f274a-a896-4abe-87c6-afc50f9fd9c6
# ╠═20d6dee6-244d-420a-86ab-bf080d186db6
# ╠═7140af96-dfa0-4439-81a8-4bc9b128d0c3
# ╟─fb12a0a8-5e2b-4e84-b9b3-28de8d6371ae
# ╠═6eae5309-b4b0-4a5c-850d-f5dd3c8aad1e
# ╠═c736605d-3b33-4ae7-9406-9830828b8584
# ╠═e30170b5-302f-434d-a800-0c49039238e1
