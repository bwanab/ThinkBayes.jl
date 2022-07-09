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
