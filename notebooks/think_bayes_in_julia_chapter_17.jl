### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 80131c5a-eb24-11ec-2d5b-71a8a2588bd7
begin
	import Pkg
	Pkg.develop(path="/Users/williamallen/src/ThinkBayes.jl")
	using ThinkBayes
end

# ╔═╡ ec9b2808-3cb3-4fb6-9afe-4d70e3fd472c
using DataFrames, CSV, PlutoUI, Dates, Plots, Distributions, GLM

# ╔═╡ a0cbc648-5c95-42ad-a31a-099b86f9cb21
TableOfContents()

# ╔═╡ 7b01e654-6ec3-4a02-b1da-79df61d9fe21
df = DataFrame(CSV.File(download("https://github.com/AllenDowney/ThinkBayes2/raw/master/data/2239075.csv")))

# ╔═╡ ade8920c-dcc3-44b9-a729-a9807ed53eaa
dropmissing!(df, :SNOW)

# ╔═╡ 5aa38844-3e94-42e4-ab1b-7a76f925500a
df.YEAR = year.(df.DATE)

# ╔═╡ 882adccc-772e-4572-96be-99fd22f8e0c1
snow_all = combine(groupby(df, :YEAR), :SNOW => sum)

# ╔═╡ e9975617-59da-4dd4-8c63-6031c41e717e
snow = snow_all[2:end-1, :]

# ╔═╡ d2ca6151-5547-4bd8-96c1-0a18e14819b4


# ╔═╡ 8a009182-eded-4f83-9460-b07b5b558451
begin
	scatter(snow.YEAR, snow.SNOW_sum)
	plot!(title="Total annual snowfall at Blue Hills, MA")
	plot!(xaxis=("Total annual snowfall (inches)"), yaxis=("Year"), label="Snow")
end

# ╔═╡ 74ed34bc-f45f-453a-85fc-de14403f8e61
snow[findall(in([1978, 1996, 2015]), snow.YEAR), :]

# ╔═╡ b6051a69-2891-4dc1-b1bd-b5e55d3b10d5
md"## Regression Model"

# ╔═╡ d02e85ed-219d-4979-b6df-8a7db22396da
pmf_snowfall = pmf_from_seq(sort(snow.SNOW_sum))

# ╔═╡ 2021ea47-7f1c-425e-938b-71b5a03c2c09
begin
	mu = mean(pmf_snowfall)
	sigma = std(pmf_snowfall)
end

# ╔═╡ fa3260ba-ced6-4530-ab41-3ddfa90fb2ae
values(pmf_snowfall)

# ╔═╡ 0da150ee-bd94-4cfc-862c-a5fe442e918f
begin
	snow_normal_pmf = make_normal_pmf(values(pmf_snowfall), mu=mu, sigma=sigma)
	snow_normal_cdf = cdf_from_dist(values(pmf_snowfall), Normal(mu, sigma))
end;

# ╔═╡ 7476c576-b767-4103-9b58-9bcf5f471ca6
begin
	plot(snow_normal_cdf, label="model")
	plot!(make_cdf(pmf_snowfall), label="data")
	plot!(title="Normal model of variation in snowfall", xaxis=("Total snowfall (inches)"), yaxis=("CDF"))
end

# ╔═╡ 4762dd99-869f-4ec3-a4a4-9cb3d13c2e35
md"## Least Squares Regression"

# ╔═╡ df6fd335-9282-4ad7-a9ae-85754282340b
begin
	offset = round(mean(snow.YEAR))
	snow.x = snow.YEAR .- offset
end

# ╔═╡ 31c80ec7-e82c-4042-abe6-24a388a5d4b3
snow.y = snow.SNOW_sum

# ╔═╡ e52a121d-b33f-4a67-9030-8a1345bf1d46
results = lm(@formula(y ~ x), snow)

# ╔═╡ 479eedaf-0b3e-433e-a0b3-231f8c44abaa
coef(results)

# ╔═╡ 1944e63d-4e0d-40b6-aa0c-a367af26513e
std(residuals(results))

# ╔═╡ 0eacb0c1-c60f-4312-9def-114a02befb58
length(residuals(results))

# ╔═╡ 8d38eabc-c843-40d2-8728-2f95d4ebe2bd
md"## Priors"

# ╔═╡ 2734affb-99b6-4bf0-a0f6-1f7dad44b571
prior_slope = pmf_from_seq(range(-0.5, 1.5, 51));

# ╔═╡ 5ab90ab6-ae0e-4aed-ac72-0e338e4d2da3
prior_inter = pmf_from_seq(range(54, 75, 41));

# ╔═╡ 485fb9a7-d35c-430c-8871-3c6679419611
prior_sigma = pmf_from_seq(range(20, 35, 31));

# ╔═╡ 079ca099-25c3-41a8-a764-f67d790de128
function make_joint3(pmf1, pmf2, pmf3)
	vals1, probs1 = stack(make_joint(*, pmf2, pmf1))
	joint2 = pmf_from_seq(vals1, probs1)
	vals2, probs2 = stack(make_joint(*, pmf3, joint2))
	pmf_from_seq(vals2, probs2)
end

# ╔═╡ fcde48b9-3300-4351-a4ea-f84a9aa24f1c
prior = make_joint3(prior_sigma, prior_inter, prior_slope);

# ╔═╡ c4fb5bb7-fc65-47aa-a553-6bfc3b5bc49d
pp = make_joint(prior_sigma, prior_inter, prior_slope);

# ╔═╡ dca66cb9-2350-406a-9737-e455ec50b028
md"## Likelihood"

# ╔═╡ a9530988-632c-444e-87b5-cfb565b77932
begin
	inter = 64
	slope = 0.51
	sigma2 = 25
	xss = snow.x
	yss = snow.y
end;

# ╔═╡ f359ce7b-82bc-4b24-91d2-5bdb66a0802c
md"It would be better to use residuals(results) from above, but this is what ThinkBayes2 does"

# ╔═╡ afd8d712-a614-4b82-8f13-edda445eea8f
begin
	expected = slope .* xss .+ inter
	resid = yss .- expected
end

# ╔═╡ fcf36d6a-6845-4446-b4fd-a5c6ff0b3f84
begin
	N = Normal(0, sigma2)
	densities = [pdf(N, x) for x in resid]
end;

# ╔═╡ 7bfbafd5-17a7-46bd-8dcd-3796e1fa5661
likelihood = prod(densities)

# ╔═╡ da14930f-1a4b-419d-b33f-d4ebf7d0cbc6
md"## The Update"

# ╔═╡ 7e7d2106-a2a1-4b1a-a0ec-3d0b91e2b515
md"""
The following code has two versions of _correctedness_. The old correct version is using my original implementation of 3 variable pmfs which was a linear pmf where the values are (x, (y, z)) shaped. The new correct version is using 3D array based joint distributions where the axes are x, y, z.

As implied, both produce correct numbers, but I'm still a bit confused about the arrangement of the x, y, z axes. I have a couple of hypotheses:

1. My original implementation was confused and thus to match it's correctness, I've had to implement the same confusion.
2. My original implementation is ok, and all I need to do is arrange the new implementation such that consistent ordering of the axes give consistent results in a predictible manner.

My current best guess is that the answer is 1.
"""

# ╔═╡ 7186ccaf-30e5-40f9-bb28-88aa522c10c3
begin
	function compute_likelihood(slope, inter, sigma, xs, ys)
		expected = slope .* xs .+ inter
		resid = ys .- expected
		N = Normal(0, sigma)
		densities = [pdf(N, x) for x in resid]
		prod(densities)
	end
	
	likelihoods = [compute_likelihood(slope, inter, sigma, xss, yss) for (slope, (inter, sigma)) in values(prior)]
end

# ╔═╡ b5311257-57d2-48d0-8efe-f266cc892f76
likes = [compute_likelihood(slope, inter, sigma, xss, yss) for slope in zs(pp) for inter in xs(pp) for sigma in ys(pp)]

# ╔═╡ 2c0d367a-5ba1-47ff-a46b-3f347c7b4b32
likes2 = reshape(likes, size(pp))

# ╔═╡ f40b39db-8298-4a2d-bcaf-5d670bcc79a4
maximum(likelihoods), maximum(likes), minimum(likelihoods), minimum(likes)

# ╔═╡ 3cd1188d-92e0-4c3f-8aca-febb97bda91c
maximum(likelihoods .- likes), minimum(likelihoods .- likes), std(likelihoods .- likes)

# ╔═╡ 8f4d6a44-6843-4275-9d26-82829c489b2b
begin
	posterior_pmf = prior * likelihoods;
	posterior_joint = pp * likes2;
end;

# ╔═╡ fca2969d-9558-4f61-8412-6742f0e82235
posterior_slopes, posterior_inters, posterior_sigmas = ThinkBayes.marginals3(posterior_pmf);

# ╔═╡ 8f69b6d5-ec80-4077-81ab-5e2ea745153b
plot(posterior_sigmas)

# ╔═╡ e9bb673e-cbd5-4e17-8cdc-d05f91529646
plot(marginal(posterior_joint, 2))

# ╔═╡ f9f3bf0c-8b7c-42e5-8752-92515b8fee0f
plot(posterior_inters, xaxis=("intercept (inches)"), yaxis=("PDF"), title="Posterior marginal distribution of intercept")

# ╔═╡ a7a3bc72-8e7d-491d-81a2-dba3efcc5e90
plot(marginal(posterior_joint, 3))

# ╔═╡ 92dbb2f7-0cc2-4b28-9dde-783ee6547d51
mean(posterior_inters), credible_interval(posterior_inters, 0.9)

# ╔═╡ 5f18e97d-1150-4d9b-b881-be3f9e94e6a7
plot(posterior_slopes, xaxis=("slope (inches)"), yaxis=("PDF"), title="Posterior marginal distribution of slope")

# ╔═╡ e3a858e2-8e04-4db6-a956-cc965eb417f6
plot(marginal(posterior_joint, 1))

# ╔═╡ 4011772f-4c2b-4783-9dcd-e11eb5d7089d
mean(posterior_slopes), credible_interval(posterior_slopes, 0.9)

# ╔═╡ 20e5609b-4415-4d46-8d17-2c9ab5a5ccd8
cdf(make_cdf(posterior_slopes), 0)

# ╔═╡ cbf40631-39f3-4bf8-aea7-46da201a8bfc
md"## Optimization"

# ╔═╡ Cell order:
# ╠═80131c5a-eb24-11ec-2d5b-71a8a2588bd7
# ╠═ec9b2808-3cb3-4fb6-9afe-4d70e3fd472c
# ╠═a0cbc648-5c95-42ad-a31a-099b86f9cb21
# ╠═7b01e654-6ec3-4a02-b1da-79df61d9fe21
# ╠═ade8920c-dcc3-44b9-a729-a9807ed53eaa
# ╠═5aa38844-3e94-42e4-ab1b-7a76f925500a
# ╠═882adccc-772e-4572-96be-99fd22f8e0c1
# ╠═e9975617-59da-4dd4-8c63-6031c41e717e
# ╠═d2ca6151-5547-4bd8-96c1-0a18e14819b4
# ╠═8a009182-eded-4f83-9460-b07b5b558451
# ╠═74ed34bc-f45f-453a-85fc-de14403f8e61
# ╟─b6051a69-2891-4dc1-b1bd-b5e55d3b10d5
# ╠═d02e85ed-219d-4979-b6df-8a7db22396da
# ╠═2021ea47-7f1c-425e-938b-71b5a03c2c09
# ╠═fa3260ba-ced6-4530-ab41-3ddfa90fb2ae
# ╠═0da150ee-bd94-4cfc-862c-a5fe442e918f
# ╠═7476c576-b767-4103-9b58-9bcf5f471ca6
# ╟─4762dd99-869f-4ec3-a4a4-9cb3d13c2e35
# ╠═df6fd335-9282-4ad7-a9ae-85754282340b
# ╠═31c80ec7-e82c-4042-abe6-24a388a5d4b3
# ╠═e52a121d-b33f-4a67-9030-8a1345bf1d46
# ╠═479eedaf-0b3e-433e-a0b3-231f8c44abaa
# ╠═1944e63d-4e0d-40b6-aa0c-a367af26513e
# ╠═0eacb0c1-c60f-4312-9def-114a02befb58
# ╟─8d38eabc-c843-40d2-8728-2f95d4ebe2bd
# ╠═2734affb-99b6-4bf0-a0f6-1f7dad44b571
# ╠═5ab90ab6-ae0e-4aed-ac72-0e338e4d2da3
# ╠═485fb9a7-d35c-430c-8871-3c6679419611
# ╠═079ca099-25c3-41a8-a764-f67d790de128
# ╠═fcde48b9-3300-4351-a4ea-f84a9aa24f1c
# ╠═c4fb5bb7-fc65-47aa-a553-6bfc3b5bc49d
# ╟─dca66cb9-2350-406a-9737-e455ec50b028
# ╠═a9530988-632c-444e-87b5-cfb565b77932
# ╟─f359ce7b-82bc-4b24-91d2-5bdb66a0802c
# ╠═afd8d712-a614-4b82-8f13-edda445eea8f
# ╠═fcf36d6a-6845-4446-b4fd-a5c6ff0b3f84
# ╠═7bfbafd5-17a7-46bd-8dcd-3796e1fa5661
# ╟─da14930f-1a4b-419d-b33f-d4ebf7d0cbc6
# ╟─7e7d2106-a2a1-4b1a-a0ec-3d0b91e2b515
# ╠═7186ccaf-30e5-40f9-bb28-88aa522c10c3
# ╠═b5311257-57d2-48d0-8efe-f266cc892f76
# ╠═2c0d367a-5ba1-47ff-a46b-3f347c7b4b32
# ╠═f40b39db-8298-4a2d-bcaf-5d670bcc79a4
# ╠═3cd1188d-92e0-4c3f-8aca-febb97bda91c
# ╠═8f4d6a44-6843-4275-9d26-82829c489b2b
# ╠═fca2969d-9558-4f61-8412-6742f0e82235
# ╠═8f69b6d5-ec80-4077-81ab-5e2ea745153b
# ╠═e9bb673e-cbd5-4e17-8cdc-d05f91529646
# ╠═f9f3bf0c-8b7c-42e5-8752-92515b8fee0f
# ╠═a7a3bc72-8e7d-491d-81a2-dba3efcc5e90
# ╠═92dbb2f7-0cc2-4b28-9dde-783ee6547d51
# ╠═5f18e97d-1150-4d9b-b881-be3f9e94e6a7
# ╠═e3a858e2-8e04-4db6-a956-cc965eb417f6
# ╠═4011772f-4c2b-4783-9dcd-e11eb5d7089d
# ╠═20e5609b-4415-4d46-8d17-2c9ab5a5ccd8
# ╟─cbf40631-39f3-4bf8-aea7-46da201a8bfc
