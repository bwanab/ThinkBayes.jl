### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 80131c5a-eb24-11ec-2d5b-71a8a2588bd7
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/ThinkBayes.jl")
	using ThinkBayes
end

# ╔═╡ ec9b2808-3cb3-4fb6-9afe-4d70e3fd472c
using DataFrames, CSV, PlutoUI, Dates, Plots, Distributions, GLM, Random, Interpolations

# ╔═╡ a0cbc648-5c95-42ad-a31a-099b86f9cb21
TableOfContents()

# ╔═╡ d213924b-4243-4b0a-a80f-de704d6e806f
md"# Snowfall in MA"

# ╔═╡ 7b01e654-6ec3-4a02-b1da-79df61d9fe21
df = DataFrame(CSV.File(download("https://github.com/AllenDowney/ThinkBayes2/raw/master/data/2239075.csv")))

# ╔═╡ ade8920c-dcc3-44b9-a729-a9807ed53eaa
begin
	dropmissing!(df, :SNOW)
	dropmissing!(df, :TMIN)
end

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
	plot!(xaxis=("Total annual snowfall (inches)"), yaxis=("Year"), label="Snow", legend_position=:topleft)
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
	mu, sigma
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
The following code has two versions of _correctness_. The old correct version is using my original implementation of 3 variable pmfs which was a linear pmf where the values are (x, (y, z)) shaped. The new correct version is using 3D array based joint distributions where the axes are x, y, z.

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

# ╔═╡ ba245d94-9222-432e-aee7-aa360943dfce
values(prior)

# ╔═╡ e0960e05-16f4-4fa1-a8b0-4407835f2f1d
ys(pp)

# ╔═╡ b5311257-57d2-48d0-8efe-f266cc892f76
likes = [compute_likelihood(slope, inter, sigma, xss, yss)  for slope in zs(pp) for sigma in xs(pp) for inter in ys(pp)  ]

# ╔═╡ 2c0d367a-5ba1-47ff-a46b-3f347c7b4b32
likes2 = reshape(likes, size(pp));

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
a1 = plot(posterior_sigmas, xaxis=("sigma"), yaxis=("PDF"));

# ╔═╡ e9bb673e-cbd5-4e17-8cdc-d05f91529646
b1 = plot(marginal(posterior_joint, 1));

# ╔═╡ f9f3bf0c-8b7c-42e5-8752-92515b8fee0f
a2 = plot(posterior_inters, xaxis=("intercept (inches)"));

# ╔═╡ a7a3bc72-8e7d-491d-81a2-dba3efcc5e90
b2 = plot(marginal(posterior_joint, 2));

# ╔═╡ 92dbb2f7-0cc2-4b28-9dde-783ee6547d51
mean(posterior_inters), credible_interval(posterior_inters, 0.9)

# ╔═╡ 5f18e97d-1150-4d9b-b881-be3f9e94e6a7
a3 = plot(posterior_slopes, xaxis=("slope (inches)"));

# ╔═╡ e3a858e2-8e04-4db6-a956-cc965eb417f6
b3 = plot(marginal(posterior_joint, 3));

# ╔═╡ 37dbfd29-0c03-489a-8ed4-6cbdde3b09c6
plot([a1, a2, a3, b1, b2, b3]..., layout=(2,3), size=[800, 400])

# ╔═╡ 4011772f-4c2b-4783-9dcd-e11eb5d7089d
mean(posterior_slopes), credible_interval(posterior_slopes, 0.9)

# ╔═╡ 20e5609b-4415-4d46-8d17-2c9ab5a5ccd8
cdf(make_cdf(posterior_slopes), 0)

# ╔═╡ 58741dcc-1b8c-4830-90d2-b91ec3b7551a
pj = make_joint(marginal(posterior_joint, 3), marginal(posterior_joint, 2));

# ╔═╡ f6fc984c-5692-4ba7-843c-193af68bd6de
contour(pj)

# ╔═╡ cbf40631-39f3-4bf8-aea7-46da201a8bfc
md"## Optimization

skipped for now"

# ╔═╡ b522db3c-6cd2-40a2-8d28-96702874c3fb
begin
	zz = zs(pp)
	yy = ys(pp)
	zlen, ylen = length(zz), length(yy)
	joint3 = [(zz[zi], yy[yi], pp.M[yi,:,zi]) for zi in 1:zlen for yi in 1:ylen]
end

# ╔═╡ b20fe8f4-8fcc-4633-83a9-be82f5dee6be
length(pp.M[1,:,1])

# ╔═╡ bd1febb8-b7d5-4e0a-925f-0a9b8094a286
md"# Marathon World Record"

# ╔═╡ 776ebc1a-9e39-44b6-9691-b4f21a763fdf
table = DataFrame(CSV.File(homedir()*"/src/ThinkBayes.jl/marathon_times.csv", dateformat="hh:mm:ss"))

# ╔═╡ c2c7650c-5c22-40bd-aaf3-579896f83788
parse_time(s) = sum(parse.(Float64, split(s, ":")) .* [60^2, 60, 1])

# ╔═╡ 8e017761-c277-462e-afc0-b668b538da0b
parse_time(table[16, :Time])

# ╔═╡ 04200a27-78c7-4a47-b0c9-78fe65c11fad
table.Seconds = [parse_time(x) for x in table.Time]

# ╔═╡ b065d6b4-9516-439b-8461-12f4bb2853e7
function parsedate(dt)
	md,ys = split(dt, ", ")
	ms,ds = split(md, " ")
	m = findfirst(x -> x == ms, ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
	d = parse(Int, ds)
	y = parse(Int, ys[1:4])
	Date(y, m, d)
end

# ╔═╡ 161c906b-0bd8-4003-84d1-8c2b0f5c316a
table.Date = [parsedate(dt) for dt in table.Date];

# ╔═╡ 5ded5619-dbd9-438f-899e-f598ca23cec7
table.y = (26.2 ./ table.Seconds) .* 3600

# ╔═╡ fd86f76a-996b-4a29-96a5-2784248af278
scatter(table.Date, table.y)

# ╔═╡ f800cc5c-c871-4d1b-aff6-6a345d93c0b6
timedelta = table.Date - Date(1995)

# ╔═╡ f052d6d3-40dd-44fc-ae15-3ba68b498cfe
table.x = Dates.value.(timedelta) ./ 365.25

# ╔═╡ 7c8e8b91-b7de-4b21-9a12-6027e0452768
recent = table[table.Date .> Date(1970),:]

# ╔═╡ e34c87fa-b51b-479c-a467-f63de478d1d8
recent

# ╔═╡ fb2077d4-27c0-4fc9-a234-e6cc48fd10d6
scatter(recent.Date, recent.y)

# ╔═╡ cdc1fc69-db45-4b22-bb7a-80e625aa6840
maximum(recent.x)

# ╔═╡ b0d16dc7-4ab0-4593-83bb-a95e13bf048b
m_results = lm(@formula(y ~ x), recent)

# ╔═╡ bf7d6f38-1eaa-4b82-b526-8bc5b59ec6b5
std(residuals(m_results))

# ╔═╡ 5e42f35e-1f5f-4703-9aa3-8e49048dc73e
md"## Priors"

# ╔═╡ 56cac61a-cbb2-46f3-b1b3-b74c6f373986
m_prior_slope = pmf_from_seq(LinRange(0.012, 0.018, 51));

# ╔═╡ 42bc7287-533e-4773-8e86-2276cabc77ba
m_prior_inter = pmf_from_seq(LinRange(12.4, 12.5, 41));

# ╔═╡ 6b52f227-14f5-4994-9daf-4d428777589d
m_prior_sigma = pmf_from_seq(LinRange(0.01, 0.21, 31));

# ╔═╡ 300b234b-9307-4701-b739-c7c1d9500535
m_prior = make_joint(m_prior_sigma, m_prior_inter, m_prior_slope);

# ╔═╡ 88cadc6e-6d80-4eef-9ea9-d41287105017
begin
	m_xs = recent.x
	m_ys = recent.y
	m_likes = [compute_likelihood(slope, inter, sigma, m_xs, m_ys)  for slope in zs(m_prior) for sigma in xs(m_prior) for inter in ys(m_prior)  ]
end

# ╔═╡ b5d1dd33-b19d-4d8c-863d-c4a993ea51d0
m_likes2 = reshape(m_likes, size(m_prior));

# ╔═╡ 57738a5c-a5fd-45da-a2aa-113ad6d6b6bf
m_posterior_joint = m_prior * m_likes2;

# ╔═╡ aabebba3-09e1-4790-9607-33d9b33d4122
plot(marginal(m_posterior_joint, 1), xaxis="Sigma")

# ╔═╡ aa6c1c01-b833-4cb8-a565-d17c53baae2c
plot(marginal(m_posterior_joint, 2), xaxis="Intercept")

# ╔═╡ 7058a81f-2a16-42a8-a9b7-ed4f3c632206
plot(marginal(m_posterior_joint, 3), xaxis="Slope")

# ╔═╡ 2bf2d0af-cdc4-4ee8-8960-2eaf67441209
md"## Prediction"

# ╔═╡ 077f2509-8255-446b-b5ae-5f2eb92b6c35
begin
	Random.seed!(17)
	sample = rand(m_posterior_joint, 101)
end

# ╔═╡ eefcf367-1358-4d42-8171-18d6bc082823
begin
	m_xvals = -25:2:50
	m_pred = [inter .+ slope .* m_xvals .+ rand(Normal(0, sigma), length(m_xvals)) for (sigma, inter, slope) in sample]
	pred = reduce(hcat, m_pred)
end

# ╔═╡ 25fbd848-4057-452f-a289-a0492298e768
low, median, high = percentile(pred, [5, 50, 95], dims=1)

# ╔═╡ ef1f93e6-f1fa-477c-84c9-dd5bd72fbfd1
mean(high .- median), mean(median .- low)

# ╔═╡ 7556a851-60f5-44b6-8802-bc296eaa0f72
begin
	plot(Date.(m_xvals .+ 1995), median, ribbon=(median .- low, high .- median), fillalpha=0.3)
	scatter!(recent.Date, recent.y)
	hline!([13.1], legend_position=:topleft)
end

# ╔═╡ fa59019d-856c-4cbd-9bf3-1b70320052db
future = [LinearInterpolation(Interpolations.deduplicate_knots!(x, move_knots=true), m_xvals)(13.1) for x in [high, median, low]]

# ╔═╡ 4406a122-898a-40da-8c4a-a41c852094a2
[Date(1995) + Period(Day(round(x * 365.24))) for x in future]

# ╔═╡ fa021aeb-28c6-4b66-9608-691000b37a02
md"# Exercises"

# ╔═╡ d525d152-56dd-4e80-a1fd-8ce431d5be5f
df.YEAR = year.(df.DATE)

# ╔═╡ 31406bc5-102d-4eb4-8f11-8dd064ba29f5
df.TMID = (df.TMIN + df.TMAX) / 2

# ╔═╡ baaaa99f-0334-4d44-9cb2-2149c32bbab8
dropmissing!(df, :TMID)

# ╔═╡ 588304b9-1037-4fab-8f13-9321ebea29d5
tmid = [year=(first(g).YEAR, mean_temp=mean(g.TMID)) for g in groupby(df, :YEAR)];

# ╔═╡ f5043e74-c3c6-4888-80fd-cf4f20d6ff9e
complete = DataFrame(tmid[2:end-1]);

# ╔═╡ 0603607d-8666-41fb-967f-5419020c0200
scatter(complete.YEAR, complete.mean_temp, xaxis="year", yaxis="Annual average temp (F)")

# ╔═╡ 76f15456-54c4-43a9-8ed7-f42be43ee3b5
t_offset = round(mean(complete.YEAR))

# ╔═╡ 51c75fff-7634-44fc-aaac-a0fdee149da6
complete.x = complete.YEAR .- t_offset

# ╔═╡ ed2127a8-ecc2-4400-9219-c94c812f8a4c
complete.y = complete.mean_temp

# ╔═╡ e93241bb-d2b8-45ad-9470-011bb97c6803
t_result = lm(@formula(y ~ x), complete)

# ╔═╡ fde360e8-2c2e-4003-80b2-2615f548899f
std(residuals(t_result))

# ╔═╡ a87c4440-7581-484c-9c32-2468c2a2893c
t_prior_inter = pmf_from_seq(LinRange(49, 50, 41));

# ╔═╡ e6097687-7bf5-4a11-a0fc-77083c766367
t_prior_slope = pmf_from_seq(LinRange(0.0, 0.1, 51));

# ╔═╡ 53fafd86-4c1d-49bb-bd7d-0055a8627ca5
t_prior_sigma = pmf_from_seq(LinRange(0.8, 1.4, 31));

# ╔═╡ eb1c5661-3020-463a-8425-223aad630d06
t_prior = make_joint(t_prior_sigma, t_prior_inter, t_prior_slope);

# ╔═╡ c9505ec9-e952-40ac-9128-5939c7e653ca
begin
	t_xs = complete.x
	t_ys = complete.y
end

# ╔═╡ 47c91c85-5367-4fc5-bfda-06a326699a09
	t_likes = [compute_likelihood(slope, inter, sigma, t_xs, t_ys)  for slope in zs(t_prior) for sigma in xs(t_prior) for inter in ys(t_prior)  ]

# ╔═╡ 26a90ae0-8985-425c-9431-14dcb3ac8c9f
t_likes2 = reshape(t_likes, size(t_prior));

# ╔═╡ 337252f7-e976-4aaf-b7bf-ed6545f4ed34
t_posterior_joint = t_prior * t_likes2;

# ╔═╡ 0fd273d2-e12c-470b-b89e-aa5e92a550a3
plot(marginal(t_posterior_joint, 1))

# ╔═╡ 6db2be87-f449-4593-982b-488654245430
plot(marginal(t_posterior_joint, 2))

# ╔═╡ 20efc171-a14c-44f5-b8a1-e9916daa775e
plot(marginal(t_posterior_joint, 3))

# ╔═╡ 1e3b5739-ec4a-4f0d-8990-a67a90bf2fd8
begin
	Random.seed!(17)
	t_sample = rand(t_posterior_joint, 101)
end

# ╔═╡ 29f6a9cf-ac02-42d1-9a81-fe529484e0e1
begin
	years = 1967:2:2067
	t_xvals = years .- offset
	t_pred_v = [inter .+ slope .* t_xvals .+ rand(Normal(0, sigma), length(t_xvals)) for (sigma, inter, slope) in t_sample]
	t_pred = reduce(hcat, t_pred_v)
end

# ╔═╡ 802be637-8123-494b-9e6a-da1b4c22cd65
length(t_xvals)

# ╔═╡ 30fd5228-42aa-406c-983d-0d6cf73b12f9
t_low, t_median, t_high = percentile(t_pred, [5, 50, 95], dims=1)

# ╔═╡ 8f081796-59d3-4010-9d26-7e7d6a38e004
begin
	dates = Date.(t_xvals .+ t_offset)
	plot(dates, t_median, ribbon=(t_median .- t_low, t_high .- t_median), fillalpha=0.3)
	scatter!(Date.(complete.YEAR), complete.y)
end

# ╔═╡ 0f2b548b-2887-4019-8001-15083fa3a768
t_median[end] - t_median[1]

# ╔═╡ Cell order:
# ╠═80131c5a-eb24-11ec-2d5b-71a8a2588bd7
# ╠═ec9b2808-3cb3-4fb6-9afe-4d70e3fd472c
# ╠═a0cbc648-5c95-42ad-a31a-099b86f9cb21
# ╟─d213924b-4243-4b0a-a80f-de704d6e806f
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
# ╠═dca66cb9-2350-406a-9737-e455ec50b028
# ╠═a9530988-632c-444e-87b5-cfb565b77932
# ╟─f359ce7b-82bc-4b24-91d2-5bdb66a0802c
# ╠═afd8d712-a614-4b82-8f13-edda445eea8f
# ╠═fcf36d6a-6845-4446-b4fd-a5c6ff0b3f84
# ╠═7bfbafd5-17a7-46bd-8dcd-3796e1fa5661
# ╟─da14930f-1a4b-419d-b33f-d4ebf7d0cbc6
# ╟─7e7d2106-a2a1-4b1a-a0ec-3d0b91e2b515
# ╠═7186ccaf-30e5-40f9-bb28-88aa522c10c3
# ╠═ba245d94-9222-432e-aee7-aa360943dfce
# ╠═e0960e05-16f4-4fa1-a8b0-4407835f2f1d
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
# ╠═37dbfd29-0c03-489a-8ed4-6cbdde3b09c6
# ╠═4011772f-4c2b-4783-9dcd-e11eb5d7089d
# ╠═20e5609b-4415-4d46-8d17-2c9ab5a5ccd8
# ╠═58741dcc-1b8c-4830-90d2-b91ec3b7551a
# ╠═f6fc984c-5692-4ba7-843c-193af68bd6de
# ╟─cbf40631-39f3-4bf8-aea7-46da201a8bfc
# ╠═b522db3c-6cd2-40a2-8d28-96702874c3fb
# ╠═b20fe8f4-8fcc-4633-83a9-be82f5dee6be
# ╟─bd1febb8-b7d5-4e0a-925f-0a9b8094a286
# ╠═776ebc1a-9e39-44b6-9691-b4f21a763fdf
# ╠═c2c7650c-5c22-40bd-aaf3-579896f83788
# ╠═8e017761-c277-462e-afc0-b668b538da0b
# ╠═04200a27-78c7-4a47-b0c9-78fe65c11fad
# ╠═b065d6b4-9516-439b-8461-12f4bb2853e7
# ╠═161c906b-0bd8-4003-84d1-8c2b0f5c316a
# ╠═5ded5619-dbd9-438f-899e-f598ca23cec7
# ╠═fd86f76a-996b-4a29-96a5-2784248af278
# ╠═f800cc5c-c871-4d1b-aff6-6a345d93c0b6
# ╠═f052d6d3-40dd-44fc-ae15-3ba68b498cfe
# ╠═7c8e8b91-b7de-4b21-9a12-6027e0452768
# ╠═e34c87fa-b51b-479c-a467-f63de478d1d8
# ╠═fb2077d4-27c0-4fc9-a234-e6cc48fd10d6
# ╠═cdc1fc69-db45-4b22-bb7a-80e625aa6840
# ╠═b0d16dc7-4ab0-4593-83bb-a95e13bf048b
# ╠═bf7d6f38-1eaa-4b82-b526-8bc5b59ec6b5
# ╟─5e42f35e-1f5f-4703-9aa3-8e49048dc73e
# ╠═56cac61a-cbb2-46f3-b1b3-b74c6f373986
# ╠═42bc7287-533e-4773-8e86-2276cabc77ba
# ╠═6b52f227-14f5-4994-9daf-4d428777589d
# ╠═300b234b-9307-4701-b739-c7c1d9500535
# ╠═88cadc6e-6d80-4eef-9ea9-d41287105017
# ╠═b5d1dd33-b19d-4d8c-863d-c4a993ea51d0
# ╠═57738a5c-a5fd-45da-a2aa-113ad6d6b6bf
# ╠═aabebba3-09e1-4790-9607-33d9b33d4122
# ╠═aa6c1c01-b833-4cb8-a565-d17c53baae2c
# ╠═7058a81f-2a16-42a8-a9b7-ed4f3c632206
# ╟─2bf2d0af-cdc4-4ee8-8960-2eaf67441209
# ╠═077f2509-8255-446b-b5ae-5f2eb92b6c35
# ╠═eefcf367-1358-4d42-8171-18d6bc082823
# ╠═25fbd848-4057-452f-a289-a0492298e768
# ╠═ef1f93e6-f1fa-477c-84c9-dd5bd72fbfd1
# ╠═7556a851-60f5-44b6-8802-bc296eaa0f72
# ╠═fa59019d-856c-4cbd-9bf3-1b70320052db
# ╠═4406a122-898a-40da-8c4a-a41c852094a2
# ╠═fa021aeb-28c6-4b66-9608-691000b37a02
# ╠═d525d152-56dd-4e80-a1fd-8ce431d5be5f
# ╠═31406bc5-102d-4eb4-8f11-8dd064ba29f5
# ╠═baaaa99f-0334-4d44-9cb2-2149c32bbab8
# ╠═588304b9-1037-4fab-8f13-9321ebea29d5
# ╠═f5043e74-c3c6-4888-80fd-cf4f20d6ff9e
# ╠═0603607d-8666-41fb-967f-5419020c0200
# ╠═76f15456-54c4-43a9-8ed7-f42be43ee3b5
# ╠═51c75fff-7634-44fc-aaac-a0fdee149da6
# ╠═ed2127a8-ecc2-4400-9219-c94c812f8a4c
# ╠═e93241bb-d2b8-45ad-9470-011bb97c6803
# ╠═fde360e8-2c2e-4003-80b2-2615f548899f
# ╠═a87c4440-7581-484c-9c32-2468c2a2893c
# ╠═e6097687-7bf5-4a11-a0fc-77083c766367
# ╠═53fafd86-4c1d-49bb-bd7d-0055a8627ca5
# ╠═eb1c5661-3020-463a-8425-223aad630d06
# ╠═c9505ec9-e952-40ac-9128-5939c7e653ca
# ╠═47c91c85-5367-4fc5-bfda-06a326699a09
# ╠═26a90ae0-8985-425c-9431-14dcb3ac8c9f
# ╠═337252f7-e976-4aaf-b7bf-ed6545f4ed34
# ╠═0fd273d2-e12c-470b-b89e-aa5e92a550a3
# ╠═6db2be87-f449-4593-982b-488654245430
# ╠═20efc171-a14c-44f5-b8a1-e9916daa775e
# ╠═1e3b5739-ec4a-4f0d-8990-a67a90bf2fd8
# ╠═29f6a9cf-ac02-42d1-9a81-fe529484e0e1
# ╠═802be637-8123-494b-9e6a-da1b4c22cd65
# ╠═30fd5228-42aa-406c-983d-0d6cf73b12f9
# ╠═8f081796-59d3-4010-9d26-7e7d6a38e004
# ╠═0f2b548b-2887-4019-8001-15083fa3a768
