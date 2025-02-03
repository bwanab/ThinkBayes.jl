using Random
using DataFrames

function mart_bet(bet, pot)
    results = []
    pots = []
    iters = 0

    println("iters, result, pot")
    while pot > 0
        result = Random.rand((-bet, bet))
        if result < 0
            bet = bet * 2
        end
        pot += result
        iters += 1
        println(iters, ", ", result, ", ", pot)
        append!(results, result)
        append!(pots, pot)

    end
    results
end

results, pots = mart_bet(5, 10000)
df = DataFrame(results=results, pots=pots)
plot(df)
