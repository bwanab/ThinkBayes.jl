module HasA
export HasA, make_tst, Tst2, make_tst2, talk
import Base.show

struct Tst
    v::Vector
end

make_tst(v) = Tst(v)

show(io::IO, t::Tst) = println(t.v)

struct Tst2
    s::String
    t::Tst
end

make_tst2(s, v) = Tst2(s, Tst(v))

function show(io::IO, t::Tst2)
    println(t.s)
    show(t.t)
end

end
