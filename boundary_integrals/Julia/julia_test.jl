using Statistics
using Plots

f(x) = (x.^2).*exp.(-x.^2)

a = range(0.0,stop=5.0,length=100)

plt = plot(a, f(a))
png(plt, "plot")
