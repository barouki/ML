module IncludesAndExports    


using DataFrames, CSV, DataStructures
using Random, Statistics, StatsBase


# Exports internal
export NaiveBayes

# Include models
include("NaiveBayesTypes.jl")

# Include utils
include("../../utils/utils.jl")

include("Gaussian.jl")
include("multinomial.jl")

export  NBModel,
        MultinomialNB,
        GaussianNB
end