
using Distributions

include("datastats.jl")

abstract type NBModel{C} end

#####################################
#####  Multinomial Naive Bayes  #####
#####################################

"""
Multinomial Naive Bayes classifier model struct

classeCounts : hash table with values Int64 and key objects
    Count of ocurrences of each class
Xcounts : hash table with values is array of numbers and key objects
    Count/sum of occurrences of each var
Xtotals : array of numbers
    Total occurrences of each var
obs : Int64
    Total number of seen observations

"""


mutable struct MultinomialNB{C} <: NBModel{C} 
    classeCounts::Dict{C, Int64}           
    Xcounts::Dict{C, Vector{Number}}       
    Xtotals::Vector{Number}                
    obs::Int64                             
end



"""
Multinomial Naive Bayes classifier
classes : array of objects
    Class names
Nvars : Int64
    Number of variables in observations
alpha : Number (optional, default 1)
    Smoothing parameter. E.g. if α = 1, each variable in each class
    is believed to have 1 observation by default
    (α is used to avoid the probability of 0, it's added to the count as black box and usually it value is set to 1)

Returns Multinomial Naive Bayes Model
"""


function MultinomialNB(classes::Vector{C}, Nvars::Int64; α=1) where C 
    classeCounts = Dict(zip(classes, ones(Int64, length(classes)) * α))
    Xcounts = Dict{C, Vector{Int64}}()
    for c in classes
        Xcounts[c] = ones(Int64, Nvars) * α
    end
    Xtotals = ones(Float64, Nvars) * α * length(classeCounts)
    MultinomialNB{C}(classeCounts, Xcounts, Xtotals, sum(Xtotals))
end



#####################################
######  Gaussian Naive Bayes  #######
#####################################


"""
Gaussian Naive Bayes classifier model struct

classeCounts : hash table with values Int64 and key objects
    Count of ocurrences of each class
classeStats : hash table with values is array of dataset 
   aggregative data statistics
gaussians : hash table with values is array of multivariate Gaussian distribution
    precomputed distribution
obs : Int64
    Total number of seen observations

"""



mutable struct GaussianNB{C} <: NBModel{C}
    classeCounts::Dict{C, Int64}         
    classeStats::Dict{C, DataStats}       
    gaussians::Dict{C, MvNormal}        
    obs::Int64                       
end





"""
Gaussian Naive Bayes classifier
classes : array of objects
    Class names
Nvars : Int64
    Number of variables in observations

Returns Gaussian Naive Bayes Model
"""



function GaussianNB(classes::Vector{C}, Nvars::Int64) where C
    classeCounts = Dict(zip(classes, zeros(Int64, length(classes))))
    classeStats = Dict(zip(classes, [DataStats(Nvars, 2) for i=1:length(classes)]))
    gaussians = Dict{C, MvNormal}()
    GaussianNB{C}(c_counts, c_stats, gaussians, 0)
end



