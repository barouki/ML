



"""
Count number of observations

x_sums : hash table with values Int64 and key objects
    sum(x_i)
cross_sums : matrix of Float64
    Lower-triangular matrix  (sum(x_i'*x_i))
n_obs : UInt64
    Number of observations
obs_axis : Int64
    Observation axis

"""


mutable struct DataStats
    x_sums::Vector{Float64}     
    cross_sums::Matrix{Float64} 
    n_obs::UInt64                
    obs_axis::Int64              
                                 
    function DataStats(n_vars, obs_axis=1)
        @assert obs_axis == 1 || obs_axis == 2
        new(zeros(Float64, n_vars), zeros(Float64, n_vars, n_vars), 0, obs_axis)
    end
end