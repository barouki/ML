
"""
Set Gaussian Naive Bayes Model With the given x an y set
m : GaussianNB
    Gaussian Naive Bayes Model
X : matrix of Float
    xSet
y : array of object
    ySet 
"""



function SetModel(m::GaussianNB, X::Matrix{Float64}, y::Vector{C}) where C
    Nvars = size(X, 1)
    for j=1:size(X, 2)         # foreach colums in the Matrix X
        c = y[j]               
        m.classeCounts[c] += 1
        updatestats(m.classeStats[c], reshape(X[:, j], Nvars, 1))   
        m.obs += 1
    end
    # precompute distributions for each class
    for c in keys(m.classeCounts)
        m.gaussians[c] = MvNormal(mean(m.classeStats[c]), cov(m.classeStats[c]))
    end
    return m
end


"""
Calculate log P(x|C) given an array of int
m : GaussianNB
    Gaussian Naive Bayes Model
x : array of Float
    Number of variables in observations
c : object
    class
"""

function logprobXgivenC(m::GaussianNB, x::Vector{Float64}, c::C) where C     
    return return logpdf(m.gaussians[c], x)
end


"""
Calculate log P(x|C) given matrix of int
m : GaussianNB
    Gaussian Naive Bayes Model
X : matrix of Float64
    Number of variables in observations
c : object
    class
"""

function logprobXgivenC(m::GaussianNB, X::Matrix{Float64}, c::C) where C     # Calculate log P(x|C) given X a matrix
    return logpdf(m.gaussians[c], X)
end




