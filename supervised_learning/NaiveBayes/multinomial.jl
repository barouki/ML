
"""
Set Multinomial Naive Bayes Model With the given x an y set
m : MultinomialNB
    Multinomial Naive Bayes Model
X : matrix of Int
    xSet
y : array of object
    ySet 
"""



function SetModel(m::MultinomialNB, X::Matrix{Int64}, y::Vector{C}) where C
    for j=1:size(X, 2)         # foreach colums in the Matrix X
        c = y[j]               
        m.classeCounts[c] += 1
        m.Xcounts[c] .+= X[:, j]    # .+= the dot refers to foreach i in list : list[i] += y[i] 
        m.Xtotals += X[:, j]
        m.obs += 1
    end
    return m
end


"""
Calculate log P(x|C) given an array of int
m : MultinomialNB
    Multinomial Naive Bayes Model
x : array of Int64
    Number of variables in observations
c : object
    class
"""

function logprobXgivenC(m::MultinomialNB, x::Vector{Int64}, c::C) where C     
    XpriorsForC = m.x_counts[c] ./ sum(m.x_counts[c])
    probsXGivenC = XpriorsForC .^ x
    logprob = sum(log(probsXGivenC))
    return logprob
end


"""
Calculate log P(x|C) given matrix of int
m : MultinomialNB
    Multinomial Naive Bayes Model
X : matrix of Int64
    Number of variables in observations
c : object
    class
"""

function logprobXgivenC(m::MultinomialNB, X::Matrix{Int64}, c::C) where C     # Calculate log P(x|C) given X a matrix
    XpriorsForC = m.x_counts[c] ./ sum(m.x_counts[c])
    probsXGivenC = XpriorsForC .^ X
    logprob = sum(log.(probsXGivenC), dims=1)
    return dropdims(logprob, dims=1)
end




