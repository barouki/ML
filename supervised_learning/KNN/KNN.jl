struct KNN
    x::DataFrames.DataFrame
    y::DataFrames.DataFrame
end


########################################################################################################
# Classifier KNN ; getting the k nearest neibghors and rendering the most frequent value as predection #
########################################################################################################


function ClassificationKNNPrediction(data::KNN, testData::DataFrames.DataFrame, k)
    predections=[]
    for i in 1:size(testData, 1)
        sourceInstance = Array(testData[i,:])
        distances = []
        for j in 1:size(data.x, 1)   
            destInstance = Array(data.x[j,:])
            distance = EuclDist(sourceInstance, destInstance)
            push!(distances, distance)
        end
    sortedIndex = sortperm(distances) 
    neighbor = Array(data.y)[sortedIndex[1:k]]
    predictedLabel = majority_vote(neighbor)
    push!(predections, predictedLabel)
    end
    return predections
end


#########################################################################################################
# Regressor KNN ; getting the k nearest neibghors and rendering the average of its values as predection #
#########################################################################################################

function RegressionKNNPrediction(data::KNN, testData::DataFrames.DataFrame, k=0.7)
    predections=[]
    for i in 1:size(testData, 1)
        sourceInstance = Array(testData[i,:])
        distances = []
        for j in 1:size(data.x, 1)   
            destInstance = Array(data.x[j,:])
            distance = EuclDist(sourceInstance, destInstance)
            push!(distances, distance)
        end
    sortedIndex = sortperm(distances) 
    neighbor = Array(data.y)[sortedIndex[1:k]]
    push!(predections, mean(neighbor))
    end
    return predections
end

###########################
# GET MAJORITY VOTE VALUE #
###########################

function majority_vote(neighbors::Vector)
    counts = countmap(neighbors)
    result = neighbors[1]
    max_value = 0
    for (key, value) in counts
        if value > max_value
            result, max_value = key, value
        end
    end
    return result
end










