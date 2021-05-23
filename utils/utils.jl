########################################################################################
# Split data into train and test -                                                     #
########################################################################################

function split(data, split_percent)

    # data is the data set that we've read from the csv filename
    # split_percent is the percentage of spliting the train set :)
    n = nrow(data)                                                      # number of rows in the dataset
    ind = Random.shuffle(1:n)                                           # Generate a random permutation of 1 to N
    train_ind = view(ind, 1:floor(Int, split_percent*n))                           # slices the dataset into train and test sets'  
    test_ind = view(ind, (floor(Int, split_percent*n)+1):n)
    return data[train_ind,:], data[test_ind,:]
end

########################################################################################
# Load And Prepare Data - loading and transfroming data from csv file into a DataFrame #
########################################################################################

function LoadAndPrepareData(filename, split_percent, xAtt, yAtt)  
    df = DataFrame(CSV.File(filename, header=true))                     # Reading the csv file and cast it to dataFrame
    trainData, testData = split(df,split_percent)
    
    xTrain = trainData[:, xAtt]                                         # Deffining the x attributes and the y's (we will predicte y)
    yTrain = trainData[:, yAtt]
    xTest = testData[:, xAtt]
    yTest = testData[:, yAtt]


    return xTrain,yTrain,xTest,yTest
end

#################################
#euclidean distance calcualtion #
#################################

function EuclDist(sourceInstance, destInstance)                               
    sum = 0
    for i in 1:length(sourceInstance)
        sum += (destInstance[i] - sourceInstance[i]) ^ 2
    end
    return sqrt(sum)
end


function Accuracy(predicted, yTest)
    accurate = 0
    yTestArray = Array(yTest)
     for i in 1:length(predicted)
         if yTestArray[i] == predicted[i]
             accurate += 1
         end

     end
     return accurate/length(predicted)
end


function MeanSquaredError(predicted, yTest)
    MSQ = 0
    yTestArray = Array(yTest)
     for i in 1:length(predicted)
        MSQ += (yTestArray[i] - predicted[i] )^2
     end
     return MSQ/length(predicted)
end
