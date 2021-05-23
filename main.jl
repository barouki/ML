include("supervised_learning/KNN/IncludesAndExports.jl")

function main()
    println("ACCURACY")
    xTrain,yTrain,xTest,yTest = LoadAndPrepareData("/Users/barouki/Downloads/knn/iris.csv", 0.7, [:"sepal.length", :"sepal.width", :"petal.length", :"petal.width"], [:variety]) 
    
    knn = KNN(xTrain, yTrain)
    predicted = ClassificationKNNPrediction(knn, xTest,5)
    println(Accuracy(predicted,yTest))
end

main()
