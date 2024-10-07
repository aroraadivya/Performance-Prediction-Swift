import CreateML
import Foundation
import TabularData

//import dataset
let dataFrame = try DataFrame(contentsOfCSVFile: URL( filePath: "/Users/divyaarora/Desktop/Student_Performance.csv"))

//split data into training and testing data
let (trainigData, testingData) = dataFrame.randomSplit(by: 0.8)

//convert the slices to dataframe
let trainingDataFrame = DataFrame(trainigData)
let testingDataFrame = DataFrame(testingData)

//Apply MLRegressor on testing data
let regressor = try MLRegressor(trainingData: trainingDataFrame, targetColumn: "Performance Index")

//evaluate the model on testing data
let evaluationMetrics = regressor.evaluation(on: testingDataFrame)

//result of testing
print(evaluationMetrics.maximumError)
print(evaluationMetrics.rootMeanSquaredError)

//add metadata
let metaData = MLModelMetadata(author: "Divya", shortDescription: "This model predicts student performance based on few known values", license: "Apache", version: "1.0")

//save the model to file system
try regressor.write(to: URL(filePath: "/Users/divyaarora/Desktop/xyz.mlmodel"), metadata: metaData)
