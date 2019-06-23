using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace NetCoreMachineLearningTesting
{

    class FeedBackTrainingData
    {
        public string FeedBackText { get; set; }
        [ColumnName(name: "Label")]
        public bool IsGood { get; set; }
    }

    class FeedBackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }
    }

    class Program
    {

        static List<FeedBackTrainingData> trainingData = new List<FeedBackTrainingData>();
        static List<FeedBackTrainingData> testData = new List<FeedBackTrainingData>();

        static void LoadTrainingData()
        {
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is good",
                IsGood = true
            });

            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is horrible",
                IsGood = false
            });

            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is excelent!",
                IsGood = true
            });

            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "could have been better",
                IsGood = false
            });

            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "fuck this food",
                IsGood = false
            });

            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "I would never come back",
                IsGood = false
            });

            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "yum!",
                IsGood = true
            });

            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "10/10",
                IsGood = true
            });
        }

        static void LoadTestData()
        {
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Very yummy",
                IsGood = true
            });

            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "excelent meal",
                IsGood = true
            });

            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "10/10 would come again!",
                IsGood = true
            });

            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "gross! never coming back.",
                IsGood = false
            });

            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "terrible service",
                IsGood = false
            });

            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad service",
                IsGood = false
            });

            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad food",
                IsGood = false
            });

            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "fucking terrible",
                IsGood = false
            });
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Loading Data...");
            LoadTrainingData();

            var mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromEnumerable(trainingData);

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "FeedBackText")
                .Append(mlContext.BinaryClassification.Trainers.FastTree
                (numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 1));

            var model = pipeline.Fit(dataView);


            LoadTestData();

            IDataView dataView1 = mlContext.Data.LoadFromEnumerable<FeedBackTrainingData>(testData);

            var predictions = model.Transform(dataView1);
            var matrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine(matrics.Accuracy);


            Console.WriteLine("Enter a feedback string:");

            string feedbackString = Console.ReadLine().ToString();



            var predictionFunction = mlContext.Model.CreatePredictionEngine<FeedBackTrainingData, FeedBackPrediction>(model);

            var feedbackInput = new FeedBackTrainingData()
            {
                FeedBackText = feedbackString
            };

            var feedbackPrediceted = predictionFunction.Predict(feedbackInput);
            Console.WriteLine("Predicted: " + feedbackPrediceted.IsGood);
            Console.ReadLine();

            //-Instead of mlContext.CreateStreamingDataView() you have to use mlContext.Data.LoadFromEnumerable()
            //- mlContext.Transforms.Text.FeaturizeText has a different order for the parameters, so correct for this example would be mlContext.Transforms.Text.FeaturizeText("Features", "FeedbackText")
            //- To get mlContext.BinaryClassification.Trainers.FastTree() you also have to use NuGet to install Microsoft.ML.FastTree
            //- mlContext.BinaryClassification.Trainers.FastTree() varible names have changed, numLeaves = numberOfLeaves; numTrees = numberOfTrees, minDataPointsInLeaves = minimumExampleCountPerLeaf
            //- The[Column] Property is now[ColumnName] in Microsoft.ML.Data
            //- Instead of model.MakePredictionFunction() you now have to use mlContext.Model.CreatePredictionEngine<FeedbackTrainingData, FeedbackPrediction>(model)
        }
    }
}
