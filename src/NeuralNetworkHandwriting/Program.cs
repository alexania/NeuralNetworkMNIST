using MNIST.IO;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkHandwriting
{
  class Program
  {
    static void Main(string[] args)
    {
      var data = FileReaderMNIST.LoadImagesAndLables(
    "./data/train-labels-idx1-ubyte.gz",
    "./data/train-images-idx3-ubyte.gz");
      var training = data.Take(50000);

      var test = FileReaderMNIST.LoadImagesAndLables(
        "./data/t10k-labels-idx1-ubyte.gz",
        "./data/t10k-images-idx3-ubyte.gz");

      //  var network = new Network(new[] { 784, 30, 10 });
      //  network.GradientDescent(training, 30, 10, 3.0, test);

      Console.WriteLine("Creating a " + 784 + "-" + 30 +  "-" + 10 + " neural network");
      var network = new NeuralNetwork(new[] { 784, 100, 10 });

      var trainingData = ConvertData(training);
      var testData = ConvertData(test);

      network.GradientDescent(trainingData, 30, 10, 3.0, testData);
    }

    private static TestData[] ConvertData(IEnumerable<TestCase> data)
    {
      return data.Select(t => new TestData(t.Image, t.Label)).ToArray();
    }
  }
}
