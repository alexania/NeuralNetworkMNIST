using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MNIST.IO;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace NeuralNetworkHandwriting
{
  public class PyNetwork
  {
    private static Random _rand = new Random();
    private static Stopwatch _timer = new Stopwatch();

    private long _empties = 0;
    private long _update = 0;
    private long _zip = 0;

    private long _setup1 = 0;
    private long _setup2 = 0;
    private long _setup3 = 0;
    private long _feedForward = 0;
    private long _cost = 0;
    private long _layers = 0;

    public PyNetwork(int[] sizes)
    {
      _timer.Start();
      Sizes = sizes;
      NumLayers = sizes.Length;
      NumActivations = NumLayers - 1;
      Biases = Sizes.Skip(1).Select(t => Matrix<double>.Build.Random(t, 1, new Normal())).ToArray();
      Weights = Sizes.Take(Sizes.Length - 1).Zip(Sizes.Skip(1), (x, y) => Matrix<double>.Build.Random(y, x, new Normal())).ToArray();
      Console.WriteLine($"Initial Weights & Biases created: {_timer.ElapsedMilliseconds}ms");
      _timer.Restart();
    }

    public int NumLayers { get; set; }
    public int NumActivations { get; set; }

    public int[] Sizes { get; }

    public Matrix<double>[] Biases { get; set; }
    public Matrix<double>[] Weights { get; set; }

    public Matrix<double> FeedForward(Matrix<double> a)
    {
      for (var i = 0; i < NumActivations; i++)
      {
        a = Sigmoid(Weights[i] * a + Biases[i]);
      }
      return a;
    }

    public void GradientDescent(IEnumerable<TestCase> trainingData, int epochs, int miniBatchSize, double eta, IEnumerable<TestCase> testData = null)
    {
      var convertedTrainingData = ConvertData(trainingData);
      var convertedTestData = ConvertData(testData);

      Console.WriteLine($"Training Data Converted: {_timer.ElapsedMilliseconds}ms");
      _timer.Restart();

      for (var j = 0; j < epochs; j++)
      {
        Shuffle(convertedTrainingData);

        _empties = 0;
        _setup1 = 0;
        _feedForward = 0;
        _cost = 0;
        _layers = 0;
        _zip = 0;
        _update = 0;

        Console.WriteLine($"Training Data Shuffled: {_timer.ElapsedMilliseconds}ms");
        _timer.Restart();

        var miniBatches = new List<List<ImageData>>();
        var numBatches = (int)Math.Floor(convertedTrainingData.Count * 1.0 / miniBatchSize);
        for (int m = 0; m < numBatches; m++)
        {
          miniBatches.Add(convertedTrainingData.Skip(m * miniBatchSize).Take(miniBatchSize).ToList());
        }

        Console.WriteLine($"Mini Batches Created: {_timer.ElapsedMilliseconds}ms");
        _timer.Restart();

        foreach (var miniBatch in miniBatches)
        {
          UpdateMiniBatch(miniBatch, eta);
        }

        //Console.WriteLine($"Empties created: {_empties}ms");
        //Console.WriteLine($"BackProp Setup Done: {_setup1}ms");
        ////Console.WriteLine($"BackProp Matrix Multiply Done: {_setup2}ms");
        ////Console.WriteLine($"BackProp Sigmoid Done: {_setup3}ms");
        //Console.WriteLine($"BackProp FeedForward Done: {_feedForward}ms");
        Console.WriteLine($"BackProp Cost Done: {_cost}ms");
        Console.WriteLine($"BackProp Layers Done: {_layers}ms");
        Console.WriteLine($"Zip Done: {_zip}ms");
        Console.WriteLine($"Weights Updated: {_update}ms");

        //Console.WriteLine($"MiniBatchUpdated: {_timer.ElapsedMilliseconds}ms");
        _timer.Restart();

        if (testData != null)
        {
          Console.WriteLine($"Epoch {j}: {Evaluate(convertedTestData)} / {convertedTestData.Count}.");
        }
        else
        {
          Console.WriteLine($"Epoch {j} complete.");
        }

        Console.WriteLine($"Model Evaluated: {_timer.ElapsedMilliseconds}ms");
        _timer.Restart();
      }
    }

    private Matrix<double> Sigmoid(Matrix<double> z)
    {
      return 1.0 / (1.0 + Matrix<double>.Exp(-z));
    }

    private Matrix<double> SigmoidPrime(Matrix<double> z)
    {
      return Sigmoid(z).PointwiseMultiply(1 - Sigmoid(z));
    }

    private int Evaluate(IList<ImageData> testData)
    {
      var numCorrect = 0;
      foreach (var t in testData)
      {
        var output = FeedForward(t.Image);
        var guess = ArgMax(output);
        if (guess == t.LabelValue)
        {
          numCorrect++;
        }
      }
      return numCorrect;
    }

    private void UpdateMiniBatch(IList<ImageData> miniBatch, double eta)
    {
      var nablaB = new Matrix<double>[NumActivations];
      var nablaW = new Matrix<double>[NumActivations];

      //_empties += _timer.ElapsedMilliseconds;
      _timer.Restart();

      for (var j = 0; j < miniBatch.Count; j++)
      {
        var time = _timer.ElapsedMilliseconds;

        var t = miniBatch[j];

        BackProp(t.Image, t.Label, out var deltaNablaB, out var deltaNablaW);

        _update += _timer.ElapsedMilliseconds - time;
        time = _timer.ElapsedMilliseconds;

        for (var i = 0; i < NumActivations; i++)
        {
          if (nablaB[i] == null)
          {
            nablaB[i] = Matrix<double>.Build.DenseOfMatrix(deltaNablaB[i]);
          } else
          {
            nablaB[i] = nablaB[i] + deltaNablaB[i];
          }
          if (nablaW[i] == null)
          {
            nablaW[i] = Matrix<double>.Build.DenseOfMatrix(deltaNablaW[i]);
          }
          else
          {
            nablaW[i] = nablaW[i] + deltaNablaW[i];
          }
        }

        _zip += _timer.ElapsedMilliseconds - time;
        //_timer.Restart();
      }

      _cost += _timer.ElapsedMilliseconds;

      for (var i = 0; i < NumActivations; i++)
      {
        Weights[i] = Weights[i] - (eta / miniBatch.Count) * nablaW[i];
        Biases[i] = Biases[i] - (eta / miniBatch.Count) * nablaB[i];
      }

      //_update += _timer.ElapsedMilliseconds;
      //_timer.Restart();
    }

    private void BackProp(Matrix<double> x, Matrix<double> y, out Matrix<double>[] deltaNablaB, out Matrix<double>[] deltaNablaW)
    {

      deltaNablaB = new Matrix<double>[NumActivations];
      deltaNablaW = new Matrix<double>[NumActivations];

      var activation = x;
      var activations = new List<Matrix<double>> { x };
      var zs = new List<Matrix<double>>();

      //_setup1 += _timer.ElapsedMilliseconds;
      //_timer.Restart();

      for (var i = 0; i < NumActivations; i++)
      {
        var z = Weights[i] * activation + Biases[i];

        zs.Add(z);
        activation = Sigmoid(z);
        activations.Add(activation);
      }

      //_feedForward += _timer.ElapsedMilliseconds;
      //_timer.Restart();

      var delta = CostDerivative(activations[activations.Count - 1], y).PointwiseMultiply(SigmoidPrime(zs[zs.Count - 1]));
      deltaNablaB[deltaNablaB.Length - 1] = delta;
      deltaNablaW[deltaNablaW.Length - 1] = delta * activations[activations.Count - 2].Transpose();

      //_cost += _timer.ElapsedMilliseconds;
      //_timer.Restart();

      for (var l = 2; l < NumLayers; l++)
      {
        var z = zs[zs.Count - l];
        var sp = SigmoidPrime(z);

        delta = (Weights.ElementAt(NumLayers - 1 - l + 1).Transpose() * delta).PointwiseMultiply(sp);
        deltaNablaB[deltaNablaB.Length - l] = delta;
        var time = _timer.ElapsedMilliseconds;
        deltaNablaW[deltaNablaW.Length - l] = delta * activations[activations.Count - l - 1].Transpose();
        _layers += _timer.ElapsedMilliseconds - time;
      }

      //_layers += _timer.ElapsedMilliseconds - time;
      //_timer.Restart();
    }

    private Matrix<double> CostDerivative(Matrix<double> outputActivations, Matrix<double> y)
    {
      return outputActivations - y;
    }

    private static int ArgMax(Matrix<double> a)
    {
      var m = 0;
      var e = a.Enumerate().ToArray();
      for (var i = 0; i < e.Length; i++)
      {
        if (e[i] > e[m])
        {
          m = i;
        }
      }
      return m;
    }
    
    private static void Shuffle<T>(IList<T> list)
    {
      var n = list.Count;
      while (n > 1)
      {
        n--;
        int k = _rand.Next(n + 1);
        T value = list[k];
        list[k] = list[n];
        list[n] = value;
      }
    }

    private IList<ImageData> ConvertData(IEnumerable<TestCase> data)
    {
      return data.Select(t => new ImageData(t.Image, t.Label)).ToList();
    }

    public struct ImageData
    {
      public ImageData(byte[,] image, int label)
      {
        LabelValue = label;
        Label = ConvertLabel(label);
        Image = ConvertImage(image);
      }

      public Matrix<double> Image { get; set; }
      public Matrix<double> Label { get; set; }
      public int LabelValue { get; set; }

      private static Matrix<double> ConvertImage(byte[,] image)
      {
        var d = new double[image.Length, 1];
        var i = 0;
        for (var x = 0; x < image.GetLength(0); x++)
        {
          for (var y = 0; y < image.GetLength(1); y++)
          {
            d[i++, 0] = (double)image[x, y] / 256.0;
          }
        }
        return Matrix<double>.Build.DenseOfArray(d);
      }

      private static Matrix<double> ConvertLabel(int label)
      {
        var d = new double[10, 1];
        d[label, 0] = 1;
        return Matrix<double>.Build.DenseOfArray(d);
      }
    }
  }
}
