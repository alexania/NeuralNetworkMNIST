using System;
using System.Linq;

namespace NeuralNetworkHandwriting
{
  public class NeuralNetwork
  {
    private static Random _rng = new Random();

    public NeuralNetwork(int[] sizes)
    {
      NumLayers = sizes.Length - 1;
      Layers = new Layer[NumLayers];
      for (var i = 0; i < NumLayers; i++)
      {
        Layers[i] = new Layer(sizes[i], sizes[i + 1]);
      }
    }

    public int NumLayers { get; set; }

    public Layer[] Layers { get; set; }

    public void GradientDescent(TestData[] trainingData, int epochs, int miniBatchSize, double eta, TestData[] testData = null)
    {
      for (var j = 0; j < epochs; j++)
      {
        Shuffle(trainingData);

        var numBatches = (int)Math.Ceiling(trainingData.Length * 1.0 / miniBatchSize);

        for (int m = 0; m < numBatches; m++)
        {
          var miniBatch = trainingData.Skip(m * miniBatchSize).Take(miniBatchSize).ToArray();
          UpdateMiniBatch(miniBatch, eta);
        }

        if (testData != null)
        {
          Console.WriteLine($"Epoch {j}: {Evaluate(testData)} / {testData.Length}.");
        }
        else
        {
          Console.WriteLine($"Epoch {j} complete.");
        }
      }
    }

    public int Evaluate(TestData[] testData)
    {
      var numCorrect = 0;
      for (var i = 0; i < testData.Length; i++)
      {
        var a = testData[i].Input;
        for (var n = 0; n < NumLayers; n++)
        {
          a = Layers[n].FeedForward(a);
        }

        var o = testData[i].Output;
        var maxIndex = 0;
        for (var n = 0; n < a.Length; n++)
        {
          if (a[n] > a[maxIndex])
          {
            maxIndex = n;
          }
        }
        numCorrect += o[maxIndex] == 1 ? 1 : 0;
      }
      return numCorrect;
    }

    private void UpdateMiniBatch(TestData[] miniBatch, double eta)
    {
      for (var j = 0; j < miniBatch.Length; j++)
      {
        BackProp(miniBatch[j]);
      }

      for (var i = 0; i < NumLayers; i++)
      {
        Layers[i].ApplyDeltas(eta / miniBatch.Length);
      }
    }

    private void BackProp(TestData x)
    {
      var activation = x.Input;

      // Feed Forward
      for (var i = 0; i < NumLayers; i++)
      {
        activation = Layers[i].TrainForward(activation);
      }

      // Calculate Cost
      var cost = CostDerivative(activation, x.Output);

      // Calculate Deltas
      for (var i = NumLayers - 1; i >= 0; i--)
      {
        var delta = Layers[i].UpdateDeltas(cost);
        cost = Layers[i].TrainBackward(delta);
      }
    }

    private double[] CostDerivative(double[] a, double[] e)
    {
      var x = new double[a.Length];
      for (var i = 0; i < a.Length; i++)
      {
        x[i] = a[i] - e[i];
      }
      return x;
    }

    private static void Shuffle<T>(T[] array)
    {
      var n = array.Length;
      while (n > 1)
      {
        n--;
        int k = _rng.Next(n + 1);
        T value = array[k];
        array[k] = array[n];
        array[n] = value;
      }
    }
  }
}
