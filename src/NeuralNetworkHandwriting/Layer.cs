using System;

namespace NeuralNetworkHandwriting
{
  public struct TestData
  {
    public TestData(byte[,] input, int output)
    {
      Input = ConvertImage(input);
      Output = ConvertLabel(output);
    }

    public double[] Input { get; set; }

    public double[] Output { get; set; }

    private static double[] ConvertImage(byte[,] image)
    {
      var d = new double[image.Length];
      var i = 0;
      for (var x = 0; x < image.GetLength(0); x++)
      {
        for (var y = 0; y < image.GetLength(1); y++)
        {
          d[i++] = image[x, y] / 256.0;
        }
      }
      return d;
    }

    private static double[] ConvertLabel(int label)
    {
      var d = new double[10];
      d[label] = 1;
      return d;
    }
  }

  public class Layer
  {
    private int _numInputs;
    private int _numOutputs;

    private double[] _activation;
    private double[] _z;

    private double[,] _deltaWeights;
    private double[] _deltaBiases;

    private static Random _rng = new Random();

    public Layer(int numInputs, int numOutputs)
    {
      _numInputs = numInputs;
      _numOutputs = numOutputs;

      Inputs = new double[numInputs];
      Outputs = new double[numOutputs];
      Weights = new double[numOutputs, numInputs];
      Biases = new double[numOutputs];

      _deltaWeights = new double[numOutputs, numInputs];
      _deltaBiases = new double[numOutputs];

      SetInitialWeights();
      SetInitialBiases();
    }

    public double[] Inputs { get; set; }

    public double[] Outputs { get; set; }

    public double[,] Weights { get; set; }

    public double[] Biases { get; set; }

    public double[] FeedForward(double[] input)
    {
      var z = new double[_numOutputs];

      for (var x = 0; x < _numOutputs; x++)
      {
        for (var y = 0; y < _numInputs; y++)
        {
          z[x] += input[y] * Weights[x, y];
        }
        z[x] += Biases[x];
      }

      return Sigmoid(z);
    }

    public double[] TrainForward(double[] input)
    {
      _activation = input;
      _z = new double[_numOutputs];

      for (var x = 0; x < _numOutputs; x++)
      {
        for (var y = 0; y < _numInputs; y++)
        {
          _z[x] += input[y] * Weights[x, y];
        }
        _z[x] += Biases[x];
      }

      return Sigmoid(_z);
    }

    public double[] TrainBackward(double[] input)
    {
      var z = new double[_numInputs];
      for (var y = 0; y < _numInputs; y++)
      {
        for (var x = 0; x < _numOutputs; x++)
        {
          z[y] += input[x] * Weights[x, y];
        }
      }
      return z;
    }

    public double[] UpdateDeltas(double[] cost)
    {
      var zs = SigmoidPrime(_z);
      var delta = new double[cost.Length];

      for (var x = 0; x < _numOutputs; x++)
      {
        delta[x] = cost[x] * zs[x];
        _deltaBiases[x] += delta[x];

        for (var y = 0; y < _numInputs; y++)
        {
          _deltaWeights[x, y] += delta[x] * _activation[y];
        }
      }

      return delta;
    }

    public void ApplyDeltas(double learningRate)
    {
      for (var x = 0; x < _numOutputs; x++)
      {
        Biases[x] -= learningRate * _deltaBiases[x];
        _deltaBiases[x] = 0;
        for (var y = 0; y < _numInputs; y++)
        {
          Weights[x, y] -= learningRate * _deltaWeights[x, y];
          _deltaWeights[x, y] = 0;
        }
      }
    }

    private double[] Sigmoid(double[] z)
    {
      var x = new double[z.Length];
      for (var i = 0; i < z.Length; i++)
      {
        x[i] = 1.0 / (1.0 + Math.Exp(-z[i]));
      }
      return x;
    }

    private double[] SigmoidPrime(double[] z)
    {
      var x = new double[z.Length];
      var sz = Sigmoid(z);

      for (var i = 0; i < z.Length; i++)
      {
        x[i] = sz[i] * (1 - sz[i]);
      }

      return x;
    }

    private void SetInitialWeights()
    {
      for (var x = 0; x < _numOutputs; x++)
      {
        for (var y = 0; y < _numInputs; y++)
        {
          Weights[x, y] = 2.0 * _rng.NextDouble() - 1.0;
        }
      }
    }

    private void SetInitialBiases()
    {
      for (var x = 0; x < _numOutputs; x++)
      {
        Biases[x] = 2.0 * _rng.NextDouble() - 1.0;
      }
    }
  }
}
