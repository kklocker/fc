using System.Collections;

using fc.Abstractions;

namespace fc.pc.app;

public class Network
{
    private readonly List<Node> _nodes = [];

    private readonly Matrix2D<double> _weights;

    private readonly IActivationFunction _activationFunction;

    private int _numberOfInputNodes;
    private int _numberOfOutputNodes;

    private bool WeightsAreLocked { get; set; } = false;

    private double LearningRate { get; set; } = 0.5;
    public double TotalEnergy => CurrentTotalEnergy();

    public Network(int numberOfNodes, IActivationFunction activationFunction)
    {
        for (var i = 0; i < numberOfNodes; i++)
        {
            _nodes.Add(new Node());
        }

        _weights = new Matrix2D<double>(numberOfNodes).Apply(_ => Random.Shared.NextDouble());
        _activationFunction = activationFunction; // Default activation function
    }


    public void SetDataDimensions(int numberOfInputNodes, int numberOfOutputNodes) // TODO: builder pattern for the networks? 
    {
        if (numberOfInputNodes <= 0 || numberOfOutputNodes <= 0)
            throw new ArgumentException("Number of input and output nodes must be greater than zero.");

        if (numberOfInputNodes + numberOfOutputNodes > _nodes.Count)
            throw new ArgumentException($"Total number of input and output nodes ({numberOfInputNodes + numberOfOutputNodes}) exceeds the total number of nodes in the network ({_nodes.Count}).");

        _numberOfInputNodes = numberOfInputNodes;
        _numberOfOutputNodes = numberOfOutputNodes;
    }


    public void LockWeights()
    {
        WeightsAreLocked = true;
    }

    public void UnlockOutputActivations()
    {
        for (var i = _nodes.Count - 1; i >= _numberOfOutputNodes; i--)
        {
            var node = _nodes[i];

            node.IsLocked = false;
            node.Activation = 0; // Reset activation
        }
    }

    public void SetAndLockInputActivations(double[] input)
    {

        if (input.Length != _numberOfInputNodes)
            throw new ArgumentException($"Input dimensions do not match the network configuration. Expected {_numberOfInputNodes} inputs, but got {input.Length} inputs.");

        for (var i = 0; i < _numberOfInputNodes; i++)
        {

            var node = _nodes[i];
            node.Activation = input[i];
            node.IsLocked = true;
            node.Error = 0;
        }
    }

    /// <summary>
    /// Locks the activations of the input and output nodes, preventing them from being updated during the training process.
    /// </summary>
    public void SetAndLockOutputActivations(double[] output)
    {
        if (output.Length != _numberOfOutputNodes)
            throw new ArgumentException($"Output dimensions do not match the network configuration. Expected {_numberOfOutputNodes} outputs, but got {output.Length} outputs.");

        for (var i = 0; i < _numberOfOutputNodes; i++)
        {
            var node = _nodes[_nodes.Count - 1 - i];

            node.Activation = output[i];
            node.IsLocked = true;
            node.Error = 0;
        }

    }

    public void Next()
    {

        UpdateActivations();
        UpdateWeights();
        UpdatePredictionErrors();
    }

    private void UpdateWeights()
    {
        if (WeightsAreLocked)
            return;

        for (var i = 0; i < _nodes.Count; i++)
        {
            var node_i = _nodes[i];
            var activation_i = _activationFunction.Activate(node_i);
            for (var j = 0; j < _nodes.Count; j++)
            {
                if (i == j)
                    continue;

                var nextWeight = _weights[i, j] + (LearningRate * _nodes[j].Error * activation_i);
                _weights[i, j] = nextWeight;

            }
        }
    }

    private void UpdatePredictionErrors()
    {
        for (var i = 0; i < _nodes.Count; i++)
        {
            var node_i = _nodes[i];
            var activation_i = node_i.Activation;
            var prediction_i = 0d;

            for (var j = 0; j < _nodes.Count; j++)
            {
                if (i == j)
                    continue;

                var node_j = _nodes[j];
                var activate_j = _activationFunction.Activate(node_j);
                var weight_ij = _weights[i, j];
                prediction_i += weight_ij * activate_j;
            }

            node_i.Error = activation_i - prediction_i;

        }
    }

    private void UpdateActivations()
    {

        for (var i = 0; i < _nodes.Count; i++)
        {
            var node_i = _nodes[i];

            if (node_i.IsLocked)
                continue; // Skip locked nodes

            var error_i = node_i.Error;
            var activation_i = node_i.Activation;

            var nextActivation = activation_i - LearningRate * error_i;
            var derivative_i = _activationFunction.Derivative(node_i);

            for (var j = 0; j < _nodes.Count; j++)
            {
                if (i == j)
                    continue;

                var weight_ji = _weights[j, i];
                var node_j = _nodes[j];
                nextActivation += LearningRate * node_j.Error * weight_ji * derivative_i;
            }

            node_i.Activation = nextActivation;
        }
    }

    /// <summary>
    /// The total energy of the network is the sum of the squares of the errors of all nodes.
    /// </summary>
    private double CurrentTotalEnergy() =>
        _nodes.Sum(node => Math.Pow(node.Error, 2));

    public int NumberOfWeightsBelowThreshold(double threshold)
    {
        int count = 0;
        for (var i = 0; i < _weights.Rows; i++)
            for (var j = 0; j < _weights.Columns; j++)
                if (Math.Abs(_weights[i, j]) < threshold)
                    count++;

        return count;
    }

    public double[] GetOutputActivations()
    {
        var result = new double[_numberOfOutputNodes];
        for (var i = 0; i < _numberOfOutputNodes; i++)
        {
            var node = _nodes[_nodes.Count - 1 - i];

            result[i] = node.Activation;
        }
        return result;
    }
}

public class Weight
{
    private double _value;

    public double Value
    {
        get => _value;
        set
        {
            if (isLocked)
                throw new InvalidOperationException("Weight is locked and cannot be modified.");
            _value = value;
        }
    }

    private bool isLocked = false;
    public Weight(double value)
    {
        _value = value;
    }

    public void Lock()
    {
        isLocked = true;
    }

    public void Unlock()
    {
        isLocked = false;
    }



}

public class X : IEnumerable<X>
{
    public IEnumerator<X> GetEnumerator()
    {
        throw new NotImplementedException();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}