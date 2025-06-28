using System.Collections;

using fc.Abstractions;

namespace fc.pc.app;

public class Network
{
    private readonly List<Node> _nodes = [];

    private readonly Matrix2D<double> _weights;

    private readonly IActivationFunction _activationFunction;

    private bool WeightsAreLocked { get; set; } = false;

    private double LearningRate { get; set; } = 0.9;
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