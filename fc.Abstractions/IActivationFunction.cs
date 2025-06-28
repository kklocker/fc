namespace fc.Abstractions;

public interface IActivationFunction
{
    double Activate(INode node);
    double Derivative(INode node);
}

public class SigmoidActivationFunction : IActivationFunction
{
    public double Activate(INode node)
    {
        return 1.0 / (1.0 + Math.Exp(-node.Activation));
    }
    public double Derivative(INode node)
    {
        var activation = Activate(node);
        return activation * (1d - activation);
    }
}

public class ReLUActivationFunction : IActivationFunction
{
    public double Activate(INode node)
    {
        return Math.Max(0, node.Activation);
    }

    public double Derivative(INode node)
    {
        return node.Activation > 0 ? 1.0 : 0.0;
    }
}