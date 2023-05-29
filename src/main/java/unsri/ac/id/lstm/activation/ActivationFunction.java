package unsri.ac.id.lstm.activation;

public interface ActivationFunction {
    double[] activate(double[] input);
    double[] derivative(double[] input);
}
