package unsri.ac.id.lstm.loss;

public interface LossFunction {
    double call(double[] yTrue, double[] yPred);
    double[] derivative(double[] yTrue, double[] yPred);
}
