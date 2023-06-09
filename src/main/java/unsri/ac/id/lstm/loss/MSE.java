package unsri.ac.id.lstm.loss;

public class MSE implements LossFunction{
    @Override
    public double call(double[] yTrue, double[] yPred) {
        // TODO: Handle errors
        int n = yTrue.length;
        double sumSquaredErrors = 0;

        for(int i = 0; i < n; i++) {
            sumSquaredErrors += Math.pow(yTrue[i] - yPred[i], 2);
        }

        return sumSquaredErrors / n;
    }

    public double[] derivative(double[] yTrue, double[] yPred) {
        int n = yTrue.length;
        double[] derivative = new double[n];

        for (int i = 0; i < n; i++) {
            derivative[i] = 2 * (yPred[i] - yTrue[i]) / n;
        }
    
        return derivative;
    }
}
