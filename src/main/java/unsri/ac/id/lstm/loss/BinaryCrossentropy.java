package unsri.ac.id.lstm.loss;

public class BinaryCrossentropy implements LossFunction {
    @Override
    public double call(double[] yTrue, double[] yPred) {
        // TODO: Handle errors
        int n = yTrue.length;
        double epsilon = 1e-7;
        double loss = 0;

        for (int i = 0; i < n; i++) {
            // Prevents division by 0 and log of 0 error
            double clippedPred = Math.max(epsilon, Math.min(1 - epsilon, yPred[i]));
            loss += -(yTrue[i] * Math.log(clippedPred) + (1 - yTrue[i]) * Math.log(1 - clippedPred));
        }

        loss = loss / n;

        return loss;
    }

    public double[] derivative(double[] yTrue, double[] yPred) {
        int n = yTrue.length;
        double[] derivative = new double[n];
    
        for (int i = 0; i < n; i++) {
            derivative[i] = (yPred[i] - yTrue[i]) / (yPred[i] * (1 - yPred[i]));
        }
    
        return derivative;
    }
}
