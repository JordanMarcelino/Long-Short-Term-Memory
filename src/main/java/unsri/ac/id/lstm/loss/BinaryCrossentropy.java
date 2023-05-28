package unsri.ac.id.lstm.loss;

public class BinaryCrossentropy implements LossFunction<double[]> {
    @Override
    public double call(double[] truth, double[] prediction) {
        // TODO: Handle errors
        int n = truth.length;
        double epsilon = 1e-7;
        double loss = 0;

        for (int i = 0; i < n; i++) {
            // Prevents division by 0 and log of 0 error
            double clippedPred = Math.max(epsilon, Math.min(1 - epsilon, prediction[i]));
            loss += -(truth[i] * Math.log(clippedPred) + (1 - truth[i]) * Math.log(1 - clippedPred));
        }

        loss = loss / n;

        return loss;
    }
}
