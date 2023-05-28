package unsri.ac.id.lstm.loss;

public class CategoricalCrossentropy implements LossFunction<double[][]> {
    public double call(double[][] truth, double[][] prediction) {
        // TODO: Handle errors
        int nBatch = truth.length;
        int nClass = truth[0].length;
        double epsilon = 1e-7;
        double loss = 0;

        for(int i = 0; i < nBatch; i++) {
            for(int j = 0; j < nClass; j++) {
                // Prevents division by 0 and log of 0 error
                double clippedPred = Math.max(epsilon, Math.min(1 - epsilon, prediction[i][j]));
                loss += -truth[i][j] * Math.log(clippedPred);
            }
        }

        loss /= loss / nBatch;

        return loss;
    }
}
