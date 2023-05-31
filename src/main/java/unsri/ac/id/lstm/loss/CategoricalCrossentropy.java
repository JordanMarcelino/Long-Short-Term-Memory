package unsri.ac.id.lstm.loss;

public class CategoricalCrossentropy implements LossFunction {
    // public double call(double[][] yTrue, double[][] yPred) {
    //     // TODO: Handle errors
    //     int nBatch = yTrue.length;
    //     int nClass = yTrue[0].length;
    //     double epsilon = 1e-7;
    //     double loss = 0;

    //     for(int i = 0; i < nBatch; i++) {
    //         for(int j = 0; j < nClass; j++) {
    //             // Prevents division by 0 and log of 0 error
    //             double clippedPred = Math.max(epsilon, Math.min(1 - epsilon, yPred[i][j]));
    //             loss += -yTrue[i][j] * Math.log(clippedPred);
    //         }
    //     }

    //     loss /= loss / nBatch;

    //     return loss;
    // }

    @Override
    public double call(double[] yTrue, double[] yPred) {
        return 0;
    }

    @Override
    public double[] derivative(double[] yTrue, double[] yPred) {
        return null;
    }
}
