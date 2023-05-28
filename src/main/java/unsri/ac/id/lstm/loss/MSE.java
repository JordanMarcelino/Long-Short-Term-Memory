package unsri.ac.id.lstm.loss;

public class MSE implements LossFunction<double[]>{
    @Override
    public double call(double[] truth, double[] prediction) {
        // TODO: Handle errors
        int n = truth.length;
        double sumSquaredErrors = 0;

        for(int i = 0; i < n; i++) {
            sumSquaredErrors += Math.pow(truth[i] - prediction[i], 2);
        }

        double loss = sumSquaredErrors / n;

        return loss;
    }
}
