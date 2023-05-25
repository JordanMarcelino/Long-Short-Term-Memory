package unsri.ac.id.lstm.activation;

public class TanH implements ActivationFunction {
    @Override
    public double[] activate(double[] input) {
        double[] result = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = Math.tanh(input[i]);
        }

        return result;
    }
}
