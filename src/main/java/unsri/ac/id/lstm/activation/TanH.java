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

    @Override
    public double[] derivative(double[] input) {
        double[] result = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = 1.0 - (Math.pow(Math.tanh(input[i]), 2));
        }

        return result;
    }
}
