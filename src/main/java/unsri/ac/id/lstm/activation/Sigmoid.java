package unsri.ac.id.lstm.activation;

public class Sigmoid implements ActivationFunction {
    @Override
    public double[] activate(double[] input) {
        double[] result = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = 1.0 / (1.0 + Math.exp(-input[i]));
        }

        return result;
    }

    @Override
    public double[] derivative(double[] input) {
        double[] result = activate(input);

        for (int i = 0; i < result.length; i++) {
            result[i] = result[i] * (1 - result[i]);
        }

        return result;
    }
}
