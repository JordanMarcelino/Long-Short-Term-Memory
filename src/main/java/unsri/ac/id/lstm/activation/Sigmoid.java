package unsri.ac.id.lstm.activation;

public class Sigmoid<T> implements ActivationFunction<T> {
    @Override
    public T activate(T input) {
        if (input instanceof double[]) {
            double[] result = new double[((double[]) input).length];

            for (int i = 0; i < ((double[]) input).length; i++) {
                result[i] = 1.0 / (1.0 + Math.exp(-((double[]) input)[i]));
            }

            return (T) result;
        }

        return null;
    }

    @Override
    public T derivative(T input) {
        if (input instanceof double[]){
            double[] result = (double[]) activate(input);

            for (int i = 0; i < ((double[]) input).length; i++) {
                result[i] = result[i] * (1 - result[i]);
            }

            return (T) result;
        }

        return null;
    }
}
