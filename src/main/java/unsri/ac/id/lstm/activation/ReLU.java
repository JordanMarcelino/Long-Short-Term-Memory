package unsri.ac.id.lstm.activation;


public class ReLU<T> implements ActivationFunction<T> {
    @Override
    public T activate(T input) {
        if (input instanceof double[]){
            double[] result = new double[((double[]) input).length];
            for (int i = 0; i < ((double[]) input).length; i++) {
                result[i] = Math.max(0, ((double[]) input)[i]);
            }

            return (T) result;
        }


        return null;
    }

    @Override
    public T derivative(T input) {
        if (input instanceof double[]){
            double[] result = new double[((double[]) input).length];

            for (int i = 0; i < ((double[]) input).length; i++) {
                result[i] = ((double[]) input)[i] > 0.0 ? 1.0 : 0.0;
            }

            return (T) result;
        }

        return null;
    }
}
