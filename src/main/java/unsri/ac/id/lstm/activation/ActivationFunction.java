package unsri.ac.id.lstm.activation;

public interface ActivationFunction<T> {
    T activate(T input);
    T derivative(T input);
}
