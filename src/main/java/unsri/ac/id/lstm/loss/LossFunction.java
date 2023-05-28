package unsri.ac.id.lstm.loss;

public interface LossFunction<T> {
    double call(T truth, T prediction);
}
