package unsri.ac.id.lstm.loss;

public interface LossFunction<T> {
    double call(T yTrue, T yPred);
}
