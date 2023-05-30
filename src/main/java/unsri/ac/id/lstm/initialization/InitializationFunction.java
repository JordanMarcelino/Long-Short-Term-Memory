package unsri.ac.id.lstm.initialization;

public interface InitializationFunction {
    double[][] initialize(int rows, int cols);
}
