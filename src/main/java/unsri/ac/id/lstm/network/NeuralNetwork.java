package unsri.ac.id.lstm.network;

public interface NeuralNetwork {
    void forward(double[][] input);
    void backProp();
    void gradientDescent();
    double dotProduct(double[][] input1, double[][] input2);
}
