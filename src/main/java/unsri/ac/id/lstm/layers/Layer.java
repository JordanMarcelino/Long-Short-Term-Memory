package unsri.ac.id.lstm.layers;

public interface Layer {
    void forward(double[] inputs);

    double[] dotProduct(double[] a, double[][] b);

    double[] add(double[] a, double[] b);
}
