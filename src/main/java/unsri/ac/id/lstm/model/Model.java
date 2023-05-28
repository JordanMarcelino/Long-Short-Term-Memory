package unsri.ac.id.lstm.model;

import unsri.ac.id.lstm.layers.Layer;

public interface Model {
    void add(Layer... layers);

    void summary();

    void compile();

    void fit(double[][] xTrain, double[] yTrain, int batchSize, int epochs, double validationSplit);
}
