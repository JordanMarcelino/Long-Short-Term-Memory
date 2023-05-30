package unsri.ac.id.lstm.model;

import lombok.Data;
import unsri.ac.id.lstm.layers.Layer;

import java.util.List;

@Data
public abstract class Model<T> {
    protected List<Layer<T>> layers;

    public abstract void add(Layer<T> layer);

    public abstract void forward(T input);

    public abstract void backPropagation();

    public abstract void updateMiniBatch();

    public abstract void summary();

    public abstract void compile();

    public abstract void fit(double[][] xTrain, double[] yTrain, int batchSize, int epochs, double validationSplit);
}
