package unsri.ac.id.lstm.model;

import lombok.Data;
import unsri.ac.id.lstm.layers.Layer;

import java.util.ArrayList;
import java.util.List;

@Data
public class Sequential implements Model {
    private List<Layer> layers;

    public Sequential() {
        this.layers = new ArrayList<>();
    }

    @Override
    public void add(Layer... layers) {
        this.layers.addAll(List.of(layers));
    }

    @Override
    public void forward() {

    }

    @Override
    public void backPropagation() {

    }

    @Override
    public void updateMiniBatch() {

    }

    @Override
    public void summary() {

    }

    @Override
    public void compile() {

    }

    @Override
    public void fit(double[][] xTrain, double[] yTrain, int batchSize, int epochs, double validationSplit) {

    }
}
