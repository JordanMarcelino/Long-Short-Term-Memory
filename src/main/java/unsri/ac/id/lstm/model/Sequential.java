package unsri.ac.id.lstm.model;

import lombok.Data;
import lombok.EqualsAndHashCode;
import unsri.ac.id.lstm.layers.Layer;
import unsri.ac.id.lstm.utils.Utils;

import java.util.ArrayList;

@EqualsAndHashCode(callSuper = true)
@Data
public class Sequential<T> extends Model<T> {

    public Sequential() {
        this.layers = new ArrayList<>();
    }

    @Override
    public void add(Layer<T> layer) {
        this.layers.add(layer);
    }

    @Override
    public void forward(T input) {
        for (int i = 0; i < this.layers.size(); i++) {
            if (i == 0) {
                this.layers.get(i).forward(input);
            } else {
                Layer<T> prevLayer = this.layers.get(i - 1);
                Layer<T> currentLayer = this.layers.get(i);
                double[] prevOutputLen = (double[]) prevLayer.getOutputBeforeActivation();

                currentLayer.setWeights(
                        currentLayer.getInitializationFunction().initialize(
                                currentLayer.getBiases().length,
                                prevOutputLen.length)
                );
                currentLayer.forward(prevLayer.getOutputAfterActivation());
            }
        }
    }

    @Override
    public void backPropagation(double[] x, double[] y) {
        double[] delta = Utils.multiplyElementWise(
                getLossFunction().derivative((double[]) layers.get(layers.size() - 1).getOutputAfterActivation(), y),
                (double[]) layers.get(layers.size() - 1).getActivationFunction().derivative(layers.get(layers.size() - 1).getOutputBeforeActivation())
        );


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
