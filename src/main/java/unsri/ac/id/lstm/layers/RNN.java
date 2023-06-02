package unsri.ac.id.lstm.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import unsri.ac.id.lstm.activation.ActivationFunction;
import unsri.ac.id.lstm.activation.TanH;
import unsri.ac.id.lstm.initialization.InitializationFunction;
import unsri.ac.id.lstm.initialization.XavierGlorotInitializer;
import unsri.ac.id.lstm.utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;

@EqualsAndHashCode(callSuper = true)
@Data
public class RNN<T> extends Layer<T> {
    private double[][] weightsHidden;
    private double[][] weightsOutput;
    private double[] biasesOutput;
    private double[][] output;
    private double[][] hidden;

    public RNN(int nInput, int nHidden, int nOutput, ActivationFunction<T> activationFunction,
               InitializationFunction initializationFunction) {
        this.biases = new double[nHidden];
        this.biasesOutput = new double[nOutput];
        this.activationFunction = activationFunction;
        this.initializationFunction = initializationFunction;
        this.weights = initializationFunction.initialize(nInput, nHidden);
        this.weightsHidden = initializationFunction.initialize(nHidden, nHidden);
        this.weightsOutput = initializationFunction.initialize(nHidden, nOutput);
    }

    public RNN(int nInput, int nHidden, int nOutput, ActivationFunction<T> activationFunction) {
        this.biases = new double[nHidden];
        this.biasesOutput = new double[nOutput];
        this.activationFunction = activationFunction;
        this.initializationFunction = new XavierGlorotInitializer();
        this.weights = initializationFunction.initialize(nInput, nHidden);
        this.weightsHidden = initializationFunction.initialize(nHidden, nHidden);
        this.weightsOutput = initializationFunction.initialize(nHidden, nOutput);
    }

    public RNN(int nInput, int nHidden, int nOutput) {
        this.biases = new double[nHidden];
        this.biasesOutput = new double[nOutput];
        this.activationFunction = new TanH<>();
        this.initializationFunction = new XavierGlorotInitializer();
        this.weights = initializationFunction.initialize(nInput, nHidden);
        this.weightsHidden = initializationFunction.initialize(nHidden, nHidden);
        this.weightsOutput = initializationFunction.initialize(nHidden, nOutput);
    }

    @Override
    public void forward(T inputs) {
        if (inputs instanceof double[]) {
            this.hidden = new double[((double[]) inputs).length][weights[0].length];
            this.output = new double[((double[]) inputs).length][weightsOutput[0].length];

            for (int i = 0; i < ((double[]) inputs).length; i++) {
                double[][] xT = Utils.dotProduct(Utils.transpose(((double[]) inputs)[i]), weights);

                double[][] hT = Utils.add(
                        Utils.add(
                                xT,
                                Utils.dotProduct(
                                        Utils.transpose(hidden[Math.max(0, i - 1)]),
                                        weightsHidden
                                )
                        ),
                        biases
                );
                hT = (double[][]) activationFunction.activate((T) hT);

                hidden[i] = Arrays.stream(hT).flatMapToDouble(Arrays::stream).toArray();

                double[][] oT = Utils.add(
                        Utils.dotProduct(
                                hT,
                                weightsOutput
                        ),
                        biasesOutput
                );

                output[i] = Arrays.stream(oT).flatMapToDouble(Arrays::stream).toArray();
            }
        } else if (inputs instanceof double[][]) {
            this.hidden = new double[((double[][]) inputs).length][weights[0].length];
            this.output = new double[((double[][]) inputs).length][weightsOutput[0].length];

            for (int i = 0; i < ((double[][]) inputs).length; i++) {
                double[][] xT = Utils.dotProduct(Utils.transpose(((double[][]) inputs)[i]), weights);

                double[][] hT = Utils.add(
                        Utils.add(
                                xT,
                                Utils.dotProduct(
                                        Utils.transpose(hidden[Math.max(0, i - 1)]),
                                        weightsHidden
                                )
                        ),
                        biases
                );
                hT = (double[][]) activationFunction.activate((T) hT);

                hidden[i] = Arrays.stream(hT).flatMapToDouble(Arrays::stream).toArray();

                double[][] oT = Utils.add(
                        Utils.dotProduct(
                                hT,
                                weightsOutput
                        ),
                        biasesOutput
                );

                output[i] = Arrays.stream(oT).flatMapToDouble(Arrays::stream).toArray();
            }

        }
    }
}
