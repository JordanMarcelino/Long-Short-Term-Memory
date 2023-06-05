package unsri.ac.id.lstm.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import unsri.ac.id.lstm.activation.ActivationFunction;
import unsri.ac.id.lstm.activation.TanH;
import unsri.ac.id.lstm.initialization.InitializationFunction;
import unsri.ac.id.lstm.initialization.XavierGlorotInitializer;
import unsri.ac.id.lstm.utils.Utils;

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

    public RNN(int nInput, int nHidden, int nOutput, InitializationFunction initializationFunction) {
        this.biases = new double[nHidden];
        this.biasesOutput = new double[nOutput];
        this.activationFunction = new TanH<>();
        this.initializationFunction = initializationFunction;
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
                forwardCell(Utils.transpose(((double[]) inputs)[i]), i);
            }
        } else if (inputs instanceof double[][]) {
            this.hidden = new double[((double[][]) inputs).length][weights[0].length];
            this.output = new double[((double[][]) inputs).length][weightsOutput[0].length];

            for (int i = 0; i < ((double[][]) inputs).length; i++) {
                forwardCell(Utils.transpose(((double[][]) inputs)[i]), i);
            }

        }
    }

    private void forwardCell(double[][] input, int timeStep) {
        double[][] xT = Utils.dotProduct(input, weights);

        double[][] hT = Utils.add(
                Utils.add(
                        xT,
                        Utils.dotProduct(
                                Utils.transpose(hidden[Math.max(0, timeStep - 1)]),
                                weightsHidden
                        )
                ),
                biases
        );
        hT = (double[][]) activationFunction.activate((T) hT);

        hidden[timeStep] = Arrays.stream(hT).flatMapToDouble(Arrays::stream).toArray();

        double[][] oT = Utils.add(
                Utils.dotProduct(
                        hT,
                        weightsOutput
                ),
                biasesOutput
        );

        output[timeStep] = Arrays.stream(oT).flatMapToDouble(Arrays::stream).toArray();
    }

    public void backward(T input, double[] loss) {
        double lr = 1e-5;

        double[][] nextHGrad = new double[1][weightsHidden.length];

        double[][] iWeightGrad = new double[weights.length][weights[0].length];
        double[][] hWeightGrad = new double[weightsHidden.length][weightsHidden.length];
        double[][] hBiasGrad = new double[1][weightsHidden.length];
        double[][] oWeightGrad = new double[weightsOutput.length][weightsOutput[0].length];
        double[][] oBiasGrad = new double[1][weightsOutput[0].length];
        double[][] one = new double[1][weightsHidden.length];

        Arrays.fill(one[0], 1);

        if (input instanceof double[]) {
            for (int i = 0; i < ((double[]) input).length; i++) {

            }
            lr /= ((double[]) input).length;
        } else if (input instanceof double[][]) {
            for (int i = ((double[][]) input).length - 1; i > -1; i--) {
                double[][] outGrad = Utils.transpose(loss[i]);

                oWeightGrad = Utils.add(
                        oWeightGrad,
                        Utils.dotProduct(
                                Utils.transpose(
                                        Utils.transpose(hidden[i])
                                ),
                                outGrad
                        )
                );

                oBiasGrad = Utils.add(
                        oBiasGrad,
                        outGrad
                );


                double[][] hGrad = Utils.dotProduct(
                        outGrad,
                        Utils.transpose(weightsOutput)
                );

                if (i < ((double[][]) input).length - 1) {
                    double[][] hhGrad = Utils.dotProduct(
                            nextHGrad,
                            Utils.transpose(weightsHidden)
                    );

                    hGrad = Utils.add(
                            hGrad,
                            hhGrad
                    );
                }

                double[][] tanHDeriv = Utils.subtract(
                        one,
                        Utils.multiply(
                                Utils.transpose(hidden[i]),
                                Utils.transpose(hidden[i])
                        )
                );

                hGrad = Utils.multiply(
                        hGrad,
                        tanHDeriv
                );

                nextHGrad = hGrad.clone();

                if (i > 0) {
                    hWeightGrad = Utils.add(
                            hWeightGrad,
                            Utils.dotProduct(
                                    Utils.transpose(
                                            Utils.transpose(hidden[i - 1])
                                    ),
                                    hGrad
                            )
                    );

                    hBiasGrad = Utils.add(
                            hBiasGrad,
                            hGrad
                    );
                }


                iWeightGrad = Utils.add(
                        iWeightGrad,
                        Utils.dotProduct(
                                Utils.transpose(
                                        Utils.transpose(((double[][]) input)[i])
                                ),
                                hGrad
                        )
                );
            }

            lr /= ((double[][]) input).length;
        }


        weights = Utils.subtract(
                weights,
                Utils.multiply(
                        iWeightGrad,
                        lr
                )
        );
        weightsHidden = Utils.subtract(
                weightsHidden,
                Utils.multiply(
                        hWeightGrad,
                        lr
                )
        );
        biases = Utils.subtract(
                biases,
                Arrays.stream(
                        Utils.multiply(
                                hBiasGrad,
                                lr
                        )
                ).flatMapToDouble(Arrays::stream).toArray()
        );
        weightsOutput = Utils.subtract(
                weightsOutput,
                Utils.multiply(
                        oWeightGrad,
                        lr
                )
        );
        biasesOutput = Utils.subtract(
                biasesOutput,
                Arrays.stream(
                        Utils.multiply(
                                oBiasGrad,
                                lr
                        )
                ).flatMapToDouble(Arrays::stream).toArray()
        );
    }

    void backwardCell() {

    }
}
