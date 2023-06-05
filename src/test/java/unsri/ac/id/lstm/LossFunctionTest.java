package unsri.ac.id.lstm;

import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.loss.LossFunction;
import unsri.ac.id.lstm.loss.MSE;

import java.util.Arrays;

public class LossFunctionTest {

    @Test
    void testMSE() {
        LossFunction mse = new MSE();
        double[] yTrue = {60.0, 70.2, 72.3};
        double[] yPred = {58.2, 66.3, 70.4};

        System.out.println(mse.call(yTrue, yPred));

        System.out.println(Arrays.toString(mse.derivative(yTrue, yPred)));
        System.out.println(Arrays.stream(mse.derivative(yTrue, yPred)).sum());
    }
}
