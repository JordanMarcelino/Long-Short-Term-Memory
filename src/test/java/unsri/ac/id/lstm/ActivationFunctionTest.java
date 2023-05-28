package unsri.ac.id.lstm;

import static org.junit.jupiter.api.Assertions.*;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.activation.*;

import java.util.Arrays;
import java.util.Random;

@Slf4j
public class ActivationFunctionTest {

    private double[] input = new double[10];
    private Random random = new Random();

    @BeforeEach
    void setUp() {
        for (int i = 0; i < this.input.length; i++) {
            this.input[i] = random.nextDouble(-1, 2);
        }

        log.info("Input : " + Arrays.toString(this.input));
    }

    void compare(double a, double b){
        log.info("(Expected - " + a + " ) : (Result - " + b + " )");
    }

    @Test
    void reluTest() {
        ActivationFunction relu = new ReLU();
        double[] expectedResult = new double[10];

        for (int i = 0; i < expectedResult.length; i++) {
            expectedResult[i] = Math.max(0, this.input[i]);
        }

        double[] realResult = relu.activate(this.input);

        for (int i = 0; i < expectedResult.length; i++) {
            assertEquals(expectedResult[i], realResult[i], "Not same");
            compare(expectedResult[i], realResult[i]);
        }
    }

    @Test
    void sigmoidTest() {
        ActivationFunction sigmoid = new Sigmoid();
        double[] expectedResult = new double[10];

        for (int i = 0; i < expectedResult.length; i++) {
            expectedResult[i] = 1.0 / (1.0 + Math.exp(-this.input[i]));
        }

        double[] realResult = sigmoid.activate(this.input);

        for (int i = 0; i < expectedResult.length; i++) {
            assertEquals(expectedResult[i], realResult[i], "Not same");
            compare(expectedResult[i], realResult[i]);
        }
    }

    @Test
    void tanhTest() {
        ActivationFunction tanH = new TanH();
        double[] expectedResult = new double[10];

        for (int i = 0; i < expectedResult.length; i++) {
            expectedResult[i] = Math.tanh(this.input[i]);
        }

        double[] realResult = tanH.activate(this.input);

        for (int i = 0; i < expectedResult.length; i++) {
            assertEquals(expectedResult[i], realResult[i], "Not same");
            compare(expectedResult[i], realResult[i]);
        }
    }

    @Test
    void softMaxTest() {
        ActivationFunction softMax = new SoftMax();
        double[] expectedResult = new double[10];

        for (int i = 0; i < expectedResult.length; i++) {
            expectedResult[i] = Math.exp(this.input[i]) / Arrays.stream(this.input).map(Math::exp).sum();
        }

        double[] realResult = softMax.activate(this.input);

        for (int i = 0; i < expectedResult.length; i++) {
            assertEquals(expectedResult[i], realResult[i], "Not same");
            compare(expectedResult[i], realResult[i]);
        }
    }


}
