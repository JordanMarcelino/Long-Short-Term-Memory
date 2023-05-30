package unsri.ac.id.lstm;

import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.initialization.*;

import java.util.Arrays;

public class InitializerTest {

    @Test
    void testRandomInitializer() {
        InitializationFunction randomInitializer = new RandomInitializer();
        double[][] initialize = randomInitializer.initialize(300, 3);

        System.out.println(Arrays.deepToString(initialize));
    }

    @Test
    void textXavierGlorotInitializer() {
        InitializationFunction xavierGlorotInitializer = new XavierGlorotInitializer();
        double[][] initialize = xavierGlorotInitializer.initialize(300, 3);

        System.out.println(Arrays.deepToString(initialize));
    }

    @Test
    void testUniformXavierGlorotInitializer() {
        InitializationFunction uniformXavierGlorotInitializer = new UniformXavierGlorotInitializer();
        double[][] initialize = uniformXavierGlorotInitializer.initialize(300, 3);

        System.out.println(Arrays.deepToString(initialize));
    }

    @Test
    void testHeInitializer() {
        InitializationFunction heInitializer = new HeInitializer();
        double[][] initialize = heInitializer.initialize(300, 3);

        System.out.println(Arrays.deepToString(initialize));
    }
}
