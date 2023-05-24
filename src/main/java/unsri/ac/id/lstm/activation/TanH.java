package unsri.ac.id.lstm.activation;

public class TanH implements Activation {
    @Override
    public double activate(double input) {
        return Math.tanh(input);
    }
}
