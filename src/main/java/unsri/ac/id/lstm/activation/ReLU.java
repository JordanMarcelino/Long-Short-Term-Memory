package unsri.ac.id.lstm.activation;


public class ReLU implements Activation {
    @Override
    public double activate(double input) {
        return Math.max(0, input);
    }
}
