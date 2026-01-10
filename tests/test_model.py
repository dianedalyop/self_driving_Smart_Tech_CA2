import numpy as np
from tensorflow.keras.models import load_model


# confirms that the trained model can load successfully and make a valid steering angle prediction for correctly shaped input
def test_model_prediction_shape():
    model = load_model("nvidia_elu_augmented.keras")
    dummy_input = np.random.rand(1,66,200,3)
    prediction = model.predict(dummy_input)
    assert prediction.shape == (1,1)
