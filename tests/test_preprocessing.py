import numpy as np
import pytest
from src.data_preprocess import preprocess_image


#verifies that the image preprocessing produces images in the correct shape for the model.
def test_preprocess_image_shape():
    dummy_image = np.random.randint(0, 255,(160, 320, 3), dtype=np.uint8)
    processed = preprocess_image(dummy_image)
    assert processed.shape == (66, 200, 3)



#ensures the preprocessing function fails safely when given invalid input (missing image)
def test_preprocess_image_invalid_input():
    with pytest.raises(ValueError):
        preprocess_image(None)
