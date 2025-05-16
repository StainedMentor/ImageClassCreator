import torch
from constants import SIZE
import coremltools as ct
from model import *

labels = ['Beech', 'Birch', 'Oak', 'Other', 'Pine', 'Spruce']

# The model pth and definition need to be changed here. Also size needs to match the models input size
model = CustomCNN(num_classes=len(labels))
model.load_state_dict(torch.load("metal_optimized_model.pth", map_location="cpu"))
model.eval()

example_input = torch.rand(1, 3, *SIZE)
traced = torch.jit.trace(model, example_input)

classifier_config = ct.ClassifierConfig(class_labels=labels)

coreml_model = ct.convert(
    traced,
    inputs=[ct.ImageType(name="input", shape=(1, 3, *SIZE), scale=1/255.0)],
    classifier_config=classifier_config,
    convert_to="mlprogram",
)

coreml_model.save("ImageClassifier.mlpackage")