import evaluate
from evaluate.utils import launch_gradio_widget

module = evaluate.load("creative-graphic-design/layout-occlusion")
launch_gradio_widget(module)
