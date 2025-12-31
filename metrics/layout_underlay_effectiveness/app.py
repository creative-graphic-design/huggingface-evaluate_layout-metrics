import evaluate
from evaluate.utils import launch_gradio_widget

module = evaluate.load("creative-graphic-design/layout-underlay-effectiveness")
launch_gradio_widget(module)
